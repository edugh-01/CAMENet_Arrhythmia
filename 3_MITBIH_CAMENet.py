# ============================================================
PIPELINE (SupCon + Attention + ADASYN + Weighted XGB)
# ============================================================

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ---------------------------
# 1. LOAD DATA
# ---------------------------
scalo_df = pd.read_csv(r"C:\Users\pnk_r\Pinaki_work\Data\MIT_BIH\merged_mitbih_scalogram_extracted_features.csv")
curve_df = pd.read_csv(r"C:\Users\pnk_r\Pinaki_work\Data\MIT_BIH\merged_mitbih_curvlet_extracted_features.csv")

# ---------------------------
# 2. FIX CURVELET ID
# ---------------------------
id_col = curve_df.iloc[:, 0].astype(str)
curve_df["patient_id"] = id_col.apply(lambda x: "_".join(x.split("_")[:-1]))
curve_df["label"] = id_col.apply(lambda x: x.split("_")[-1])
curve_df = curve_df.drop(columns=curve_df.columns[0])

# ---------------------------
# 3. RENAME
# ---------------------------
scalo_df.columns = ["patient_id", "label"] + [f"s_{i}" for i in range(scalo_df.shape[1]-2)]
curve_df.columns = [f"c_{i}" for i in range(curve_df.shape[1]-2)] + ["patient_id", "label"]

# ---------------------------
# 4. MERGE
# ---------------------------
df = pd.merge(scalo_df, curve_df, on=["patient_id", "label"])

# ---------------------------
# 5. LABEL MAPPING
# ---------------------------
AAMI_MAP = {
    'N':'N','L':'N','R':'N','e':'N','j':'N',
    'A':'S','a':'S','J':'S','S':'S',
    'V':'V','E':'V',
    'F':'F',
    '/':'Q','f':'Q','Q':'Q','|':'Q'
}

labels = np.array([AAMI_MAP.get(l, 'Q') for l in df["label"].values])
label_to_int = {'N':0,'S':1,'V':2,'F':3,'Q':4}
y = np.array([label_to_int[l] for l in labels])

# ---------------------------
# 6. FEATURES
# ---------------------------
X_scalo = df.filter(regex="^s_").values.astype(float)
X_curve = df.filter(regex="^c_").values.astype(float)

X_scalo = np.nan_to_num(X_scalo)
X_curve = np.nan_to_num(X_curve)

X_scalo = StandardScaler().fit_transform(X_scalo)
X_curve = StandardScaler().fit_transform(X_curve)

# ---------------------------
# 7. SPLIT
# ---------------------------
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X_scalo, X_curve, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 8. DATASET
# ---------------------------
class MultiViewDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

train_loader = DataLoader(MultiViewDataset(X1_train, X2_train, y_train),
                          batch_size=256, shuffle=True)

test_loader = DataLoader(MultiViewDataset(X1_test, X2_test, y_test),
                         batch_size=256)

# ---------------------------
# 9. MODEL
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1), h


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, h1, h2):
        weights = self.attn(torch.cat([h1, h2], dim=1))
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        return w1 * h1 + w2 * h2


class Model(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.enc1 = Encoder(d1)
        self.enc2 = Encoder(d2)
        self.fusion = AttentionFusion(128)

    def forward(self, x1, x2):
        z1, h1 = self.enc1(x1)
        z2, h2 = self.enc2(x2)
        fused = self.fusion(h1, h2)
        return z1, z2, fused


# ---------------------------
# 10. IMPROVED LOSS (CLASS-BALANCED)
# ---------------------------
def supervised_contrastive_loss(z, y, temp=0.07):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp

    mask = torch.eq(y.unsqueeze(1), y.unsqueeze(0)).float().to(z.device)

    class_counts = torch.bincount(y)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[y].unsqueeze(1)

    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

    weighted_mask = mask * sample_weights

    loss = -(weighted_mask * log_prob).sum(dim=1) / weighted_mask.sum(dim=1)
    return loss.mean()


# ---------------------------
# 11. TRAIN
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(X_scalo.shape[1], X_curve.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

print("Training...")

for epoch in range(120):   #  increased epochs
    model.train()
    total_loss = 0

    for x1, x2, yb in train_loader:
        x1, x2, yb = x1.to(device), x2.to(device), yb.to(device)

        z1, z2, fused = model(x1, x2)

        #  fusion-aware loss
        loss = (
            supervised_contrastive_loss(z1, yb) +
            supervised_contrastive_loss(z2, yb) +
            supervised_contrastive_loss(fused, yb)
        ) / 3

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# ---------------------------
# 12. EMBEDDINGS
# ---------------------------
def get_embeddings(loader):
    model.eval()
    Z, Y = [], []

    with torch.no_grad():
        for x1, x2, yb in loader:
            x1, x2 = x1.to(device), x2.to(device)

            _, _, fused = model(x1, x2)

            Z.append(fused.cpu().numpy())
            Y.append(yb.numpy())

    return np.vstack(Z), np.hstack(Y)


Z_train, Y_train = get_embeddings(train_loader)
Z_test, Y_test = get_embeddings(test_loader)

# ---------------------------
# 13. ADASYN (BETTER THAN SMOTE)
# ---------------------------
smote = ADASYN(random_state=42, n_neighbors=5)
Z_train_bal, Y_train_bal = smote.fit_resample(Z_train, Y_train)

# ---------------------------
# 14. MODELS (WEIGHTED XGB)
# ---------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

classes = np.unique(Y_train_bal)
weights = compute_class_weight('balanced', classes=classes, y=Y_train_bal)
class_weights = dict(zip(classes, weights))
sample_weights = np.array([class_weights[y] for y in Y_train_bal])

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss'
)

rf.fit(Z_train_bal, Y_train_bal)
xgb.fit(Z_train_bal, Y_train_bal, sample_weight=sample_weights)

# ---------------------------
# 15. ENSEMBLE (OPTIMIZED)
# ---------------------------
p1 = rf.predict_proba(Z_test)
p2 = xgb.predict_proba(Z_test)

preds = np.argmax(0.25 * p1 + 0.75 * p2, axis=1)



import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

classes = ['N','S', 'V','F','Q']  # order like paper

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(Y_test, preds)

# Reorder CM to match desired class order
label_order = [ 0, 1, 2, 3, 4]  # mapping from your label_to_int
cm = cm[label_order][:, label_order]

# ---------------------------
# METRICS CALCULATION
# ---------------------------
results = []

for i, cls in enumerate(classes):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    support = TP + FN

    acc = (TP + TN) / np.sum(cm)
    sen = TP / (TP + FN) if (TP + FN) > 0 else 0  # sensitivity = recall
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0

    row = list(cm[i]) + [support, acc*100, sen*100, spe*100, pre*100, f1*100]
    results.append(row)

# ---------------------------
# CREATE TABLE
# ---------------------------
columns = ['N','S', 'V','F','Q',
           'Support', 'Acc (%)', 'Sen (%)', 'Spe (%)', 'Pre (%)', 'F1 (%)']

df_final = pd.DataFrame(results, columns=columns, index=classes)

print("\n================ FINAL TABLE (PAPER FORMAT) ================\n")
print(df_final.round(2))

# ---------------------------
# OVERALL METRICS
# ---------------------------
macro_avg = df_final[['Acc (%)','Sen (%)','Spe (%)','Pre (%)','F1 (%)']].mean()

weights = df_final['Support'] / df_final['Support'].sum()
weighted_avg = (df_final[['Acc (%)','Sen (%)','Spe (%)','Pre (%)','F1 (%)']]
                .multiply(weights, axis=0)
                .sum())

print("\n================ OVERALL METRICS ================\n")
print("MACRO AVERAGE:\n", macro_avg.round(2))
print("\nWEIGHTED AVERAGE:\n", weighted_avg.round(2))

