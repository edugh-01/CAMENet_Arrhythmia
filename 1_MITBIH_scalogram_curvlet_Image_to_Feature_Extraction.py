

# List Of LLM based feature extraction: Energy Concentration, Frequency Distribution, Temporal Localization, Symmetry and Smooth Decay, Energy moments (mean, variance, skewness), Radial energy distribution, Gabor-based concentration metrics, Multi-band energy ratios, Local energy window statistics. Conversion Image to LLM embedding:convert Energy Concentration into a numerical feature vector.

# In[2]:


import numpy as np
import cv2

def extract_energy_features(image_path):

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Normalize to [0,1]
    img = img / 255.0

    # Energy map (squared magnitude)
    energy = img ** 2

    total_energy = np.sum(energy)

    # -----------------------------
    # 1. Global Energy
    # -----------------------------
    global_energy = total_energy

    # -----------------------------
    # 2. Temporal Energy Distribution
    # -----------------------------
    temporal_energy = np.sum(energy, axis=0)   # sum over scales
    temporal_center = np.argmax(temporal_energy) / energy.shape[1]

    # -----------------------------
    # 3. Scale (Frequency) Energy Distribution
    # -----------------------------
    scale_energy = np.sum(energy, axis=1)      # sum over time
    dominant_scale = np.argmax(scale_energy) / energy.shape[0]

    # -----------------------------
    # 4. Energy Entropy
    # -----------------------------
    prob = energy / (total_energy + 1e-8)
    entropy = -np.sum(prob * np.log(prob + 1e-8))

    # -----------------------------
    # 5. Energy Concentration Ratio
    # -----------------------------
    max_energy = np.max(energy)
    concentration_ratio = max_energy / (total_energy + 1e-8)

    # -----------------------------
    # Final Feature Vector
    # -----------------------------
    feature_vector = np.array([
        global_energy,
        temporal_center,
        dominant_scale,
        entropy,
        concentration_ratio
    ])

    return feature_vector



# In[3]:


########### convert Frequency Distribution into a numerical feature vector.

import numpy as np
import cv2

def extract_frequency_features(image_path, num_bands=8):

    # Load scalogram as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = img / 255.0

    # Energy map
    energy = img ** 2

    # -----------------------------------
    # 1. Frequency Energy Distribution
    # -----------------------------------
    freq_energy = np.sum(energy, axis=1)  # sum over time

    total_energy = np.sum(freq_energy)
    freq_prob = freq_energy / (total_energy + 1e-8)

    # -----------------------------------
    # 2. Statistical Descriptors
    # -----------------------------------
    freq_mean = np.mean(freq_prob)
    freq_std = np.std(freq_prob)
    freq_skew = np.mean((freq_prob - freq_mean)**3) / (freq_std**3 + 1e-8)
    freq_kurt = np.mean((freq_prob - freq_mean)**4) / (freq_std**4 + 1e-8)

    # -----------------------------------
    # 3. Dominant Frequency Index
    # -----------------------------------
    dominant_freq = np.argmax(freq_prob) / len(freq_prob)

    # -----------------------------------
    # 4. Spectral Entropy
    # -----------------------------------
    spectral_entropy = -np.sum(freq_prob * np.log(freq_prob + 1e-8))

    # -----------------------------------
    # 5. Band Energy Ratios
    # -----------------------------------
    band_size = len(freq_prob) // num_bands
    band_ratios = []

    for i in range(num_bands):
        start = i * band_size
        end = (i + 1) * band_size
        band_energy = np.sum(freq_prob[start:end])
        band_ratios.append(band_energy)

    band_ratios = np.array(band_ratios)

    # -----------------------------------
    # Final Frequency Feature Vector
    # -----------------------------------
    feature_vector = np.concatenate([
        [dominant_freq],
        [spectral_entropy],
        [freq_mean, freq_std, freq_skew, freq_kurt],
        band_ratios
    ])

    return feature_vector




# In[4]:


############# convert Temporal Localization into a numerical feature vector.

import numpy as np
import cv2

def extract_temporal_localization_features(image_path):

    # Load grayscale scalogram
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = img / 255.0

    # Energy map
    energy = img ** 2

    # -----------------------------------
    # 1. Temporal Energy Distribution
    # -----------------------------------
    temporal_energy = np.sum(energy, axis=0)

    total_energy = np.sum(temporal_energy)
    prob_t = temporal_energy / (total_energy + 1e-8)

    T = len(prob_t)
    time_axis = np.arange(T)

    # -----------------------------------
    # 2. Temporal Centroid (Center of Mass)
    # -----------------------------------
    centroid = np.sum(time_axis * prob_t) / (T + 1e-8)
    centroid_norm = centroid / T

    # -----------------------------------
    # 3. Temporal Spread (Variance)
    # -----------------------------------
    variance = np.sum(((time_axis - centroid)**2) * prob_t)
    spread = np.sqrt(variance) / T

    # -----------------------------------
    # 4. Temporal Skewness
    # -----------------------------------
    skewness = np.sum(((time_axis - centroid)**3) * prob_t) / (spread**3 + 1e-8)

    # -----------------------------------
    # 5. Temporal Kurtosis
    # -----------------------------------
    kurtosis = np.sum(((time_axis - centroid)**4) * prob_t) / (spread**4 + 1e-8)

    # -----------------------------------
    # 6. Peak Localization
    # -----------------------------------
    peak_index = np.argmax(prob_t)
    peak_position = peak_index / T

    # -----------------------------------
    # 7. Temporal Entropy
    # -----------------------------------
    entropy = -np.sum(prob_t * np.log(prob_t + 1e-8))

    # -----------------------------------
    # Final Feature Vector
    # -----------------------------------
    feature_vector = np.array([
        centroid_norm,
        spread,
        skewness,
        kurtosis,
        peak_position,
        entropy
    ])

    return feature_vector




# In[5]:



import numpy as np
import cv2


# -----------------------------
# Utility: Load Image Properly
# -----------------------------
def load_scalogram(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found or invalid path.")

    # Convert to grayscale if color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32) / 255.0
    return img


# -----------------------------
# 1. Symmetry & Smooth Decay
# -----------------------------
def symmetry_smooth_decay_features(energy):

    temporal_energy = np.sum(energy, axis=0)
    center = np.argmax(temporal_energy)

    left = temporal_energy[:center]
    right = temporal_energy[center+1:]

    min_len = min(len(left), len(right))

    if min_len < 3:
        return np.array([0.0, 0.0])

    left = left[-min_len:]
    right = right[:min_len]

    symmetry = np.corrcoef(left, right[::-1])[0, 1]
    gradient = np.diff(temporal_energy)
    smoothness = np.mean(np.abs(np.diff(gradient)))

    return np.array([symmetry, smoothness])


# -----------------------------
# 2. Energy Moments
# -----------------------------
def energy_moments_features(energy):

    flat = energy.flatten()
    mean = np.mean(flat)
    variance = np.var(flat)
    skewness = np.mean((flat - mean)**3) / (np.std(flat)**3 + 1e-8)

    return np.array([mean, variance, skewness])


# -----------------------------
# 3. Radial Energy Distribution
# -----------------------------
def radial_energy_distribution_features(energy):

    rows, cols = energy.shape
    center_row = np.argmax(np.sum(energy, axis=1))
    center_col = np.argmax(np.sum(energy, axis=0))

    y, x = np.indices((rows, cols))
    radius = np.sqrt((y - center_row)**2 + (x - center_col)**2)

    max_radius = np.max(radius)
    bins = np.linspace(0, max_radius, 6)

    radial_features = []

    for i in range(len(bins)-1):
        mask = (radius >= bins[i]) & (radius < bins[i+1])
        radial_features.append(np.sum(energy[mask]))

    radial_features = np.array(radial_features)
    radial_features /= (np.sum(radial_features) + 1e-8)

    return radial_features


# -----------------------------
# 4. Gabor Concentration
# -----------------------------
def gabor_concentration_features(image):

    kernels = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21,21), 5, theta, 10, 0.5)
        kernels.append(kernel)

    responses = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        responses.append(np.mean(np.abs(filtered)))

    responses = np.array(responses)
    responses /= (np.sum(responses) + 1e-8)

    return responses


# -----------------------------
# 5. Multi-Band Energy Ratios
# -----------------------------
def multi_band_energy_ratios(energy, bands=6):

    scale_energy = np.sum(energy, axis=1)
    total = np.sum(scale_energy) + 1e-8

    band_size = len(scale_energy) // bands
    ratios = []

    for i in range(bands):
        start = i * band_size
        end = (i + 1) * band_size
        ratios.append(np.sum(scale_energy[start:end]) / total)

    return np.array(ratios)


# -----------------------------
# 6. Local Window Statistics
# -----------------------------
def local_energy_window_statistics(energy, window_size=20):

    temporal_energy = np.sum(energy, axis=0)
    T = len(temporal_energy)

    means = []
    stds = []

    for i in range(0, T - window_size, window_size):
        window = temporal_energy[i:i+window_size]
        means.append(np.mean(window))
        stds.append(np.std(window))

    return np.array([
        np.mean(means),
        np.std(means),
        np.mean(stds),
        np.std(stds)
    ])


# -----------------------------
# FINAL FEATURE VECTOR
# -----------------------------
def full_scalogram_feature_vector(image_path):

    image = load_scalogram(image_path)
    energy = image ** 2

    f1 = symmetry_smooth_decay_features(energy)
    f2 = energy_moments_features(energy)
    f3 = radial_energy_distribution_features(energy)
    f4 = gabor_concentration_features(image)
    f5 = multi_band_energy_ratios(energy)
    f6 = local_energy_window_statistics(energy)
    feature_vector1=extract_energy_features(image_path)
    feature_vector2=extract_frequency_features(image_path)
    feature_vector3=extract_temporal_localization_features(image_path)

    # print("\n symmetry_smooth_decay_features dimension", len(f1))
    # print("\n energy_moments_features dimension", len(f2))
    # print("\n radial_energy_distribution_features dimension", len(f3))
    # print("\n gabor_concentration_features dimension", len(f4))
    # print("\n multi_band_energy_ratios dimension", len(f5))
    # print("\n local_energy_window_statistics dimension", len(f6))
    # print("\n energy_features dimension", len(feature_vector1))
    # print("\n frequency_features dimension", len(feature_vector2))
    # print("\n temporal_localization_features dimension", len(feature_vector3))

    return np.concatenate([f1, f2, f3, f4, f5, f6, feature_vector1, feature_vector2, feature_vector3])




# In[6]:


import os
import pandas as pd

# ============================================================
# RECORD LIST
# ============================================================


records = ['100','101','102','103','104','105','106','107','108','109',
'111','112','113','114','115','116','117','118','119','121',
'122','123','124','200','201','202','203','205','207','208',
'209','210','212','213','214','215','217','219','220','221',
'222','223','228','230','231','232','233','234'
]


# ============================================================
# DIRECTORIES
# ============================================================
image_dir = r"\Data\MIT_BIH\MITBIH_Scalogram_250HZ"
# image_dir = r"\Data\MIT_BIH\OUTPUT_curvlet_Image_MITBIH_250Hz"

temp =r"\Data\MIT_BIH\mitbih_ecg_segments_250hz.csv"
mitbih_data = pd.read_csv(temp)

output_dir =r"\Data\MIT_BIH\MIT_BIH_Scalogram1_Extracted_Features"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# CREATE FAST LOOKUP DICTIONARY
# ============================================================
label_dict = dict(zip(mitbih_data.iloc[:,0], mitbih_data.iloc[:,-1]))

# ============================================================
# PROCESS EACH RECORD
# ============================================================
for record in records:

    print(f"Processing record {record}...")

    # --------------------------------------------------------
    # FILTER IMAGE FILES FOR CURRENT RECORD
    # --------------------------------------------------------
    image_files = [
        f for f in os.listdir(image_dir)
        if f.startswith(record) and f.endswith(".png")
    ]

    image_files.sort()

    data = []

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------
    for img_name in image_files:

        image_path = os.path.join(image_dir, img_name)

        # Extract feature vector
        feature_vector = full_scalogram_feature_vector(image_path)

        sample_id = os.path.splitext(img_name)[0]

        # Label lookup
        label = label_dict.get(sample_id, None)

        row = [sample_id, label] + list(feature_vector)

        data.append(row)

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------
    if len(data) == 0:
        print(f"No images found for record {record}")
        continue

    num_features = len(feature_vector)

    columns = ["ID", "Label"] + [f"F{i}" for i in range(num_features)]

    df = pd.DataFrame(data, columns=columns)


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------
    output_path = os.path.join(
        output_dir,
        # f"MITBIH_curvlet_Extracted_Features_{record}.csv"
        f"MITBIH_scalogram_Extracted_Features_{record}.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# 
