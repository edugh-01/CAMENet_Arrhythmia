# In[1]:


# merge  all the files which contains extracted features from scalogram MITBIH
import pandas as pd
import os
import glob

# Folder containing csv files
# folder_path = r"\Data\MIT_BIH\MIT_BIH_Scalogram_Extracted_Features"    # change to your folder path
folder_path = r"\Data\MIT_BIH\MIT_BIH_Scalogram1_Extracted_Features"
folder_path1 = r"\Data\MIT_BIH"

# Output file
output_file = os.path.join(folder_path1, "merged_mitbih_scalogram1_extracted_features.csv")

# Get all csv files
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

first_file = True

for file in csv_files:
    print("Processing:", file)

    if first_file:
        # Write first file with header
        df = pd.read_csv(file)
        df.to_csv(output_file, mode='w', index=False)
        first_file = False
    else:
        # Append remaining files without header
        df = pd.read_csv(file)
        df.to_csv(output_file, mode='a', index=False, header=False)

print("All files merged successfully!")
print("Saved to:", output_file)


# In[3]:


# merge  all the files which contains extracted features from curvelet MITBIH
import pandas as pd
import os
import glob

# Folder containing csv files
folder_path = r"\Data\MIT_BIH\MIT_BIH_curvelet_Extracted_Features"    # change to your folder path
folder_path1 = r"\Data\MIT_BIH"

# Output file
output_file = os.path.join(folder_path1, "merged_mitbih_curvlet_extracted_features.csv")

# Get all csv files
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

first_file = True

for file in csv_files:
    print("Processing:", file)

    if first_file:
        # Write first file with header
        df = pd.read_csv(file)
        df.to_csv(output_file, mode='w', index=False)
        first_file = False
    else:
        # Append remaining files without header
        df = pd.read_csv(file)
        df.to_csv(output_file, mode='a', index=False, header=False)

print("All files merged successfully!")
print("Saved to:", output_file)





