import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -------- CHANGE THESE PATHS --------
metadata_path = r"C:\Users\puvvada sai prakash\Downloads\archive\HAM10000_metadata.csv"

images_folder1 = r"C:\Users\puvvada sai prakash\Downloads\archive\HAM10000_images_part_1"

images_folder2 = r"C:\Users\puvvada sai prakash\Downloads\archive\HAM10000_images_part_2"
output_folder = "dataset"
# -----------------------------------

metadata = pd.read_csv(metadata_path)

metadata["label"] = metadata["dx"].apply(
    lambda x: "malignant" if x in ["mel", "bcc"] else "benign"
)

train_df, test_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=metadata["label"]
)

for folder in [
    "dataset/train/benign",
    "dataset/train/malignant",
    "dataset/test/benign",
    "dataset/test/malignant",
]:
    os.makedirs(folder, exist_ok=True)

def copy_images(df, split):

    for _, row in df.iterrows():

        filename = row["image_id"] + ".jpg"

        src1 = os.path.join(images_folder1, filename)
        src2 = os.path.join(images_folder2, filename)

        if os.path.exists(src1):
            src = src1
        else:
            src = src2

        dst = os.path.join(
            output_folder,
            split,
            row["label"],
            filename,
        )

        shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(test_df, "test")

print("Dataset Prepared Successfully!")