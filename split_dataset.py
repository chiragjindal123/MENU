import os
import shutil
import random
import glob

# --- CONFIGURATION ---
SOURCE_DIR = "dataset_combined_all_new"  # Your existing folder
DEST_DIR = "yolo_dataset"            # New folder to be created
SPLIT_RATIO = 0.8                    # 80% Training, 20% Validation

# 1. Create the YOLO folder structure
subdirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
for path in subdirs:
    os.makedirs(os.path.join(DEST_DIR, path), exist_ok=True)

# 2. Get all images from your source
# (Supports .jpg and .png)
image_paths = glob.glob(os.path.join(SOURCE_DIR, "images", "*.*"))
random.shuffle(image_paths)  # Shuffle to mix the menus

# 3. Calculate split index
split_idx = int(len(image_paths) * SPLIT_RATIO)
train_imgs = image_paths[:split_idx]
val_imgs = image_paths[split_idx:]

def copy_data(image_list, split_type):
    print(f"Copying {len(image_list)} files to {split_type}...")
    for img_path in image_list:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # Define paths
        src_img = img_path
        src_label = os.path.join(SOURCE_DIR, "labels", base_name + ".txt")
        
        dst_img = os.path.join(DEST_DIR, split_type, "images", filename)
        dst_label = os.path.join(DEST_DIR, split_type, "labels", base_name + ".txt")
        
        # Copy Image
        shutil.copy(src_img, dst_img)
        
        # Copy Label (only if it exists)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: Label missing for {filename}")

# 4. Execute Copy
copy_data(train_imgs, "train")
copy_data(val_imgs, "val")

print("Done! Your data is ready in the 'yolo_dataset' folder.")