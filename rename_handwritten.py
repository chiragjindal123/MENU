import os
import glob

# --- CONFIGURATION ---
DATASET_FOLDER = "new_dataset_realistic_new"  # Change this to your dataset folder
IMAGES_FOLDER = os.path.join(DATASET_FOLDER, "images")
LABELS_FOLDER = os.path.join(DATASET_FOLDER, "labels")

def rename_files_with_suffix(folder_path, extensions, suffix="_realistic"):
    """
    Rename all files in folder by adding suffix before extension
    
    Example: file.jpg -> file_realistic.jpg
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return
    
    renamed_count = 0
    
    for ext in extensions:
        files = glob.glob(os.path.join(folder_path, f"*{ext}"))
        
        for file_path in files:
            # Get directory, filename, and extension
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name, extension = os.path.splitext(filename)
            
            # Skip if already has the suffix
            if name.endswith(suffix):
                print(f"SKIP (already renamed): {filename}")
                continue
            
            # Create new filename
            new_name = f"{name}{suffix}{extension}"
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            try:
                os.rename(file_path, new_path)
                print(f"✓ Renamed: {filename} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ Error renaming {filename}: {e}")
    
    return renamed_count

# --- MAIN EXECUTION ---
print("=" * 60)
print("RENAMING FILES - Adding '_realistic' suffix")
print("=" * 60)

# Rename images
print("\n[ IMAGES ]")
image_count = rename_files_with_suffix(IMAGES_FOLDER, ['.jpg', '.png', '.jpeg'])

# Rename labels
print("\n[ LABELS ]")
label_count = rename_files_with_suffix(LABELS_FOLDER, ['.txt'])

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total images renamed: {image_count}")
print(f"Total labels renamed: {label_count}")
print(f"\nDataset location: {DATASET_FOLDER}")
print("=" * 60)