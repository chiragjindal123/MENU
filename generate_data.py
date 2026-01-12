# ---------------------------single Image-----------------------------



# import cv2
# import numpy as np
# import random
# import os

# # --- CONFIGURATION ---
# INPUT_IMAGE = "C:\\Users\\wmlab\\Desktop\\MENU\\test_img.jpg"   # Change to your image name
# INPUT_LABELS = "C:\\Users\\wmlab\\Desktop\\MENU\\test_img.txt"  # Change to your txt name
# OUTPUT_DIR = "test_img" # New folder name
# NUM_VARIATIONS = 10              # How many images to generate

# # Red Color (Blue=0, Green=0, Red=255)
# PEN_COLOR = (0, 0, 255)

# # Ensure output directories exist
# os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
# os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# def read_yolo_labels(txt_path, img_width, img_height):
#     boxes = []
#     if not os.path.exists(txt_path):
#         print(f"Error: Label file not found at {txt_path}")
#         return []
        
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = line.strip().split()
#             if not parts: continue
#             cx, cy, w, h = map(float, parts[1:])
            
#             pixel_w = int(w * img_width)
#             pixel_h = int(h * img_height)
#             pixel_x = int((cx * img_width) - (pixel_w / 2))
#             pixel_y = int((cy * img_height) - (pixel_h / 2))
            
#             boxes.append([pixel_x, pixel_y, pixel_w, pixel_h])
#     return boxes

# def draw_handwritten_mark(img, box):
#     x, y, w, h = box
    
#     # 1. CENTER POINT
#     # Add small jitter so it looks human (not perfect robot center)
#     jitter = random.randint(-1, 1) 
#     center_x = x + w // 2 + jitter
#     center_y = y + h // 2 + jitter

#     # 2. SIZE (Little Large)
#     # We calculate the radius based on the box size.
#     radius_x = int( (w // 2) * random.uniform(0.85, 0.95) )
#     radius_y = int( (h // 2) * random.uniform(0.85, 0.95) )
    
#     # 3. DRAW FILLED ELLIPSE (DOT)
#     # thickness=-1 makes it a solid filled dot
#     cv2.ellipse(img, 
#                 (center_x, center_y), 
#                 (radius_x, radius_y), 
#                 0, 0, 360, 
#                 PEN_COLOR, 
#                 thickness=-1) 

#     return img

# # --- MAIN GENERATOR ---
# img = cv2.imread(INPUT_IMAGE)
# if img is None:
#     print(f"Error: Could not load {INPUT_IMAGE}")
#     exit()

# h_img, w_img, _ = img.shape
# all_boxes = read_yolo_labels(INPUT_LABELS, w_img, h_img)

# if not all_boxes:
#     print("Error: No boxes found. Check your .txt file.")
#     exit()

# print(f"Generating {NUM_VARIATIONS} images with LARGE RED DOTS...")

# # SET YOUR RANGE HERE
# MIN_MARKS = 3   # Minimum items to order
# MAX_MARKS = 8   # Maximum items to order

# for i in range(NUM_VARIATIONS):
#     img_copy = img.copy()
#     new_labels = []
    
#     count_this_image = random.randint(MIN_MARKS, MAX_MARKS)
    
#     # 2. Randomly pick that many boxes from the list
#     selected_boxes = random.sample(all_boxes, count_this_image)
    
#     # 3. Draw ONLY on the selected boxes
#     for box in selected_boxes:
#         # Draw the mark
#         draw_handwritten_mark(img_copy, box)
        
#         # Save the label
#         pixel_x, pixel_y, pixel_w, pixel_h = box
#         norm_cx = (pixel_x + pixel_w / 2) / w_img
#         norm_cy = (pixel_y + pixel_h / 2) / h_img
#         norm_w = pixel_w / w_img
#         norm_h = pixel_h / h_img
        
#         new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

#     # Save
#     filename = f"fixed_range_sample_{i}"
#     cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", img_copy)
#     with open(f"{OUTPUT_DIR}/labels/{filename}.txt", 'w') as f:
#         f.write('\n'.join(new_labels))

#     print(f" -> Created {filename} with exactly {count_this_image} marks.")

# print(f"Done! Check the '{OUTPUT_DIR}' folder.")




# ------------ Batch------------------------------




import cv2
import numpy as np
import random
import os
import glob

# --- CONFIGURATION ---
# 1. Point this to the folder containing ALL your clean images and .txt files
INPUT_FOLDER = r"C:\Users\wmlab\Desktop\MENU\menu_data" 

OUTPUT_DIR = "dataset_combined_all_new" 

# 3. How many marked versions to make PER MENU IMAGE
NUM_VARIATIONS_PER_MENU = 10 

# Red Color (Blue=0, Green=0, Red=255)
PEN_COLOR = (0, 0, 255)

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# --- HELPER FUNCTIONS ---
def read_yolo_labels(txt_path, img_width, img_height):
    boxes = []
    if not os.path.exists(txt_path):
        return []
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                cx, cy, w, h = map(float, parts[1:])
                pixel_w = int(w * img_width)
                pixel_h = int(h * img_height)
                pixel_x = int((cx * img_width) - (pixel_w / 2))
                pixel_y = int((cy * img_height) - (pixel_h / 2))
                boxes.append([pixel_x, pixel_y, pixel_w, pixel_h])
            except ValueError:
                continue
    return boxes

def draw_handwritten_mark(img, box):
    x, y, w, h = box
    jitter = random.randint(-1, 1) 
    center_x = x + w // 2 + jitter
    center_y = y + h // 2 + jitter

    radius_x = int( (w // 2) * random.uniform(0.85, 0.95) )
    radius_y = int( (h // 2) * random.uniform(0.85, 0.95) )
    
    cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, PEN_COLOR, thickness=-1) 
    return img

# --- MAIN GENERATOR LOOP ---

# 1. Find all JPG/PNG images in the folder
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

print(f"Found {len(image_files)} source images in {INPUT_FOLDER}...")

for img_path in image_files:
    # Get the filename without extension (e.g., "5d237ef...")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Construct the expected path for the corresponding .txt file
    txt_path = os.path.join(INPUT_FOLDER, base_name + ".txt")
    
    # Check if .txt file exists for this image
    if not os.path.exists(txt_path):
        print(f"SKIPPING: {base_name} (No matching .txt file found)")
        continue

    # Load the Image
    img = cv2.imread(img_path)
    if img is None:
        continue
    h_img, w_img, _ = img.shape

    # Load the Boxes
    all_boxes = read_yolo_labels(txt_path, w_img, h_img)
    if not all_boxes:
        print(f"SKIPPING: {base_name} (Label file empty)")
        continue

    print(f"Processing: {base_name} ({len(all_boxes)} checkboxes found)")

    # Generate Variations for THIS specific menu
    MIN_MARKS = 3   
    MAX_MARKS = 8   

    for i in range(NUM_VARIATIONS_PER_MENU):
        img_copy = img.copy()
        new_labels = []
        
        # Decide how many marks (ensure we don't try to mark more boxes than exist)
        current_max = min(MAX_MARKS, len(all_boxes))
        if current_max < MIN_MARKS:
            count_this_image = len(all_boxes)
        else:
            count_this_image = random.randint(MIN_MARKS, current_max)
        
        selected_boxes = random.sample(all_boxes, count_this_image)
        
        for box in selected_boxes:
            draw_handwritten_mark(img_copy, box)
            
            pixel_x, pixel_y, pixel_w, pixel_h = box
            norm_cx = (pixel_x + pixel_w / 2) / w_img
            norm_cy = (pixel_y + pixel_h / 2) / h_img
            norm_w = pixel_w / w_img
            norm_h = pixel_h / h_img
            
            new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

        # Save with a UNIQUE filename (BaseName + Number)
        # Example: 5d237ef_sample_0.jpg
        save_name = f"{base_name}_sample_{i}"
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{save_name}.jpg", img_copy)
        with open(f"{OUTPUT_DIR}/labels/{save_name}.txt", 'w') as f:
            f.write('\n'.join(new_labels))

print("-" * 30)
print(f"Done! All images saved to: {OUTPUT_DIR}")