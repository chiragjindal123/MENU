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




# import cv2
# import numpy as np
# import random
# import os
# import glob

# # --- CONFIGURATION ---
# # 1. Point this to the folder containing ALL your clean images and .txt files
# INPUT_FOLDER = r"C:\Users\wmlab\Desktop\MENU\menu_data" 

# OUTPUT_DIR = "mixed_sign_dataset" 

# # 3. How many marked versions to make PER MENU IMAGE
# NUM_VARIATIONS_PER_MENU = 10 

# # Red Color (Blue=0, Green=0, Red=255)
# PEN_COLOR = (0, 0, 255)

# os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
# os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# # --- HELPER FUNCTIONS ---
# def read_yolo_labels(txt_path, img_width, img_height):
#     boxes = []
#     if not os.path.exists(txt_path):
#         return []
        
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = line.strip().split()
#             if not parts: continue
#             try:
#                 cx, cy, w, h = map(float, parts[1:])
#                 pixel_w = int(w * img_width)
#                 pixel_h = int(h * img_height)
#                 pixel_x = int((cx * img_width) - (pixel_w / 2))
#                 pixel_y = int((cy * img_height) - (pixel_h / 2))
#                 boxes.append([pixel_x, pixel_y, pixel_w, pixel_h])
#             except ValueError:
#                 continue
#     return boxes

# def draw_handwritten_mark(img, box):
#     x, y, w, h = box
#     jitter = random.randint(-1, 1) 
#     center_x = x + w // 2 + jitter
#     center_y = y + h // 2 + jitter

#     radius_x = int( (w // 2) * random.uniform(0.85, 0.95) )
#     radius_y = int( (h // 2) * random.uniform(0.85, 0.95) )
    
#     cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, PEN_COLOR, thickness=-1) 
#     return img

# # --- MAIN GENERATOR LOOP ---

# # 1. Find all JPG/PNG images in the folder
# image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
#               glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

# print(f"Found {len(image_files)} source images in {INPUT_FOLDER}...")

# for img_path in image_files:
#     # Get the filename without extension (e.g., "5d237ef...")
#     base_name = os.path.splitext(os.path.basename(img_path))[0]
    
#     # Construct the expected path for the corresponding .txt file
#     txt_path = os.path.join(INPUT_FOLDER, base_name + ".txt")
    
#     # Check if .txt file exists for this image
#     if not os.path.exists(txt_path):
#         print(f"SKIPPING: {base_name} (No matching .txt file found)")
#         continue

#     # Load the Image
#     img = cv2.imread(img_path)
#     if img is None:
#         continue
#     h_img, w_img, _ = img.shape

#     # Load the Boxes
#     all_boxes = read_yolo_labels(txt_path, w_img, h_img)
#     if not all_boxes:
#         print(f"SKIPPING: {base_name} (Label file empty)")
#         continue

#     print(f"Processing: {base_name} ({len(all_boxes)} checkboxes found)")

#     # Generate Variations for THIS specific menu
#     MIN_MARKS = 3   
#     MAX_MARKS = 8   

#     for i in range(NUM_VARIATIONS_PER_MENU):
#         img_copy = img.copy()
#         new_labels = []
        
#         # Decide how many marks (ensure we don't try to mark more boxes than exist)
#         current_max = min(MAX_MARKS, len(all_boxes))
#         if current_max < MIN_MARKS:
#             count_this_image = len(all_boxes)
#         else:
#             count_this_image = random.randint(MIN_MARKS, current_max)
        
#         selected_boxes = random.sample(all_boxes, count_this_image)
        
#         for box in selected_boxes:
#             draw_handwritten_mark(img_copy, box)
            
#             pixel_x, pixel_y, pixel_w, pixel_h = box
#             norm_cx = (pixel_x + pixel_w / 2) / w_img
#             norm_cy = (pixel_y + pixel_h / 2) / h_img
#             norm_w = pixel_w / w_img
#             norm_h = pixel_h / h_img
            
#             new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

#         # Save with a UNIQUE filename (BaseName + Number)
#         # Example: 5d237ef_sample_0.jpg
#         save_name = f"{base_name}_sample_{i}"
        
#         cv2.imwrite(f"{OUTPUT_DIR}/images/{save_name}.jpg", img_copy)
#         with open(f"{OUTPUT_DIR}/labels/{save_name}.txt", 'w') as f:
#             f.write('\n'.join(new_labels))

# print("-" * 30)
# print(f"Done! All images saved to: {OUTPUT_DIR}")


import cv2
import numpy as np
import random
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = "test_img_mixed"
OUTPUT_DIR = "test_mixed_sign_dataset" 
NUM_VARIATIONS_PER_MENU = 10 

# --- DEFINE COLORS ---
COLORS = {
    'red': (0, 0, 255),           # Bright red
    'light_red': (102, 102, 255),  # Light red (pink-ish)
    'blue': (255, 0, 0),           # Bright blue
    'light_blue': (255, 153, 51),  # Light blue
}

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

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

def draw_circle_mark(img, box, color):
    """Draw a filled circle (current style)"""
    x, y, w, h = box
    jitter = random.randint(-1, 1) 
    center_x = x + w // 2 + jitter
    center_y = y + h // 2 + jitter
    radius_x = int((w // 2) * random.uniform(0.85, 0.95))
    radius_y = int((h // 2) * random.uniform(0.85, 0.95))
    cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, color, thickness=-1)
    return img

def draw_tick_mark(img, box, color):
    """Draw a checkmark/tick ✓"""
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Create tick coordinates
    # Bottom point
    pt1 = (int(center_x - w*0.3), int(center_y + h*0.1))
    # Middle point (corner of tick)
    pt2 = (int(center_x - w*0.1), int(center_y + h*0.3))
    # Top point
    pt3 = (int(center_x + w*0.3), int(center_y - h*0.3))
    
    thickness = max(1, int(w * 0.15))
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt2, pt3, color, thickness)
    return img

def draw_x_mark(img, box, color):
    """Draw an X mark"""
    x, y, w, h = box
    thickness = max(1, int(w * 0.15))
    
    # Draw X (two diagonal lines)
    pt1 = (int(x + w*0.2), int(y + h*0.2))
    pt2 = (int(x + w*0.8), int(y + h*0.8))
    pt3 = (int(x + w*0.8), int(y + h*0.2))
    pt4 = (int(x + w*0.2), int(y + h*0.8))
    
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt3, pt4, color, thickness)
    return img

def draw_number(img, box, color, number):
    """Draw a number (1-9)"""
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate font scale based on box size
    font_scale = w / 30.0
    thickness = max(1, int(w * 0.1))
    
    # Get text size to center it
    text = str(number)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness)
    return img

def draw_hollow_circle(img, box, color):
    """Draw a hollow circle (outline only)"""
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2
    radius = int(min(w, h) * 0.4)
    thickness = max(1, int(w * 0.1))
    cv2.circle(img, (center_x, center_y), radius, color, thickness)
    return img

def apply_random_mark(img, box):
    """Apply a random mark type with random color"""
    # Choose random color
    color = random.choice(list(COLORS.values()))
    
    # Choose random mark type
    mark_types = [
        ('circle', 0.30),      # 30% filled circle
        ('tick', 0.25),        # 25% tick mark
        ('x', 0.15),           # 15% X mark
        ('number', 0.20),      # 20% numbers
        ('hollow', 0.10),      # 10% hollow circle
    ]
    
    # Weighted random selection
    mark_type = random.choices(
        [m[0] for m in mark_types],
        weights=[m[1] for m in mark_types]
    )[0]
    
    if mark_type == 'circle':
        draw_circle_mark(img, box, color)
    elif mark_type == 'tick':
        draw_tick_mark(img, box, color)
    elif mark_type == 'x':
        draw_x_mark(img, box, color)
    elif mark_type == 'number':
        number = random.randint(1, 9)
        draw_number(img, box, color, number)
    elif mark_type == 'hollow':
        draw_hollow_circle(img, box, color)
    
    return img

# --- MAIN GENERATOR LOOP ---
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

print(f"Found {len(image_files)} source images in {INPUT_FOLDER}...")
print("\nMark Types:")
print("  - Filled circles (30%)")
print("  - Tick marks ✓ (25%)")
print("  - X marks ✗ (15%)")
print("  - Numbers 1-9 (20%)")
print("  - Hollow circles ○ (10%)")
print("\nColors: Red, Light Red, Blue, Light Blue")
print("-" * 50)

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(INPUT_FOLDER, base_name + ".txt")
    
    if not os.path.exists(txt_path):
        print(f"SKIPPING: {base_name} (No matching .txt file found)")
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h_img, w_img, _ = img.shape

    all_boxes = read_yolo_labels(txt_path, w_img, h_img)
    if not all_boxes:
        print(f"SKIPPING: {base_name} (Label file empty)")
        continue

    print(f"Processing: {base_name} ({len(all_boxes)} checkboxes found)")

    MIN_MARKS = 8      
    MAX_MARKS = 15   

    for i in range(NUM_VARIATIONS_PER_MENU):
        img_copy = img.copy()
        new_labels = []
        
        current_max = min(MAX_MARKS, len(all_boxes))
        if current_max < MIN_MARKS:
            count_this_image = len(all_boxes)
        else:
            count_this_image = random.randint(MIN_MARKS, current_max)
        
        selected_boxes = random.sample(all_boxes, count_this_image)
        
        for box in selected_boxes:
            # Apply random mark with random color
            apply_random_mark(img_copy, box)
            
            pixel_x, pixel_y, pixel_w, pixel_h = box
            norm_cx = (pixel_x + pixel_w / 2) / w_img
            norm_cy = (pixel_y + pixel_h / 2) / h_img
            norm_w = pixel_w / w_img
            norm_h = pixel_h / h_img
            
            new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

        save_name = f"{base_name}_sample_{i}"
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{save_name}.jpg", img_copy)
        with open(f"{OUTPUT_DIR}/labels/{save_name}.txt", 'w') as f:
            f.write('\n'.join(new_labels))

print("-" * 50)
print(f"Done! All images saved to: {OUTPUT_DIR}")
print(f"Total variations created: {len(image_files) * NUM_VARIATIONS_PER_MENU}")