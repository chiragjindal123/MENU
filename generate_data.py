# import cv2
# import numpy as np
# import random
# import os

# # --- CONFIGURATION ---
# INPUT_IMAGE = "C:\\Users\\wmlab\\Desktop\\MENU\\5d237ef91e16d18db319c0a2a7049186.jpg"   # Change to your image name
# INPUT_LABELS = "C:\\Users\\wmlab\\Desktop\\MENU\\5d237ef91e16d18db319c0a2a7049186.txt"  # Change to your txt name
# OUTPUT_DIR = "dataset_train"
# NUM_VARIATIONS = 10              # How many fake images to create

# # Ensure output directories exist
# os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
# os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# def read_yolo_labels(txt_path, img_width, img_height):
#     """Reads YOLO txt and converts normalized coordinates to pixel box [x, y, w, h]"""
#     boxes = []
#     with open(txt_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = line.strip().split()
#             # YOLO format: class_id center_x center_y width height
#             # We ignore class_id for now, assuming everything is a checkbox
#             cx, cy, w, h = map(float, parts[1:])
            
#             # Convert back to pixels for drawing
#             pixel_w = int(w * img_width)
#             pixel_h = int(h * img_height)
#             pixel_x = int((cx * img_width) - (pixel_w / 2))
#             pixel_y = int((cy * img_height) - (pixel_h / 2))
            
#             boxes.append([pixel_x, pixel_y, pixel_w, pixel_h])
#     return boxes

# def draw_handwritten_mark(img, box):
#     x, y, w, h = box
    
#     # Random Pen Color (Blue, Black, Red)
#     colors = [(255, 0, 0), (20, 20, 20), (0, 0, 200)] 
#     color = random.choice(colors)
#     thickness = random.randint(2, 3)
    
#     # Jitter (Human imperfection - rarely mark exactly in center)
#     jitter_x = random.randint(-3, 3)
#     jitter_y = random.randint(-3, 3)

#     mark_type = random.choice(['circle', 'check', 'scribble'])

#     if mark_type == 'circle':
#         center_x = x + w // 2 + jitter_x
#         center_y = y + h // 2 + jitter_y
#         axes = (w // 2 + random.randint(-2, 4), h // 2 + random.randint(-2, 4))
#         angle = random.randint(0, 360)
#         cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, color, thickness)

#     elif mark_type == 'check':
#         # Draw a tick/check mark
#         pt1 = (x + random.randint(0, 5), y + h//2)
#         pt2 = (x + w//2, y + h - random.randint(0, 5))
#         pt3 = (x + w + random.randint(0,5), y + random.randint(0, 10))
#         pts = np.array([pt1, pt2, pt3], np.int32)
#         cv2.polylines(img, [pts], False, color, thickness)
        
#     elif mark_type == 'scribble':
#         # Just a messy line
#         pt1 = (x + random.randint(0, 5), y + h)
#         pt2 = (x + w - random.randint(0, 5), y)
#         cv2.line(img, pt1, pt2, color, thickness)

#     return img

# # --- MAIN PROCESS ---
# img = cv2.imread(INPUT_IMAGE)
# h_img, w_img, _ = img.shape

# # 1. Get all checkbox locations
# all_boxes = read_yolo_labels(INPUT_LABELS, w_img, h_img)

# print(f"Found {len(all_boxes)} checkboxes in the original file.")

# for i in range(NUM_VARIATIONS):
#     img_copy = img.copy()
#     new_labels = []
    
#     # --- LOGIC: SOME LESS, SOME MORE ---
#     # We choose a random "density" for this specific image.
#     # 0.1 means only 10% of items ordered (Small order)
#     # 0.8 means 80% of items ordered (Huge party order)
#     density = random.uniform(0.05, 0.5) 
    
#     for box in all_boxes:
#         # Roll the dice: Do we mark this box?
#         if random.random() < density:
#             # 1. DRAW the mark
#             draw_handwritten_mark(img_copy, box)
            
#             # 2. SAVE the label
#             # We must convert pixels back to YOLO normalized format
#             pixel_x, pixel_y, pixel_w, pixel_h = box
            
#             # Recalculate normalized center for YOLO
#             norm_cx = (pixel_x + pixel_w / 2) / w_img
#             norm_cy = (pixel_y + pixel_h / 2) / h_img
#             norm_w = pixel_w / w_img
#             norm_h = pixel_h / h_img
            
#             # Class ID 0 = "Marked Box"
#             new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

#     # Save the new Image
#     filename = f"train_sample_{i}"
#     cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", img_copy)
    
#     # Save the new Label txt
#     with open(f"{OUTPUT_DIR}/labels/{filename}.txt", 'w') as f:
#         f.write('\n'.join(new_labels))

#     print(f"Generated Image {i+1}: Density {int(density*100)}% ({len(new_labels)} marks)")

# print("Done! Check the 'dataset_train' folder.")














import cv2
import numpy as np
import random
import os

# --- CONFIGURATION ---
# Change these names to match your files!
INPUT_IMAGE = "C:\\Users\\wmlab\\Desktop\\MENU\\5d237ef91e16d18db319c0a2a7049186.jpg"   # Change to your image name
INPUT_LABELS = "C:\\Users\\wmlab\\Desktop\\MENU\\5d237ef91e16d18db319c0a2a7049186.txt"  # Change to your txt name
OUTPUT_DIR = "dataset_red_pen"   # New folder name
NUM_VARIATIONS = 10              # Create 10 variations of this menu

# Define Fixed Red Color (OpenCV uses BGR, so this is 0 Blue, 0 Green, 255 Red)
PEN_COLOR = (0, 0, 255)

# Ensure output directories exist
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

def read_yolo_labels(txt_path, img_width, img_height):
    """Reads YOLO txt and converts normalized coordinates to pixel box [x, y, w, h]"""
    boxes = []
    if not os.path.exists(txt_path):
        print(f"Error: Label file not found at {txt_path}")
        return []
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            # YOLO format: class_id center_x center_y width height
            cx, cy, w, h = map(float, parts[1:])
            
            # Convert back to pixels for drawing
            pixel_w = int(w * img_width)
            pixel_h = int(h * img_height)
            pixel_x = int((cx * img_width) - (pixel_w / 2))
            pixel_y = int((cy * img_height) - (pixel_h / 2))
            
            boxes.append([pixel_x, pixel_y, pixel_w, pixel_h])
    return boxes

def draw_handwritten_mark(img, box):
    x, y, w, h = box
    
    # Thickness for tick mark
    line_thickness = random.randint(2, 3)
    
    # Jitter (Human imperfection - rarely mark exactly in center)
    jitter_x = random.randint(-2, 2)
    jitter_y = random.randint(-2, 2)
    center_x = x + w // 2 + jitter_x
    center_y = y + h // 2 + jitter_y

    # --- REQUIREMENT UPDATE ---
    # Randomly choose only between tick (check) or filled_circle
    mark_type = random.choice(['filled_circle', 'check'])

    if mark_type == 'filled_circle':
        # Draw a slightly chaotic filled circle
        # Axes slightly smaller than box to look like ink inside
        axis_w = (w // 2) - random.randint(2, 5)
        axis_h = (h // 2) - random.randint(2, 5)
        
        if axis_w > 0 and axis_h > 0:
            # thickness=-1 means FILLED in OpenCV
            cv2.ellipse(img, (center_x, center_y), (axis_w, axis_h), 
                        0, 0, 360, PEN_COLOR, thickness=-1)

    elif mark_type == 'check':
        # Draw a red tick/check mark
        # Points are random but inside the box
        pt1 = (x + random.randint(0, w//4), y + h//2)
        pt2 = (x + w//2, y + h - random.randint(0, 4))
        pt3 = (x + w + random.randint(0,3), y + random.randint(0, h//3))
        pts = np.array([pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, PEN_COLOR, line_thickness)

    return img

# --- MAIN PROCESS ---
# Load image and check if it worked
img = cv2.imread(INPUT_IMAGE)
if img is None:
    print(f"Error: Could not load image {INPUT_IMAGE}")
    exit()

h_img, w_img, _ = img.shape

# 1. Get all checkbox locations
all_boxes = read_yolo_labels(INPUT_LABELS, w_img, h_img)
if not all_boxes:
    print("Error: No boxes found in label file. Dataset generation stopped.")
    exit()

print(f"Found {len(all_boxes)} checkboxes. Generating {NUM_VARIATIONS} images...")

for i in range(NUM_VARIATIONS):
    img_copy = img.copy()
    new_labels = []
    
    # Choose random order density (some less, some more)
    density = random.uniform(0.05, 0.6) # Up to 60% items marked
    
    for box in all_boxes:
        if random.random() < density:
            # 1. DRAW the red tick or filled circle
            draw_handwritten_mark(img_copy, box)
            
            # 2. SAVE the label (normalized back to YOLO format)
            pixel_x, pixel_y, pixel_w, pixel_h = box
            norm_cx = (pixel_x + pixel_w / 2) / w_img
            norm_cy = (pixel_y + pixel_h / 2) / h_img
            norm_w = pixel_w / w_img
            norm_h = pixel_h / h_img
            
            # Class ID 0 = "Marked Order"
            new_labels.append(f"0 {norm_cx} {norm_cy} {norm_w} {norm_h}")

    # Save new Image and Label txt
    filename = f"order_sample_{i}"
    cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", img_copy)
    with open(f"{OUTPUT_DIR}/labels/{filename}.txt", 'w') as f:
        f.write('\n'.join(new_labels))

    print(f" -> Created {filename} ({len(new_labels)} red orders)")

print(f"Done! Check the '{OUTPUT_DIR}' folder.")