import cv2
import json
import numpy as np
from pathlib import Path

class MenuMappingTool:
    """
    Interactive tool to create JSON mapping for menu checkboxes
    Improved workflow:
        1. Load blank menu image
        2. Draw bounding box around ONE checkbox
        3. Enter item name and option (e.g., 'apple', 'S')
        4. Repeat for all checkboxes
        5. Save JSON mapping
    """
    
    def __init__(self, image_path, max_display_width=1200, max_display_height=800):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.img_height, self.img_width = self.original_image.shape[:2]
        
        # Calculate display scale to fit screen
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        self.scale = self.calculate_display_scale()
        
        # Create display image (scaled for viewing)
        self.display_width = int(self.img_width * self.scale)
        self.display_height = int(self.img_height * self.scale)
        self.display_image = cv2.resize(self.original_image, 
                                       (self.display_width, self.display_height))
        self.image = self.display_image.copy()
        
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.menu_mapping = {}
        self.temp_box = None  # Store box before labeling
        self.waiting_for_input = False  # Flag to prevent keyboard interference
        
        self.window_name = "Menu Mapping Tool"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"📐 Original image size: {self.img_width}x{self.img_height}")
        print(f"📐 Display size: {self.display_width}x{self.display_height}")
        print(f"📐 Scale factor: {self.scale:.3f}")
    
    def calculate_display_scale(self):
        """Calculate scale factor to fit image on screen"""
        width_scale = self.max_display_width / self.img_width
        height_scale = self.max_display_height / self.img_height
        return min(width_scale, height_scale, 1.0)
        
    def display_to_original_coords(self, x, y):
        """Convert display coordinates to original image coordinates"""
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)
        return orig_x, orig_y
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if not self.drawing:
                return  # Ignore if not drawing (just a click without drag)
            
            self.drawing = False
            end_point = (x, y)
            
            # Calculate box size to validate it's not just a click
            width = abs(end_point[0] - self.start_point[0])
            height = abs(end_point[1] - self.start_point[1])
            
            # Ignore tiny boxes (accidental clicks)
            if width < 10 or height < 10:
                print("⚠️  Box too small. Draw a larger bounding box.")
                self.image = self.image.copy()  # Refresh to remove temporary box
                cv2.imshow(self.window_name, self.image)
                return
            
            # Convert display coordinates to original image coordinates
            x1_orig, y1_orig = self.display_to_original_coords(
                min(self.start_point[0], end_point[0]),
                min(self.start_point[1], end_point[1])
            )
            x2_orig, y2_orig = self.display_to_original_coords(
                max(self.start_point[0], end_point[0]),
                max(self.start_point[1], end_point[1])
            )
            
            self.temp_box = {
                'x1': x1_orig,
                'y1': y1_orig,
                'x2': x2_orig,
                'y2': y2_orig,
                'display': {
                    'x1': min(self.start_point[0], end_point[0]),
                    'y1': min(self.start_point[1], end_point[1]),
                    'x2': max(self.start_point[0], end_point[0]),
                    'y2': max(self.start_point[1], end_point[1])
                }
            }
            
            # Draw final box on display image (temporary)
            cv2.rectangle(self.image, 
                         self.start_point,
                         end_point,
                         (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)
            
            # Process the drawn box
            self.process_box()
            
    def pixel_to_yolo(self, box_pixel):
        """Convert pixel coordinates to YOLO normalized format"""
        x = (box_pixel['x1'] + box_pixel['x2']) / 2 / self.img_width
        y = (box_pixel['y1'] + box_pixel['y2']) / 2 / self.img_height
        w = (box_pixel['x2'] - box_pixel['x1']) / self.img_width
        h = (box_pixel['y2'] - box_pixel['y1']) / self.img_height
        return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
    
    def process_box(self):
        """Process the drawn box and get user input with retry capability"""
        if not self.temp_box:
            return
        
        self.waiting_for_input = True  # Disable keyboard shortcuts
        
        while True:  # Loop until valid input or cancel
            print("\n" + "="*60)
            print("📦 New Checkbox Detected")
            print("="*60)
            print(f"Box size: {self.temp_box['x2'] - self.temp_box['x1']}x{self.temp_box['y2'] - self.temp_box['y1']} pixels")
            print("\nType 'cancel' to discard this box and draw again")
            print("-"*60)
            
            # Get item name
            item_name = input("Enter item name (e.g., 'apple'): ").strip()
            
            if item_name.lower() == 'cancel':
                print("❌ Box cancelled. Draw a new one.")
                self.image = self.display_image.copy()
                self.redraw_all_boxes()
                cv2.imshow(self.window_name, self.image)
                self.temp_box = None
                self.waiting_for_input = False
                return
            
            if not item_name:
                print("⚠️  Item name cannot be empty. Try again.")
                continue
            
            # Get option name
            option = input("Enter option name (e.g., 'S', 'M', 'L' or leave empty): ").strip()
            
            if option.lower() == 'cancel':
                print("❌ Box cancelled. Draw a new one.")
                self.image = self.display_image.copy()
                self.redraw_all_boxes()
                cv2.imshow(self.window_name, self.image)
                self.temp_box = None
                self.waiting_for_input = False
                return
            
            if not option:
                option = "default"
            
            # Confirm
            print(f"\n✓ Item: '{item_name}', Option: '{option}'")
            confirm = input("Confirm? (y/n): ").strip().lower()
            
            if confirm == 'y' or confirm == 'yes' or confirm == '':
                # Add to mapping
                yolo_box = self.pixel_to_yolo(self.temp_box)
                
                if item_name not in self.menu_mapping:
                    self.menu_mapping[item_name] = {'checkboxes': []}
                
                self.menu_mapping[item_name]['checkboxes'].append({
                    'option': option,
                    'bbox': yolo_box
                })
                
                # Draw permanent box with label
                display_box = self.temp_box['display']
                cv2.rectangle(self.image,
                             (display_box['x1'], display_box['y1']),
                             (display_box['x2'], display_box['y2']),
                             (0, 200, 0), 2)
                
                # Add label text
                label = f"{item_name}:{option}"
                cv2.putText(self.image, label,
                           (display_box['x1'] + 5, display_box['y1'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                
                print(f"✅ Added '{item_name}:{option}'")
                print(f"   Total checkboxes mapped: {self.count_total_checkboxes()}")
                print("\n💡 TIP: Click on image window, then use keyboard shortcuts:")
                print("   's' = Save | 'u' = Undo | 'v' = View summary | 'q' = Quit")
                
                cv2.imshow(self.window_name, self.image)
                self.temp_box = None
                self.waiting_for_input = False
                return
            else:
                print("↩️  Let's try again...")
                # Loop continues for retry
    
    def count_total_checkboxes(self):
        """Count total number of checkboxes mapped"""
        total = 0
        for item_data in self.menu_mapping.values():
            total += len(item_data['checkboxes'])
        return total
    
    def redraw_all_boxes(self):
        """Redraw all existing mapped boxes"""
        for item_name, item_data in self.menu_mapping.items():
            for checkbox in item_data['checkboxes']:
                # Convert YOLO back to display coordinates
                yolo = checkbox['bbox']
                x_center = yolo[0] * self.img_width
                y_center = yolo[1] * self.img_height
                width = yolo[2] * self.img_width
                height = yolo[3] * self.img_height
                
                x1 = int((x_center - width/2) * self.scale)
                y1 = int((y_center - height/2) * self.scale)
                x2 = int((x_center + width/2) * self.scale)
                y2 = int((y_center + height/2) * self.scale)
                
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 200, 0), 2)
                
                label = f"{item_name}:{checkbox['option']}"
                cv2.putText(self.image, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    
    def undo_last(self):
        """Undo the last added checkbox"""
        if not self.menu_mapping:
            print("⚠️  Nothing to undo!")
            return
        
        # Find last item and remove last checkbox
        last_item = list(self.menu_mapping.keys())[-1]
        if self.menu_mapping[last_item]['checkboxes']:
            removed = self.menu_mapping[last_item]['checkboxes'].pop()
            print(f"↩️  Undone: {last_item}:{removed['option']}")
            
            # Remove item if no checkboxes left
            if not self.menu_mapping[last_item]['checkboxes']:
                del self.menu_mapping[last_item]
                print(f"   Removed item '{last_item}' (no checkboxes left)")
            
            # Redraw
            self.image = self.display_image.copy()
            self.redraw_all_boxes()
            cv2.imshow(self.window_name, self.image)
    
    def save_mapping(self, output_path="menu_mapping.json"):
        """Save the menu mapping to JSON file"""
        if not self.menu_mapping:
            print("⚠️  No mapping to save!")
            return False
        
        mapping_data = {
            'image_info': {
                'width': self.img_width,
                'height': self.img_height,
                'source': str(self.image_path)
            },
            'menu_mapping': self.menu_mapping
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print("\n" + "="*60)
        print(f"🎉 Menu mapping saved to: {output_path}")
        print(f"   Total items: {len(self.menu_mapping)}")
        print(f"   Total checkboxes: {self.count_total_checkboxes()}")
        print("="*60)
        
        return True
    
    def show_summary(self):
        """Show current mapping summary"""
        print("\n" + "="*60)
        print("📋 CURRENT MAPPING SUMMARY")
        print("="*60)
        
        if not self.menu_mapping:
            print("No checkboxes mapped yet.")
        else:
            for item_name, item_data in self.menu_mapping.items():
                options = [cb['option'] for cb in item_data['checkboxes']]
                print(f"  {item_name}: {', '.join(options)}")
        
        print(f"\nTotal items: {len(self.menu_mapping)}")
        print(f"Total checkboxes: {self.count_total_checkboxes()}")
        print("="*60)
    
    def run(self):
        """Run the interactive tool"""
        print("\n" + "="*60)
        print("🎨 MENU MAPPING TOOL")
        print("="*60)
        print("Instructions:")
        print("  1. DRAG (click and hold) a box around ONE checkbox")
        print("  2. Enter item name and option in terminal")
        print("  3. After adding checkbox, CLICK on image window")
        print("  4. Then use keyboard shortcuts")
        print("\n⌨️  Keyboard shortcuts (click image window first!):")
        print("  's' - Save mapping to JSON")
        print("  'u' - Undo last checkbox")
        print("  'v' - View current mapping summary")
        print("  'r' - Reset all (start over)")
        print("  'q' - Quit")
        print("="*60 + "\n")
        
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Disable keyboard shortcuts while waiting for terminal input
            if self.waiting_for_input:
                continue
            
            if key == ord('q'):
                # Quit
                if self.menu_mapping:
                    print("\n💾 Save before quitting? (y/n): ", end='', flush=True)
                    self.waiting_for_input = True
                    save_prompt = input().strip().lower()
                    self.waiting_for_input = False
                    if save_prompt == 'y' or save_prompt == 'yes':
                        self.save_mapping()
                break
                
            elif key == ord('s'):
                # Save mapping
                print("\n💾 Saving mapping...")
                self.save_mapping()
                
            elif key == ord('u'):
                # Undo last
                self.undo_last()
                
            elif key == ord('v'):
                # View summary
                self.show_summary()
                
            elif key == ord('r'):
                # Reset all
                print("\n⚠️  Reset all mapping? This cannot be undone! (yes/no): ", end='', flush=True)
                self.waiting_for_input = True
                confirm = input().strip().lower()
                self.waiting_for_input = False
                if confirm == 'yes':
                    self.menu_mapping = {}
                    self.image = self.display_image.copy()
                    cv2.imshow(self.window_name, self.image)
                    print("🔄 All mapping reset!")
                else:
                    print("❌ Reset cancelled")
        
        cv2.destroyAllWindows()


def main():
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: use first image from menu_data folder
        menu_folder = Path("test_images_make")
        image_files = list(menu_folder.glob("*.jpg")) + list(menu_folder.glob("*.png"))
        
        if not image_files:
            print("❌ No images found in test_images_make folder")
            print("Usage: python create_menu_mapping.py <path_to_blank_menu_image>")
            return
        
        image_path = image_files[0]
        print(f"Using image: {image_path}")
    
    try:
        tool = MenuMappingTool(str(image_path), 
                              max_display_width=1200,
                              max_display_height=800)
        tool.run()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()