import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class OrderDetector:
    """
    Detect customer orders using YOLO model and menu mapping JSON
    """
    
    def __init__(self, model_path, mapping_path):
        self.model = YOLO(model_path)
        
        # Load menu mapping
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.template_size = {
            'width': mapping_data['image_info']['width'],
            'height': mapping_data['image_info']['height']
        }
        self.menu_mapping = mapping_data['menu_mapping']
        
        print(f"📋 Loaded mapping with {len(self.menu_mapping)} items")
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two boxes
        Boxes in YOLO format: [center_x, center_y, width, height]
        """
        # Convert to [x1, y1, x2, y2]
        def to_corners(box):
            cx, cy, w, h = box
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return [x1, y1, x2, y2]
        
        b1 = to_corners(box1)
        b2 = to_corners(box2)
        
        # Calculate intersection
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def extract_quantity(self, class_name):
        """Extract quantity from mark type"""
        if 'number_' in class_name:
            return int(class_name.split('_')[-1])
        return 1
    
    def resize_image(self, image_path):
        """Resize customer image to match template dimensions"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to template size for accurate coordinate matching
        resized = cv2.resize(img, 
                            (self.template_size['width'], 
                             self.template_size['height']))
        return resized
    
    def detect_marks(self, image, conf=0.6):
        """Run YOLO detection on image"""
        results = self.model(image, imgsz=1280, conf=conf)
        
        detections = []
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    # Get normalized coordinates
                    x, y, w, h = box.xywhn[0].tolist()
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'class': class_name,
                        'bbox': [x, y, w, h],  # YOLO format
                        'confidence': confidence
                    })
        
        return detections
    
    def match_with_mapping(self, detections, iou_threshold=0.3):
        """Match detections with menu mapping using IOU"""
        order = {}
        unmatched_detections = []
        
        for detection in detections:
            det_box = detection['bbox']
            mark_class = detection['class']
            quantity = self.extract_quantity(mark_class)
            confidence = detection['confidence']
            
            # Find best matching checkbox
            best_match = None
            best_iou = iou_threshold
            
            for item_name, item_data in self.menu_mapping.items():
                for checkbox in item_data['checkboxes']:
                    cb_box = checkbox['bbox']
                    iou = self.calculate_iou(det_box, cb_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = {
                            'item': item_name,
                            'option': checkbox['option'],
                            'quantity': quantity,
                            'confidence': confidence,
                            'iou': iou
                        }
            
            if best_match:
                item = best_match['item']
                option = best_match['option']
                qty = best_match['quantity']
                
                if item not in order:
                    order[item] = {}
                
                if option not in order[item]:
                    order[item][option] = {
                        'quantity': 0,
                        'marks': []
                    }
                
                order[item][option]['quantity'] += qty
                order[item][option]['marks'].append({
                    'type': mark_class,
                    'confidence': confidence,
                    'iou': best_iou
                })
            else:
                unmatched_detections.append({
                    'mark': mark_class,
                    'confidence': confidence
                })
        
        return order, unmatched_detections
    
    def process_order(self, customer_image_path, save_visualization=True):
        """Complete order processing pipeline"""
        print("\n" + "="*70)
        print("🔍 ORDER DETECTION STARTED")
        print("="*70)
        
        # Step 1: Resize image to match template
        print(f"📐 Resizing image to match template ({self.template_size['width']}x{self.template_size['height']})...")
        resized_image = self.resize_image(customer_image_path)
        
        # Step 2: Detect marks with YOLO
        print("🤖 Running YOLO detection...")
        detections = self.detect_marks(resized_image)
        print(f"   Found {len(detections)} marks")
        
        # Step 3: Match with mapping
        print("🔗 Matching marks with menu mapping...")
        order, unmatched = self.match_with_mapping(detections)
        
        # Step 4: Display results
        self.display_results(order, unmatched, len(detections))
        
        # Step 5: Save visualization
        if save_visualization:
            self.save_visualization(resized_image, detections, order)
        
        return order
    
    def display_results(self, order, unmatched, total_detections):
        """Display order results"""
        print("\n" + "="*70)
        print("📊 ORDER SUMMARY")
        print("="*70)
        
        total_quantity = 0
        total_items = 0
        
        if order:
            for item_name, options in order.items():
                print(f"\n🍽️  {item_name.upper()}:")
                for option, data in options.items():
                    qty = data['quantity']
                    marks = data['marks']
                    total_quantity += qty
                    total_items += len(marks)
                    
                    print(f"   {option}: {qty}")
                    for mark in marks:
                        print(f"      └─ {mark['type']} (conf: {mark['confidence']:.2f}, iou: {mark['iou']:.2f})")
        else:
            print("No orders matched with menu mapping!")
        
        print("\n" + "-"*70)
        print(f"Total Items Detected: {total_detections}")
        print(f"Total Items Matched: {total_items}")
        print(f"Total Quantity Ordered: {total_quantity}")
        print(f"Unmatched Marks: {len(unmatched)}")
        
        if unmatched:
            print("\n⚠️  Unmatched Detections:")
            for u in unmatched:
                print(f"   - {u['mark']} (confidence: {u['confidence']:.2f})")
        
        print("="*70)
        
        # Return JSON format
        json_order = self.format_order_json(order)
        print("\n📄 JSON ORDER:")
        print(json.dumps(json_order, indent=2))
        
        return json_order
    
    def format_order_json(self, order):
        """Format order as clean JSON"""
        json_order = {}
        for item, options in order.items():
            json_order[item] = {}
            for option, data in options.items():
                json_order[item][option] = data['quantity']
        return json_order
    
    def save_visualization(self, image, detections, order):
        """Save visualization with bounding boxes"""
        output_dir = Path("runs/order_detection")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        vis_image = image.copy()
        
        # Draw all detections
        for det in detections:
            bbox = det['bbox']  # [cx, cy, w, h] normalized
            h, w = vis_image.shape[:2]
            
            x1 = int((bbox[0] - bbox[2]/2) * w)
            y1 = int((bbox[1] - bbox[3]/2) * h)
            x2 = int((bbox[0] + bbox[2]/2) * w)
            y2 = int((bbox[1] + bbox[3]/2) * h)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, det['class'], (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = output_dir / "order_visualization.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"\n💾 Visualization saved to: {output_path}")


def main():
    import sys
    
    # Configuration
    MODEL_PATH = "handwritten_best.pt"
    MAPPING_PATH = "menu_mapping_react.json"
    
    if len(sys.argv) > 1:
        customer_image = sys.argv[1]
    else:
        customer_image = "test_image_handwritten_dataset/images/test_img_sample_2.jpg"
    
    # Check files exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    if not Path(MAPPING_PATH).exists():
        print(f"❌ Mapping file not found: {MAPPING_PATH}")
        print("   Please create mapping first using: python create_menu_mapping.py")
        return
    
    # Run detection
    detector = OrderDetector(MODEL_PATH, MAPPING_PATH)
    order = detector.process_order(customer_image)
    
    # Save order to file
    with open("detected_order.json", 'w') as f:
        json.dump(order, f, indent=2)
    print(f"\n✅ Order saved to: detected_order.json")


if __name__ == "__main__":
    main()