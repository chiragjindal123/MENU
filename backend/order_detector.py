import cv2
import numpy as np
from ultralytics import YOLO

class OrderDetector:
    """
    Detect customer orders using YOLO model and menu mapping JSON
    Modified to work as API service
    """
    
    def __init__(self, model_path, mapping_data):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
            mapping_data: Menu mapping dictionary (not file path)
        """
        self.model = YOLO(model_path)
        
        # Extract mapping info
        self.template_size = {
            'width': mapping_data['image_info']['width'],
            'height': mapping_data['image_info']['height']
        }
        self.menu_mapping = mapping_data['menu_mapping']
        
        print(f"📋 Loaded mapping with {len(self.menu_mapping)} items")
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes (YOLO format)"""
        def to_corners(box):
            cx, cy, w, h = box
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return [x1, y1, x2, y2]
        
        b1 = to_corners(box1)
        b2 = to_corners(box2)
        
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def extract_quantity(self, class_name):
        """Extract quantity from mark type"""
        if 'number_' in class_name:
            try:
                return int(class_name.split('_')[-1])
            except:
                return 1
        return 1
    
    def resize_image(self, image):
        """Resize customer image to match template dimensions"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
            
        if img is None:
            raise ValueError("Could not load image")
        
        resized = cv2.resize(img, 
                            (self.template_size['width'], 
                             self.template_size['height']))
        return resized
    
    def detect_marks(self, image, conf=0.55):
        """Run YOLO detection on image"""
        results = self.model(image, imgsz=1280, conf=conf, verbose=False)
        
        detections = []
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box in r.boxes:
                    x, y, w, h = box.xywhn[0].tolist()
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'class': class_name,
                        'bbox': [x, y, w, h],
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
    
    def process_order(self, customer_image):
        """
        Complete order processing pipeline
        
        Args:
            customer_image: OpenCV image (numpy array) or image path
        
        Returns:
            tuple: (order_dict, unmatched_list)
        """
        # Resize image
        resized_image = self.resize_image(customer_image)
        
        # Detect marks
        detections = self.detect_marks(resized_image)
        
        # Match with mapping
        order, unmatched = self.match_with_mapping(detections)
        
        return order, unmatched