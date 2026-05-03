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
        self.match_near_px = 18      # consider marks near checkbox as valid
        self.model_imgsz = 1280       # more stable for your current model
        
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
    
    def detect_marks(self, image, conf=0.40, imgsz=None):
        """Run YOLO detection on image."""
        if imgsz is None:
            imgsz = self.model_imgsz
        results = self.model(image, imgsz=imgsz, conf=conf, verbose=False)

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
    
    def _expand_box(self, box, pad_x, pad_y):
        """Expand YOLO box by normalized padding."""
        cx, cy, w, h = box
        return [cx, cy, min(1.0, w + 2 * pad_x), min(1.0, h + 2 * pad_y)]

    def _point_in_box(self, point, box):
        """Check if normalized point is inside YOLO box."""
        px, py = point
        cx, cy, w, h = box
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return x1 <= px <= x2 and y1 <= py <= y2

    def _center_distance_ratio(self, point, box):
        """
        Distance from point to checkbox center, normalized by checkbox diagonal.
        <= 1 means within ~one checkbox diagonal.
        """
        px, py = point
        cx, cy, w, h = box
        diag = max(np.sqrt(w * w + h * h), 1e-6)
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        return dist / diag    

    def match_with_mapping(self, detections, iou_threshold=0.05):
        """Match detections with checkbox mapping using overlap OR touch/near logic."""
        order = {}
        unmatched_detections = []

        pad_x = self.match_near_px / self.template_size['width']
        pad_y = self.match_near_px / self.template_size['height']

        for detection in detections:
            det_box = detection['bbox']
            det_center = (det_box[0], det_box[1])
            mark_class = detection['class']
            quantity = self.extract_quantity(mark_class)
            confidence = detection['confidence']

            best_match = None
            best_score = -1.0

            for item_name, item_data in self.menu_mapping.items():
                for checkbox in item_data['checkboxes']:
                    cb_box = checkbox['bbox']

                    iou = self.calculate_iou(det_box, cb_box)
                    expanded_cb = self._expand_box(cb_box, pad_x, pad_y)
                    iou_near = self.calculate_iou(det_box, expanded_cb)
                    center_in_near = self._point_in_box(det_center, expanded_cb)
                    dist_ratio = self._center_distance_ratio(det_center, cb_box)
                    near_by_distance = dist_ratio <= 1.05

                    # score prefers true overlap, but also accepts touch/near
                    score = max(iou, iou_near * 0.9)
                    if center_in_near:
                        score = max(score, 0.12)
                    if near_by_distance:
                        score += max(0.0, (1.05 - dist_ratio)) * 0.05

                    is_valid = (iou >= iou_threshold) or center_in_near or near_by_distance
                    if is_valid and score > best_score:
                        best_score = score
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
                    order[item][option] = {'quantity': 0, 'marks': []}

                order[item][option]['quantity'] += qty
                order[item][option]['marks'].append({
                    'type': mark_class,
                    'confidence': confidence,
                    'iou': best_match['iou']
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