
# from pathlib import Path

# try:
#     from ultralytics import YOLO
# except Exception as e:
#     raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e

# # INFER_WEIGHTS = "best_float32.tflite"          
# INFER_WEIGHTS = "mixed_menu_model_best.pt"          
# # TEST_IMAGE = "C:\\Users\\wmlab\\Desktop\\MENU\\test_img\\images\\fixed_range_sample_2.jpg"
# TEST_IMAGE = "test_mixed_sign_dataset/images/test_img_sample_9.jpg"
# OUTDIR = "runs/infer"
# IMGSZ = 1280
# CONF = 0.5

# def test_image(image_path=TEST_IMAGE,
#                weights=INFER_WEIGHTS,
#                outdir=OUTDIR,
#                imgsz=IMGSZ,
#                conf=CONF):
#     model = YOLO(weights)
#     results = model(image_path, imgsz=imgsz, conf=conf)
#     out_dir = Path(outdir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for i, r in enumerate(results):
#         plotted = r.plot(font_size=5, line_width=4)
#         img_out = out_dir / f"inference_{i}.jpg"

#         try:
#             import cv2
#             cv2.imwrite(str(img_out), plotted)
#         except Exception:
#             from PIL import Image
#             Image.fromarray(plotted[:, :, ::-1]).save(img_out)
#         print(f"Saved: {img_out}")
#         if hasattr(r, "boxes") and len(r.boxes) > 0:
#             for box in r.boxes:
#                 x, y, w, h = box.xywh[0].tolist()
#                 print(f"box: x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")

# if __name__ == "__main__":
#     test_image()




from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e

INFER_WEIGHTS = "multiclass_new.pt"          
TEST_IMAGE = "test_mixed_sign_dataset/images/test_img_sample_7.jpg"
OUTDIR = "runs/infer"
IMGSZ = 1280
CONF = 0.6

def extract_quantity(class_name):
    """Extract order quantity from detected mark"""
    if 'number_' in class_name:
        # Extract number from class name (e.g., 'marked_number_5' -> 5)
        return int(class_name.split('_')[-1])
    else:
        # Non-number marks count as quantity 1
        return 1

def test_image(image_path=TEST_IMAGE,
               weights=INFER_WEIGHTS,
               outdir=OUTDIR,
               imgsz=IMGSZ,
               conf=CONF):
    model = YOLO(weights)
    results = model(image_path, imgsz=imgsz, conf=conf)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_quantity = 0
    order_details = {}

    for i, r in enumerate(results):
        plotted = r.plot(font_size=5, line_width=4)
        img_out = out_dir / f"inference_{i}.jpg"

        try:
            import cv2
            cv2.imwrite(str(img_out), plotted)
        except Exception:
            from PIL import Image
            Image.fromarray(plotted[:, :, ::-1]).save(img_out)
        
        print(f"Saved: {img_out}")
        
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            print(f"\n📋 Detected {len(r.boxes)} marked items:")
            print("-" * 60)
            
            for idx, box in enumerate(r.boxes):
                x, y, w, h = box.xywh[0].tolist()
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])
                quantity = extract_quantity(class_name)
                
                print(f"Item {idx+1}:")
                print(f"  Mark Type: {class_name}")
                print(f"  Quantity: {quantity}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Position: x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")
                print()
                
                total_quantity += quantity
                order_details[class_name] = order_details.get(class_name, 0) + quantity
                
            print("-" * 60)
            print(f"📊 ORDER SUMMARY:")
            print(f"Total Items Marked: {len(r.boxes)}")
            print(f"Total Quantity: {total_quantity}")
            print("\nBreakdown by Mark Type:")
            for mark_type, qty in sorted(order_details.items()):
                count = sum(1 for box in r.boxes if r.names[int(box.cls[0])] == mark_type)
                if 'number_' in mark_type:
                    # Extract the number value from the mark type
                    number_value = int(mark_type.split('_')[-1])
                    print(f"  {mark_type}: {count} × {number_value} = {qty}")
                else:
                    # For non-number marks (circle, tick, x), each counts as 1
                    print(f"  {mark_type}: {count} × 1 = {qty}")


if __name__ == "__main__":
    test_image()