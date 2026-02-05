
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e

# INFER_WEIGHTS = "best_float32.tflite"          
INFER_WEIGHTS = "mixed_menu_model_best.pt"          
# TEST_IMAGE = "C:\\Users\\wmlab\\Desktop\\MENU\\test_img\\images\\fixed_range_sample_2.jpg"
TEST_IMAGE = "test_mixed_sign_dataset/images/test_img_sample_7.jpg"
OUTDIR = "runs/infer"
IMGSZ = 1280
CONF = 0.5

def test_image(image_path=TEST_IMAGE,
               weights=INFER_WEIGHTS,
               outdir=OUTDIR,
               imgsz=IMGSZ,
               conf=CONF):
    model = YOLO(weights)
    results = model(image_path, imgsz=imgsz, conf=conf)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            for box in r.boxes:
                x, y, w, h = box.xywh[0].tolist()
                print(f"box: x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")

if __name__ == "__main__":
    test_image()