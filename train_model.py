import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e

# --- SIMPLE TRAIN SCRIPT (edit constants below) ---
WEIGHTS = "yolov8x.pt"    # default pretrained weights in this workspace
DATA_YAML = "data.yaml"
EPOCHS = 1
IMGSZ = 640
BATCH = 16
RUN_NAME = "menu_model"
SAVE_PATH = "best.pt"

def train(weights=WEIGHTS,
          data=DATA_YAML,
          epochs=EPOCHS,
          imgsz=IMGSZ,
          batch=BATCH,
          name=RUN_NAME,
          save_path=SAVE_PATH):
    model = YOLO(weights)
    print(f"Starting training: weights={weights}, data={data}, epochs={epochs}")
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, name=name)

    best_path = Path("runs") / "detect" / name / "weights" / "best.pt"
    if best_path.exists():
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_path, dest)
        print(f"Trained model copied to: {dest.resolve()}")
    else:
        print(f"Could not find {best_path}. Check runs/train/{name}")

if __name__ == "__main__":
    train()