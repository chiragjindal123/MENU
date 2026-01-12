# MENU — Synthetic Checkbox Marking & YOLO Training

Small toolkit to generate synthetic "handwritten" checkbox marks on menu images, prepare a YOLO dataset, train a YOLO model, and run inference.

Quick links
- Training script: [train_model.py](train_model.py) — contains the [`train`](train_model.py) function.
- Test/inference script: [test_model.py](test_model.py) — contains the [`test_image`](test_model.py) function.
- Data generation: [generate_data.py](generate_data.py)
- Dataset splitter: [split_dataset.py](split_dataset.py)
- YOLO config: [data.yaml](data.yaml)
- Sample datasets and outputs:
  - [dataset_combined_all_new/](dataset_combined_all_new/)
  - [yolo_dataset/](yolo_dataset/)
  - [test_img/](test_img/)
- Provided weights: [yolov8n.pt](yolov8n.pt), [yolov8x.pt](yolov8x.pt), [best.pt](best.pt), [last.pt](last.pt)

Requirements
- Python 3.8+
- Install: pip install ultralytics opencv-python numpy

Usage

1. Generate synthetic marked images (edit constants inside the script):
    python generate_data.py

2. Split into YOLO train/val (edit constants inside the script):
    python split_dataset.py

3. Train the model (edit constants in [train_model.py](train_model.py)):
    python train_model.py

The training entrypoint is [`train`](train_model.py). Trained best weights are copied to the path configured in the script.

4. Run inference (edit constants in [test_model.py](test_model.py)):
    python test_model.py

The inference entrypoint is [`test_image`](test_model.py). Output images are saved to the `runs/infer` folder by default.

Notes
- Labels use YOLO format: class cx cy w h (normalized).
- Edit the constants at the top of each script for simple configuration (no argparse).
- If ultralytics is missing the scripts will exit with an installation hint.
