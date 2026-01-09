from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the tiny YOLO model (fastest)
    model = YOLO('yolov8n.pt') 

    # 2. Train it on your data
    model.train(
        data='data.yaml',   # Config file from Step 2
        epochs=50,          # 50 rounds of learning
        imgsz=640,          # Resize images to 640x640 pixels
        batch=8,            # Process 8 images at once
        name='menu_model'   # Name of the output folder
    )