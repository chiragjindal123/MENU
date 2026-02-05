from pathlib import Path
import cv2
import time
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

INFER_WEIGHTS = resource_path("best.pt")

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics is required. Install with: pip install ultralytics") from e

# --- OPTIMIZED CONFIGURATION FOR SPEED ---
# INFER_WEIGHTS = "best.pt"  # Use .pt instead of TFLite for better performance
INFER_WEIGHTS = "mixed_menu_model_best.pt"  # Use .pt instead of TFLite for better performance
IMGSZ = 640  # REDUCED from 1280 - this is the KEY to speed!
CONF = 0.5
CAMERA_ID = 0
FPS_LIMIT = 30  # Increased target FPS

def try_open_camera(camera_id=0):
    """
    Try different camera backends for Windows compatibility
    """
    print("Attempting to open camera...")
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    for backend, name in backends:
        print(f"Trying {name} backend...")
        cap = cv2.VideoCapture(camera_id, backend)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Successfully opened camera with {name} backend!")
                return cap
            else:
                print(f"✗ {name} opened but cannot read frames")
                cap.release()
        else:
            print(f"✗ Failed to open with {name}")
    
    return None

def run_camera_detection(weights=INFER_WEIGHTS,
                         camera_id=CAMERA_ID,
                         imgsz=IMGSZ,
                         conf=CONF,
                         fps_limit=FPS_LIMIT):
    """
    Run OPTIMIZED real-time detection on camera feed
    """
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    print("Model loaded successfully!")
    
    cap = try_open_camera(camera_id)
    
    if cap is None:
        print("\n❌ ERROR: Could not open camera!")
        print("\nTroubleshooting steps:")
        print("1. Close all apps using camera (Zoom, Teams, Skype, Camera app)")
        print("2. Check Windows Settings > Privacy > Camera")
        print("3. Try different camera_id (0, 1, 2)")
        print("4. Restart your computer")
        return
    
    # OPTIMIZED camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from camera
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Model inference size: {imgsz}x{imgsz}")
    
    print("\n✓ Camera opened successfully!")
    print("Controls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'S' to save screenshot")
    print("  - Press '1-6' to change speed (1=fastest, 6=most accurate)")
    print("-" * 50)
    
    frame_count = 0
    screenshot_count = 0
    current_conf = conf
    current_imgsz = imgsz
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    # Skip frames for even faster processing
    skip_frames = 0  # Process every frame initially
    frame_skip_counter = 0
    
    Path("screenshots").mkdir(exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame_skip_counter += 1
            
            # Skip frames if needed for speed
            if frame_skip_counter % (skip_frames + 1) != 0:
                cv2.imshow('Menu Item Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Run detection on this frame
            results = model(frame, imgsz=current_imgsz, conf=current_conf, verbose=False)
            
            # Draw results
            if results and len(results) > 0:
                annotated_frame = results[0].plot(font_size=2, line_width=1)
                
                num_detections = 0
                if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    num_detections = len(results[0].boxes)
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:  # Update FPS every 10 frames
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_frame_count / elapsed
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Compact overlay
                overlay_h = 120
                cv2.rectangle(annotated_frame, (5, 5), (280, overlay_h), (0, 0, 0), -1)
                cv2.rectangle(annotated_frame, (5, 5), (280, overlay_h), (0, 255, 0), 2)
                
                y = 25
                cv2.putText(annotated_frame, f"Items: {num_detections}", 
                           (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"FPS: {int(current_fps)}", 
                           (15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Size: {current_imgsz}", 
                           (15, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, "Q:Quit S:Save 1-5:Speed", 
                           (15, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                display_frame = annotated_frame
            else:
                display_frame = frame
            
            cv2.imshow('Menu Item Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\nQuitting...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_path = f"screenshots/screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
            # Speed presets
            elif key == ord('1'):  # FASTEST
                current_imgsz = 160
                skip_frames = 1
                print("Mode: FASTEST (160px, skip frames)")
            elif key == ord('2'):  # FAST
                current_imgsz = 256
                skip_frames = 0
                print("Mode: FAST (256px)")
            elif key == ord('3'):  # BALANCED (default)
                current_imgsz = 320
                skip_frames = 0
                print("Mode: BALANCED (320px)")
            elif key == ord('4'):  # QUALITY
                current_imgsz = 480
                skip_frames = 0
                print("Mode: QUALITY (480px)")
            elif key == ord('5'):  # BEST
                current_imgsz = 640
                skip_frames = 0
                print("Mode: BEST QUALITY (640px) - slow!")
            elif key == ord('6'):  # BEST
                current_imgsz = 1280
                skip_frames = 0
                print("Mode: BEST QUALITY (1280px) - slow!")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Camera closed. Total frames: {frame_count}")

if __name__ == "__main__":
    run_camera_detection()