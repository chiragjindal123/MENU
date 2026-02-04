from flask import Flask, render_template, jsonify, request, send_file
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# --- OPTIMIZED CONFIGURATION FOR MOBILE ---
INFER_WEIGHTS = "best.pt"
IMGSZ = 640  # Reduced from 1280 for better mobile performance
CONF = 0.5  # Lower threshold for better detection

# Global variables
model = YOLO(INFER_WEIGHTS)
print("✅ Model loaded!")

@app.route('/')
def index():
    return render_template('mobile.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        print(f"📸 Received image: {file.filename}")
        
        # Read and convert image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        print(f"🔍 Running detection... (image size: {img_array.shape})")
        
        # Run detection
        results = model(img_array, imgsz=IMGSZ, conf=CONF, verbose=False)
        
        detections = []
        annotated_image = None
        
        if results and len(results) > 0:
            # Get annotated image
            annotated_array = results[0].plot(font_size=12, line_width=5)
            annotated_array = cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
            annotated_img = Image.fromarray(annotated_array)
            
            # Convert to base64 for sending to browser
            buffered = io.BytesIO()
            annotated_img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            annotated_image = f"data:image/jpeg;base64,{img_base64}"
            
            # Get detection details
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x, y, w, h = box.xywh[0].tolist()
                    conf = float(box.conf[0])
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'confidence': round(conf, 3)
                    })
        
        print(f"✅ Detection complete! Found {len(detections)} items")
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'annotated_image': annotated_image
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Mobile Menu Detection Server")
    print("="*50)
    print(f"📱 Access from phone: http://192.168.50.44:5000")
    print(f"🖥️  Local access: http://127.0.0.1:5000")
    print(f"📊 Model: {INFER_WEIGHTS}")
    print(f"🎯 Confidence: {CONF}")
    print(f"📐 Image size: {IMGSZ}")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)