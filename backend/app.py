from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys

# Add parent directory to path to import from root
sys.path.append(str(Path(__file__).parent.parent))

from backend.order_detector import OrderDetector

app = FastAPI(title="Menu Order Detection API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path (adjust as needed)
MODEL_PATH = Path(__file__).parent.parent / "handwritten_best.pt"

class OrderDetectionRequest(BaseModel):
    """Request model for order detection"""
    image: str  # Base64 encoded image
    mapping: dict  # Menu mapping JSON

class OrderDetectionResponse(BaseModel):
    """Response model for order detection"""
    success: bool
    order: dict
    total_items: int
    total_quantity: int
    unmatched_marks: list
    message: str = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Menu Order Detection API",
        "version": "1.0.0",
        "model_loaded": MODEL_PATH.exists()
    }


@app.post("/api/detect-order", response_model=OrderDetectionResponse)
async def detect_order(request: OrderDetectionRequest):
    """
    Detect order from filled menu image
    
    Args:
        image: Base64 encoded image string
        mapping: Menu mapping JSON object
    
    Returns:
        Detected order with quantities
    """
    try:
        # Validate model exists
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"YOLO model not found at {MODEL_PATH}"
            )
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            image_data = request.image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
        
        # Validate mapping
        if not request.mapping or 'menu_mapping' not in request.mapping:
            raise HTTPException(
                status_code=400,
                detail="Invalid mapping format. Must contain 'menu_mapping' key"
            )
        
        # Initialize detector
        detector = OrderDetector(
            model_path=str(MODEL_PATH),
            mapping_data=request.mapping
        )
        
        # Process order
        order, unmatched = detector.process_order(cv_image)
        
        # Calculate statistics
        total_items = len(order)
        total_quantity = sum(
            sum(opt_data['quantity'] for opt_data in item.values())
            for item in order.values()
        )
        
        return OrderDetectionResponse(
            success=True,
            order=order,
            total_items=total_items,
            total_quantity=total_quantity,
            unmatched_marks=unmatched,
            message=f"Successfully detected {total_items} items with total quantity {total_quantity}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Order detection failed: {str(e)}"
        )


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Alternative endpoint: Upload image as multipart/form-data
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image_base64": f"data:image/jpeg;base64,{img_str}",
            "size": {
                "width": image.width,
                "height": image.height
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Menu Order Detection API...")
    print(f"📦 Model path: {MODEL_PATH}")
    print(f"✅ Model exists: {MODEL_PATH.exists()}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )