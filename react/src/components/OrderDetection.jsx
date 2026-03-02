import { useState, useRef, useEffect } from 'react';
import { MenuAPI } from '../services/api';
import OrderResults from './OrderResults';
import './Components.css';

function OrderDetection() {
  const [mappingFile, setMappingFile] = useState(null);
  const [menuImage, setMenuImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkBackend();
  }, []);

  const checkBackend = async () => {
    try {
      const isHealthy = await MenuAPI.healthCheck();
      setBackendStatus(isHealthy ? 'online' : 'offline');
    } catch (err) {
      console.error('Backend health check failed:', err);
      setBackendStatus('offline');
    }
  };

  const handleMappingUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/json') {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target.result);
          setMappingFile(json);
          setError(null);
        } catch (err) {
          setError('Invalid JSON file');
        }
      };
      reader.readAsText(file);
    } else {
      setError('Please upload a valid JSON file');
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setMenuImage(event.target.result);
        setError(null);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please upload a valid image file');
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      
      videoRef.current.srcObject = stream;
      setIsCameraActive(true);
      setError(null);
    } catch (err) {
      console.error('Camera error:', err);
      setError('Camera access denied or not available');
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      setError('Camera elements not ready');
      return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg');
    setMenuImage(imageData);
    
    // Stop camera
    const stream = video.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    setIsCameraActive(false);
  };

  const handleDetectOrder = async () => {
    if (!menuImage || !mappingFile) {
      setError('Please upload both menu image and mapping file');
      return;
    }

    if (backendStatus === 'offline') {
      setError('Backend server is offline. Please start the FastAPI server.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResults(null);

    try {
      console.log('📤 Sending detection request...');
      const response = await MenuAPI.detectOrder(menuImage, mappingFile);
      console.log('✅ Detection response:', response);
      setResults(response);
    } catch (err) {
      console.error('❌ Detection error:', err);
      setError(err.message || 'Order detection failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setMenuImage(null);
    setMappingFile(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="order-detection">
      <header className="detection-header">
        <h1>🔍 Order Detection</h1>
        <p>Upload filled menu to detect customer order</p>
        <div className={`backend-status ${backendStatus}`}>
          <span className="status-dot"></span>
          Backend: {backendStatus === 'online' ? 'Connected ✅' : backendStatus === 'checking' ? 'Checking...' : 'Disconnected ❌'}
        </div>
      </header>

      <div className="detection-container">
        {/* Left Panel - Upload & Camera */}
        <div className="upload-panel">
          <div className="upload-section">
            <h3>1️⃣ Upload Menu Mapping</h3>
            <input
              type="file"
              accept=".json"
              onChange={handleMappingUpload}
              className="file-input"
            />
            {mappingFile && (
              <div className="success-message">
                ✅ Mapping loaded ({Object.keys(mappingFile.menu_mapping || {}).length} items)
              </div>
            )}
          </div>

          <div className="upload-section">
            <h3>2️⃣ Capture/Upload Filled Menu</h3>
            
            {!isCameraActive && !menuImage && (
              <div className="camera-options">
                <button onClick={startCamera} className="btn btn-primary">
                  📷 Use Camera
                </button>
                <button onClick={() => fileInputRef.current?.click()} className="btn btn-secondary">
                  📁 Upload Image
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={{ display: 'none' }}
                />
              </div>
            )}

            {isCameraActive && (
              <div className="camera-view">
                <video ref={videoRef} autoPlay playsInline className="camera-preview" />
                <button onClick={capturePhoto} className="btn btn-success">
                  📸 Capture Photo
                </button>
                <button onClick={() => {
                  const stream = videoRef.current?.srcObject;
                  if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                  }
                  setIsCameraActive(false);
                }} className="btn btn-warning">
                  ❌ Cancel
                </button>
              </div>
            )}

            {menuImage && !isCameraActive && (
              <div className="image-preview">
                <img src={menuImage} alt="Filled menu" />
                <button onClick={() => setMenuImage(null)} className="btn btn-warning">
                  ❌ Remove
                </button>
              </div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {error && (
            <div className="error-banner">
              ⚠️ {error}
            </div>
          )}

          <div className="action-section">
            <button
              onClick={handleDetectOrder}
              disabled={!menuImage || !mappingFile || isProcessing || backendStatus !== 'online'}
              className="btn btn-detect"
            >
              {isProcessing ? '⏳ Processing...' : '🚀 Detect Order'}
            </button>
            
            {results && (
              <button onClick={handleReset} className="btn btn-secondary">
                🔄 New Detection
              </button>
            )}
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="results-panel">
          {results ? (
            <OrderResults results={results} />
          ) : (
            <div className="placeholder">
              <h3>📋 Order Results</h3>
              <p>Detection results will appear here</p>
              {backendStatus === 'offline' && (
                <div style={{ marginTop: '20px', padding: '15px', background: '#fee2e2', borderRadius: '8px' }}>
                  <strong>⚠️ Backend Offline</strong>
                  <p style={{ fontSize: '0.9rem', marginTop: '10px' }}>
                    Please start the FastAPI server:
                  </p>
                  <code style={{ display: 'block', marginTop: '10px', padding: '10px', background: 'white', borderRadius: '4px' }}>
                    cd backend && python app.py
                  </code>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OrderDetection;