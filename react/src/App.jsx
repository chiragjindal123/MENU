import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import BoundingBoxCanvas from './components/BoundingBoxCanvas'
import ItemDetailsForm from './components/ItemDetailsForm'
import DownloadButton from './components/DownloadButton'
import { useMenuItems } from './hooks/useMenuItems'
import { useBoundingBox } from './hooks/useBoundingBox'
import './App.css'

function App() {
  const [uploadedImage, setUploadedImage] = useState(null)
  const [imageInfo, setImageInfo] = useState(null)
  
  const { menuItems, addMenuItem, removeLastItem, resetItems } = useMenuItems()
  const {
    isDrawing,
    currentBox,
    tempBox,
    showForm,
    startDrawing,
    updateDrawing,
    finishDrawing,
    cancelBox,
    saveBox
  } = useBoundingBox()

  const handleImageUpload = (imageData, info) => {
    setUploadedImage(imageData)
    setImageInfo(info)
    resetItems() // Reset items when new image is uploaded
  }

  const handleSaveItem = (itemName, option) => {
    if (tempBox && imageInfo) {
      const yoloBox = saveBox(itemName, option, imageInfo)
      addMenuItem(itemName, option, yoloBox)
    }
  }

  const handleUndo = () => {
    removeLastItem()
  }

  const handleReset = () => {
    if (window.confirm('⚠️ Reset all mappings? This cannot be undone!')) {
      resetItems()
    }
  }

  const totalCheckboxes = Object.values(menuItems).reduce(
    (sum, item) => sum + item.checkboxes.length,
    0
  )

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎨 Menu Mapping Tool</h1>
        <p>Draw bounding boxes around checkboxes to create menu mapping</p>
      </header>

      <div className="app-container">
        {/* Left Panel - Image Upload & Canvas */}
        <div className="left-panel">
          {!uploadedImage ? (
            <ImageUploader onImageUpload={handleImageUpload} />
          ) : (
            <BoundingBoxCanvas
              image={uploadedImage}
              imageInfo={imageInfo}
              menuItems={menuItems}
              isDrawing={isDrawing}
              currentBox={currentBox}
              onMouseDown={startDrawing}
              onMouseMove={updateDrawing}
              onMouseUp={finishDrawing}
            />
          )}

          {/* Action Buttons */}
          {uploadedImage && (
            <div className="action-buttons">
              <button onClick={handleUndo} className="btn btn-warning" disabled={totalCheckboxes === 0}>
                ↩️ Undo Last
              </button>
              <button onClick={handleReset} className="btn btn-danger" disabled={totalCheckboxes === 0}>
                🔄 Reset All
              </button>
              <button onClick={() => setUploadedImage(null)} className="btn btn-secondary">
                📁 New Image
              </button>
            </div>
          )}
        </div>

        {/* Right Panel - Info & Controls */}
        <div className="right-panel">
          {/* Statistics */}
          <div className="stats-card">
            <h3>📊 Statistics</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Total Items:</span>
                <span className="stat-value">{Object.keys(menuItems).length}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Total Checkboxes:</span>
                <span className="stat-value">{totalCheckboxes}</span>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="instructions-card">
            <h3>📋 Instructions</h3>
            <ol>
              <li>Upload a blank menu image</li>
              <li><strong>Click and drag</strong> to draw a box around ONE checkbox</li>
              <li>Enter item name and option (e.g., "apple", "S")</li>
              <li>Repeat for all checkboxes</li>
              <li>Download the JSON mapping</li>
            </ol>
          </div>

          {/* Mapping Summary */}
          {Object.keys(menuItems).length > 0 && (
            <div className="summary-card">
              <h3>🗂️ Current Mapping</h3>
              <div className="mapping-list">
                {Object.entries(menuItems).map(([itemName, itemData]) => (
                  <div key={itemName} className="mapping-item">
                    <strong>{itemName}:</strong>
                    <span className="options">
                      {itemData.checkboxes.map(cb => cb.option).join(', ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Download Button */}
          {totalCheckboxes > 0 && imageInfo && (
            <DownloadButton menuItems={menuItems} imageInfo={imageInfo} />
          )}
        </div>
      </div>

      {/* Item Details Form Modal */}
      {showForm && (
        <ItemDetailsForm
          onSave={handleSaveItem}
          onCancel={cancelBox}
          boxSize={tempBox ? {
            width: tempBox.x2 - tempBox.x1,
            height: tempBox.y2 - tempBox.y1
          } : null}
        />
      )}
    </div>
  )
}

export default App