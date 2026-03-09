import { useState, useEffect } from 'react'
import ImageUploader from './components/ImageUploader'
import BoundingBoxCanvas from './components/BoundingBoxCanvas'
import ItemDetailsForm from './components/ItemDetailsForm'
import EditControls from './components/EditControls'
import Toolbox from './components/Toolbox'
import DownloadButton from './components/DownloadButton'
import OrderDetection from './components/OrderDetection'
import { useMenuItems } from './hooks/useMenuItems'
import { useBoundingBox } from './hooks/useBoundingBox'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('mapping')
  const [uploadedImage, setUploadedImage] = useState(null)
  const [imageInfo, setImageInfo] = useState(null)
  const [selectedBox, setSelectedBox] = useState(null)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [copiedBox, setCopiedBox] = useState(null)
  const [quickBoxSize, setQuickBoxSize] = useState(null)
  const [quickBoxCount, setQuickBoxCount] = useState(1)
  const [canvasScale, setCanvasScale] = useState(1)
  const [resizeMode, setResizeMode] = useState(false)
  
  const { 
    menuItems, 
    addMenuItem, 
    updateMenuItem,
    deleteMenuItem,
    copyMenuItem,
    moveBox,
    resizeBox,
    removeLastItem, 
    resetItems 
  } = useMenuItems()
  
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

  // Turn off resize mode when box is deselected
  useEffect(() => {
    if (!selectedBox) {
      setResizeMode(false)
    }
  }, [selectedBox])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (activeTab !== 'mapping') return

      // Don't capture keys when typing in inputs
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return

      if (e.key === 'Escape') {
        e.preventDefault()
        if (quickBoxSize) {
          setQuickBoxSize(null)
          setQuickBoxCount(1)
        } else if (resizeMode) {
          setResizeMode(false)
        } else if (showEditDialog) {
          setShowEditDialog(false)
        } else {
          setSelectedBox(null)
        }
      }

      // R — toggle resize mode
      if (e.key === 'r' || e.key === 'R') {
        if (selectedBox && !showEditDialog && !showForm) {
          e.preventDefault()
          setResizeMode(prev => !prev)
        }
      }

      if (e.ctrlKey && e.key === 'c' && selectedBox) {
        e.preventDefault()
        handleCopy()
      }

      if (e.ctrlKey && e.key === 'v' && copiedBox) {
        e.preventDefault()
        handlePaste()
      }

      if (e.key === 'Delete' && selectedBox && !showEditDialog && !showForm) {
        e.preventDefault()
        handleDelete()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedBox, copiedBox, activeTab, showEditDialog, showForm, quickBoxSize, resizeMode])

  const handleImageUpload = (imageData, info) => {
    setUploadedImage(imageData)
    setImageInfo(info)
    resetItems()
    setSelectedBox(null)
    setCopiedBox(null)
    setQuickBoxSize(null)
    setResizeMode(false)
  }

  const handleSaveItem = (itemName, option) => {
    if (tempBox && imageInfo) {
      const yoloBox = saveBox(itemName, option, imageInfo)
      addMenuItem(itemName, option, yoloBox)
    }
  }

  const handleBoxSelect = (itemName, checkboxIndex) => {
    setSelectedBox({ itemName, checkboxIndex })
    setQuickBoxSize(null)
    // Don't reset resizeMode here — let user keep it on between boxes if they want
  }

  const handleBoxDeselect = () => {
    setSelectedBox(null)
    setResizeMode(false)
  }

  const handleBoxMove = (itemName, checkboxIndex, deltaX, deltaY) => {
    if (imageInfo) {
      moveBox(itemName, checkboxIndex, deltaX, deltaY, imageInfo, canvasScale)
    }
  }

  const handleBoxResize = (itemName, checkboxIndex, handle, deltaX, deltaY) => {
    if (imageInfo) {
      resizeBox(itemName, checkboxIndex, handle, deltaX, deltaY, imageInfo, canvasScale)
    }
  }

  const handleEdit = () => {
    if (selectedBox) {
      setShowEditDialog(true)
    }
  }

  const handleToggleResize = () => {
    setResizeMode(prev => !prev)
  }

  const handleCopy = () => {
    if (selectedBox) {
      const checkbox = menuItems[selectedBox.itemName]?.checkboxes[selectedBox.checkboxIndex]
      if (checkbox) {
        setCopiedBox({
          itemName: selectedBox.itemName,
          option: checkbox.option,
          bbox: [...checkbox.bbox]
        })
      }
    }
  }

  const handlePaste = () => {
    if (copiedBox) {
      const newBbox = [...copiedBox.bbox]
      newBbox[0] += 0.03
      newBbox[1] += 0.03
      
      addMenuItem(
        copiedBox.itemName,
        copiedBox.option + '_copy',
        newBbox
      )
    }
  }

  const handleDelete = () => {
    if (selectedBox) {
      const itemName = selectedBox.itemName
      const checkbox = menuItems[itemName]?.checkboxes[selectedBox.checkboxIndex]
      
      if (window.confirm(`Delete ${itemName}:${checkbox?.option}?`)) {
        deleteMenuItem(selectedBox.itemName, selectedBox.checkboxIndex)
        setSelectedBox(null)
      }
    }
  }

  const handleQuickBox = (size) => {
    setQuickBoxSize(size)
    setQuickBoxCount(size.count || 1)
    setSelectedBox(null)
    setResizeMode(false)
  }

  const handleQuickBoxPlace = (pos, scale) => {
    if (!quickBoxSize || !imageInfo) return

    const origX = pos.x / scale
    const origY = pos.y / scale
    
    const normX = origX / imageInfo.width
    const normY = origY / imageInfo.height
    const normW = quickBoxSize.width / imageInfo.width
    const normH = quickBoxSize.height / imageInfo.height

    for (let i = 0; i < quickBoxCount; i++) {
      const offsetX = i * normW * 1.1
      const yoloBox = [
        normX + offsetX,
        normY,
        normW,
        normH
      ]
      
      addMenuItem(
        `quick_item_${Date.now()}_${i}`,
        `box_${i + 1}`,
        yoloBox
      )
    }

    setQuickBoxSize(null)
    setQuickBoxCount(1)
  }

  const handleUndo = () => {
    removeLastItem()
    setSelectedBox(null)
    setResizeMode(false)
  }

  const handleReset = () => {
    if (window.confirm('⚠️ Reset all mappings? This cannot be undone!')) {
      resetItems()
      setSelectedBox(null)
      setCopiedBox(null)
      setQuickBoxSize(null)
      setResizeMode(false)
    }
  }

  const totalCheckboxes = Object.values(menuItems).reduce(
    (sum, item) => sum + item.checkboxes.length,
    0
  )

  return (
    <div className="app">
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'mapping' ? 'active' : ''}`}
          onClick={() => setActiveTab('mapping')}
        >
          🎨 Menu Mapping
        </button>
        <button
          className={`tab-button ${activeTab === 'detection' ? 'active' : ''}`}
          onClick={() => setActiveTab('detection')}
        >
          🔍 Order Detection
        </button>
      </div>

      {activeTab === 'mapping' ? (
        <>
          <header className="app-header">
            <h1>🎨 Menu Mapping Tool</h1>
            <p>Draw bounding boxes around checkboxes • Use keyboard shortcuts for faster workflow</p>
          </header>

          <div className="app-container">
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
                  selectedBox={selectedBox}
                  quickBoxSize={quickBoxSize}
                  resizeMode={resizeMode}
                  onMouseDown={startDrawing}
                  onMouseMove={updateDrawing}
                  onMouseUp={finishDrawing}
                  onBoxSelect={handleBoxSelect}
                  onBoxMove={handleBoxMove}
                  onBoxResize={handleBoxResize}
                  onBoxDeselect={handleBoxDeselect}
                  onQuickBoxPlace={handleQuickBoxPlace}
                  onScaleChange={setCanvasScale}
                />
              )}

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

            <div className="right-panel">
              {uploadedImage && (
                <Toolbox
                  onQuickBox={handleQuickBox}
                  selectedBox={selectedBox}
                  onEdit={handleEdit}
                  onCopy={handleCopy}
                  onDelete={handleDelete}
                  imageInfo={imageInfo}
                  resizeMode={resizeMode}
                  onToggleResize={handleToggleResize}
                />
              )}

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

              <div className="instructions-card">
                <h3>📋 Instructions</h3>
                <ol>
                  <li>Upload a blank menu image</li>
                  <li><strong>Click and drag</strong> to draw boxes</li>
                  <li><strong>Click box</strong> to select → drag to move</li>
                  <li>Click <strong>🔧 Resize</strong> or press <strong>R</strong> to resize</li>
                  <li>Use <strong>Quick Box</strong> for rapid placement</li>
                  <li>Use <strong>Ctrl+C/V</strong> for copy/paste</li>
                  <li>Download the JSON mapping</li>
                </ol>
              </div>

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

              {totalCheckboxes > 0 && imageInfo && (
                <DownloadButton menuItems={menuItems} imageInfo={imageInfo} />
              )}
            </div>
          </div>

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

          {showEditDialog && selectedBox && (
            <EditControls
              selectedBox={selectedBox}
              menuItems={menuItems}
              onUpdate={updateMenuItem}
              onDelete={deleteMenuItem}
              onCopy={() => {
                copyMenuItem(selectedBox.itemName, selectedBox.checkboxIndex)
                setShowEditDialog(false)
              }}
              onDeselect={() => {
                setShowEditDialog(false)
                setSelectedBox(null)
              }}
            />
          )}
        </>
      ) : (
        <OrderDetection />
      )}
    </div>
  )
}

export default App