import { useState } from 'react'
import './Components.css'

function Toolbox({ onQuickBox, selectedBox, onEdit, onCopy, onDelete, imageInfo, resizeMode, onToggleResize }) {
  const [boxSize, setBoxSize] = useState('medium')
  const [quickBoxMode, setQuickBoxMode] = useState(false)
  
  const boxSizes = {
    tiny: { width: 20, height: 20, label: 'Tiny (20×20)' },
    small: { width: 30, height: 30, label: 'Small (30×30)' },
    medium: { width: 40, height: 40, label: 'Medium (40×40)' },
    large: { width: 50, height: 50, label: 'Large (50×50)' },
    xlarge: { width: 60, height: 60, label: 'X-Large (60×60)' }
  }

  const handleQuickBox = (multiplier = 1) => {
    const size = boxSizes[boxSize]
    setQuickBoxMode(true)
    onQuickBox({
      width: size.width,
      height: size.height,
      count: multiplier
    })
  }

  return (
    <div className="toolbox-card">
      <h3>🛠️ Toolbox</h3>
      
      {/* Box Size Selector */}
      <div className="tool-section">
        <label className="tool-label">Box Size:</label>
        <select 
          value={boxSize} 
          onChange={(e) => setBoxSize(e.target.value)}
          className="size-select"
        >
          {Object.entries(boxSizes).map(([key, size]) => (
            <option key={key} value={key}>{size.label}</option>
          ))}
        </select>
      </div>

      {/* Quick Add Boxes */}
      <div className="tool-section">
        <label className="tool-label">Quick Add (Click canvas after selecting):</label>
        <div className="quick-boxes">
          {[1, 2, 3, 4, 5].map(num => (
            <button
              key={num}
              onClick={() => handleQuickBox(num)}
              className={`quick-box-btn ${quickBoxMode ? 'active' : ''}`}
              title={`Click canvas to add ${num}× box${num > 1 ? 'es' : ''}`}
            >
              <div className="box-preview" style={{
                width: `${Math.min(boxSizes[boxSize].width * 0.4, 40)}px`,
                height: `${Math.min(boxSizes[boxSize].height * 0.4, 40)}px`
              }}></div>
              <span>×{num}</span>
            </button>
          ))}
        </div>
        {quickBoxMode && (
          <div className="quick-box-hint">
            ✨ Click on canvas to place boxes
          </div>
        )}
      </div>

      {/* Edit Controls — only when a box is selected */}
      {selectedBox && (
        <div className={`tool-section ${resizeMode ? 'resize-active-tools' : 'selected-tools'}`}>
          <label className="tool-label">
            Selected: {selectedBox.itemName}
          </label>
          <div className="edit-buttons">
            <button onClick={onEdit} className="tool-btn edit-btn">
              ✏️ Edit
            </button>

            {/* Resize Toggle */}
            <button 
              onClick={onToggleResize} 
              className={`tool-btn ${resizeMode ? 'resize-active-btn' : 'resize-btn'}`}
            >
              {resizeMode ? '🔧 Exit Resize' : '🔧 Resize'}
            </button>

            <button onClick={onCopy} className="tool-btn copy-btn" title="Ctrl+C">
              📋 Copy
            </button>
            <button onClick={onDelete} className="tool-btn delete-btn" title="Delete">
              🗑️ Delete
            </button>
          </div>

          {resizeMode && (
            <div className="resize-info">
              🔧 Drag corners or edges to resize the box. Click "Exit Resize" when done.
            </div>
          )}
        </div>
      )}

      {/* Keyboard Shortcuts */}
      <div className="tool-section">
        <label className="tool-label">Shortcuts:</label>
        <div className="shortcuts-list">
          <div className="shortcut-item">
            <kbd>Ctrl+C</kbd>
            <span>Copy selected box</span>
          </div>
          <div className="shortcut-item">
            <kbd>Ctrl+V</kbd>
            <span>Paste copied box</span>
          </div>
          <div className="shortcut-item">
            <kbd>R</kbd>
            <span>Toggle resize mode</span>
          </div>
          <div className="shortcut-item">
            <kbd>Delete</kbd>
            <span>Delete selected</span>
          </div>
          <div className="shortcut-item">
            <kbd>Esc</kbd>
            <span>Deselect / cancel</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Toolbox