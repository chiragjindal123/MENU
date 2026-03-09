import { useState } from 'react'
import './Components.css'

function EditControls({ selectedBox, menuItems, onUpdate, onDelete, onCopy, onDeselect }) {
  const [itemName, setItemName] = useState(selectedBox?.itemName || '')
  const [option, setOption] = useState(
    selectedBox ? menuItems[selectedBox.itemName]?.checkboxes[selectedBox.checkboxIndex]?.option : ''
  )

  if (!selectedBox) return null

  const handleUpdate = () => {
    if (itemName.trim() && option.trim()) {
      const currentBox = menuItems[selectedBox.itemName].checkboxes[selectedBox.checkboxIndex]
      onUpdate(selectedBox.itemName, selectedBox.checkboxIndex, itemName.trim(), option.trim(), currentBox.bbox)
      onDeselect()
    }
  }

  const handleDelete = () => {
    if (window.confirm(`Delete ${selectedBox.itemName}:${option}?`)) {
      onDelete(selectedBox.itemName, selectedBox.checkboxIndex)
      onDeselect()
    }
  }

  const handleCopy = () => {
    onCopy(selectedBox.itemName, selectedBox.checkboxIndex)
    onDeselect()
  }

  return (
    <div className="edit-controls-overlay">
      <div className="edit-controls-card">
        <div className="edit-header">
          <h3>✏️ Edit Checkbox</h3>
          <button onClick={onDeselect} className="close-btn">✕</button>
        </div>

        <div className="edit-form">
          <div className="form-group">
            <label>Item Name:</label>
            <input
              type="text"
              value={itemName}
              onChange={(e) => setItemName(e.target.value)}
              placeholder="e.g., apple"
            />
          </div>

          <div className="form-group">
            <label>Option (Size/Type):</label>
            <input
              type="text"
              value={option}
              onChange={(e) => setOption(e.target.value)}
              placeholder="e.g., S, M, L"
            />
          </div>

          <div className="edit-actions">
            <button onClick={handleUpdate} className="btn btn-update">
              💾 Update
            </button>
            <button onClick={handleCopy} className="btn btn-copy">
              📋 Copy
            </button>
            <button onClick={handleDelete} className="btn btn-delete">
              🗑️ Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EditControls