import { useState } from 'react'
import './Components.css'

function ItemDetailsForm({ onSave, onCancel, boxSize }) {
  const [itemName, setItemName] = useState('')
  const [option, setOption] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    
    if (!itemName.trim()) {
      setError('Item name cannot be empty')
      return
    }

    const finalOption = option.trim() || 'default'
    onSave(itemName.trim(), finalOption)
    
    // Reset form
    setItemName('')
    setOption('')
    setError('')
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>📦 New Checkbox Details</h2>
        
        {boxSize && (
          <div className="box-info">
            Box size: {boxSize.width.toFixed(0)} × {boxSize.height.toFixed(0)} pixels
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="itemName">Item Name *</label>
            <input
              type="text"
              id="itemName"
              value={itemName}
              onChange={(e) => {
                setItemName(e.target.value)
                setError('')
              }}
              placeholder="e.g., apple, burger, pizza"
              autoFocus
            />
            <small>The name of the menu item (required)</small>
          </div>

          <div className="form-group">
            <label htmlFor="option">Option Name</label>
            <input
              type="text"
              id="option"
              value={option}
              onChange={(e) => setOption(e.target.value)}
              placeholder="e.g., S, M, L (optional)"
            />
            <small>Size or variant (leave empty for default)</small>
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="form-actions">
            <button type="button" onClick={onCancel} className="btn-cancel">
              ❌ Cancel
            </button>
            <button type="submit" className="btn-submit">
              ✅ Save Checkbox
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default ItemDetailsForm