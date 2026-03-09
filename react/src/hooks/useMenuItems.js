import { useState } from 'react'

export function useMenuItems() {
  const [menuItems, setMenuItems] = useState({})

  const addMenuItem = (itemName, option, yoloBox) => {
    setMenuItems(prev => {
      const newItems = { ...prev }
      
      if (!newItems[itemName]) {
        newItems[itemName] = { checkboxes: [] }
      }
      
      const isDuplicate = newItems[itemName].checkboxes.some(cb => {
        const tolerance = 0.001
        return Math.abs(cb.bbox[0] - yoloBox[0]) < tolerance &&
               Math.abs(cb.bbox[1] - yoloBox[1]) < tolerance &&
               Math.abs(cb.bbox[2] - yoloBox[2]) < tolerance &&
               Math.abs(cb.bbox[3] - yoloBox[3]) < tolerance
      })
      
      if (isDuplicate) {
        console.warn('⚠️ Duplicate checkbox detected, skipping...')
        return prev
      }
      
      newItems[itemName].checkboxes.push({
        option: option,
        bbox: yoloBox
      })
      
      return newItems
    })
  }

  const updateMenuItem = (itemName, checkboxIndex, newItemName, newOption, newYoloBox) => {
    setMenuItems(prev => {
      const newItems = { ...prev }
      
      if (itemName !== newItemName) {
        newItems[itemName].checkboxes.splice(checkboxIndex, 1)
        if (newItems[itemName].checkboxes.length === 0) {
          delete newItems[itemName]
        }
        if (!newItems[newItemName]) {
          newItems[newItemName] = { checkboxes: [] }
        }
        newItems[newItemName].checkboxes.push({
          option: newOption,
          bbox: newYoloBox
        })
      } else {
        newItems[itemName].checkboxes[checkboxIndex] = {
          option: newOption,
          bbox: newYoloBox
        }
      }
      
      return newItems
    })
  }

  const deleteMenuItem = (itemName, checkboxIndex) => {
    setMenuItems(prev => {
      const newItems = JSON.parse(JSON.stringify(prev)) // deep clone
      if (!newItems[itemName]) return prev
      
      newItems[itemName].checkboxes.splice(checkboxIndex, 1)
      if (newItems[itemName].checkboxes.length === 0) {
        delete newItems[itemName]
      }
      return newItems
    })
  }

  const copyMenuItem = (itemName, checkboxIndex) => {
    setMenuItems(prev => {
      const newItems = JSON.parse(JSON.stringify(prev)) // deep clone
      const original = newItems[itemName].checkboxes[checkboxIndex]
      const copiedBox = [...original.bbox]
      copiedBox[0] += 0.02
      copiedBox[1] += 0.02
      
      newItems[itemName].checkboxes.push({
        option: original.option + '_copy',
        bbox: copiedBox
      })
      
      return newItems
    })
  }

  /**
   * Move box by pixel delta. scale = actual canvas scale factor.
   */
  const moveBox = (itemName, checkboxIndex, deltaX, deltaY, imageInfo, scale) => {
    setMenuItems(prev => {
      const newItems = JSON.parse(JSON.stringify(prev)) // deep clone
      const checkbox = newItems[itemName]?.checkboxes[checkboxIndex]
      if (!checkbox) return prev
      
      // Convert canvas pixel delta → normalized delta
      const normDeltaX = deltaX / (imageInfo.width * scale)
      const normDeltaY = deltaY / (imageInfo.height * scale)
      
      const newBox = [...checkbox.bbox]
      newBox[0] += normDeltaX
      newBox[1] += normDeltaY
      
      // Clamp center so box stays inside image
      const halfW = newBox[2] / 2
      const halfH = newBox[3] / 2
      newBox[0] = Math.max(halfW, Math.min(1 - halfW, newBox[0]))
      newBox[1] = Math.max(halfH, Math.min(1 - halfH, newBox[1]))
      
      newItems[itemName].checkboxes[checkboxIndex] = {
        ...checkbox,
        bbox: newBox
      }
      
      return newItems
    })
  }

  /**
   * Resize box from any of the 8 handles: nw, ne, sw, se, n, s, w, e
   */
  const resizeBox = (itemName, checkboxIndex, handle, deltaX, deltaY, imageInfo, scale) => {
    setMenuItems(prev => {
      const newItems = JSON.parse(JSON.stringify(prev)) // deep clone
      const checkbox = newItems[itemName]?.checkboxes[checkboxIndex]
      if (!checkbox) return prev
      
      // Convert canvas pixel delta → normalized delta
      const ndx = deltaX / (imageInfo.width * scale)
      const ndy = deltaY / (imageInfo.height * scale)
      
      const [cx, cy, w, h] = checkbox.bbox
      
      // Convert YOLO center format to edges
      let left   = cx - w / 2
      let top    = cy - h / 2
      let right  = cx + w / 2
      let bottom = cy + h / 2
      
      // Move the appropriate edges based on handle
      switch (handle) {
        case 'nw':
          left += ndx
          top += ndy
          break
        case 'ne':
          right += ndx
          top += ndy
          break
        case 'sw':
          left += ndx
          bottom += ndy
          break
        case 'se':
          right += ndx
          bottom += ndy
          break
        case 'n':
          top += ndy
          break
        case 's':
          bottom += ndy
          break
        case 'w':
          left += ndx
          break
        case 'e':
          right += ndx
          break
        default:
          return prev
      }
      
      // Enforce minimum size (in normalized coords)
      const minSize = 0.005
      if (right - left < minSize) {
        if (handle.includes('w')) left = right - minSize
        else right = left + minSize
      }
      if (bottom - top < minSize) {
        if (handle.includes('n') || handle === 'n') top = bottom - minSize
        else bottom = top + minSize
      }
      
      // Clamp to image bounds
      left   = Math.max(0, left)
      top    = Math.max(0, top)
      right  = Math.min(1, right)
      bottom = Math.min(1, bottom)
      
      // Convert back to YOLO center format
      const newW = right - left
      const newH = bottom - top
      const newCx = left + newW / 2
      const newCy = top + newH / 2
      
      newItems[itemName].checkboxes[checkboxIndex] = {
        ...checkbox,
        bbox: [newCx, newCy, newW, newH]
      }
      
      return newItems
    })
  }

  const removeLastItem = () => {
    setMenuItems(prev => {
      const newItems = JSON.parse(JSON.stringify(prev)) // deep clone
      const keys = Object.keys(newItems)
      if (keys.length === 0) return prev
      
      const lastKey = keys[keys.length - 1]
      
      newItems[lastKey].checkboxes.pop()
      if (newItems[lastKey].checkboxes.length === 0) {
        delete newItems[lastKey]
      }
      
      return newItems
    })
  }

  const resetItems = () => {
    setMenuItems({})
  }

  return {
    menuItems,
    addMenuItem,
    updateMenuItem,
    deleteMenuItem,
    copyMenuItem,
    moveBox,
    resizeBox,
    removeLastItem,
    resetItems
  }
}