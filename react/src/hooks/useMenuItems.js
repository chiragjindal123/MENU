import { useState } from 'react'

export function useMenuItems() {
  const [menuItems, setMenuItems] = useState({})

  const addMenuItem = (itemName, option, yoloBox) => {
    setMenuItems(prev => {
      const newItems = { ...prev }
      
      // Initialize item if doesn't exist
      if (!newItems[itemName]) {
        newItems[itemName] = { checkboxes: [] }
      }
      
      // Check for duplicate bbox (within small tolerance)
      const isDuplicate = newItems[itemName].checkboxes.some(cb => {
        const tolerance = 0.001
        return Math.abs(cb.bbox[0] - yoloBox[0]) < tolerance &&
               Math.abs(cb.bbox[1] - yoloBox[1]) < tolerance &&
               Math.abs(cb.bbox[2] - yoloBox[2]) < tolerance &&
               Math.abs(cb.bbox[3] - yoloBox[3]) < tolerance
      })
      
      if (isDuplicate) {
        console.warn('⚠️ Duplicate checkbox detected, skipping...')
        return prev // Return previous state unchanged
      }
      
      // Add new checkbox
      newItems[itemName].checkboxes.push({
        option: option,
        bbox: yoloBox
      })
      
      return newItems
    })
  }

  const removeLastItem = () => {
    setMenuItems(prev => {
      const items = { ...prev }
      const keys = Object.keys(items)
      
      if (keys.length === 0) return items
      
      const lastKey = keys[keys.length - 1]
      const lastItem = items[lastKey]
      
      if (lastItem.checkboxes.length > 0) {
        lastItem.checkboxes.pop()
        
        // Remove item if no checkboxes left
        if (lastItem.checkboxes.length === 0) {
          delete items[lastKey]
        }
      }
      
      return items
    })
  }

  const resetItems = () => {
    setMenuItems({})
  }

  return {
    menuItems,
    addMenuItem,
    removeLastItem,
    resetItems
  }
}