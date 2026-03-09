import { useState } from 'react'

export function useBoxManagement() {
  const [selectedBox, setSelectedBox] = useState(null) // { itemName, checkboxIndex }
  const [editMode, setEditMode] = useState(null) // 'move' | 'resize' | null
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const [resizeHandle, setResizeHandle] = useState(null) // 'nw' | 'ne' | 'sw' | 'se'

  const selectBox = (itemName, checkboxIndex) => {
    setSelectedBox({ itemName, checkboxIndex })
    setEditMode(null)
  }

  const deselectBox = () => {
    setSelectedBox(null)
    setEditMode(null)
    setIsDragging(false)
    setDragStart(null)
    setResizeHandle(null)
  }

  const startMove = () => {
    setEditMode('move')
  }

  const startResize = (handle) => {
    setEditMode('resize')
    setResizeHandle(handle)
  }

  const startDrag = (pos) => {
    setIsDragging(true)
    setDragStart(pos)
  }

  const endDrag = () => {
    setIsDragging(false)
    setDragStart(null)
  }

  return {
    selectedBox,
    editMode,
    isDragging,
    dragStart,
    resizeHandle,
    selectBox,
    deselectBox,
    startMove,
    startResize,
    startDrag,
    endDrag
  }
}