import { useState } from 'react'

export function useBoundingBox() {
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState(null)
  const [currentBox, setCurrentBox] = useState(null)
  const [tempBox, setTempBox] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [currentScale, setCurrentScale] = useState(1) // Store scale for conversion

  const startDrawing = (pos, scale) => {
    setIsDrawing(true)
    setStartPoint(pos)
    setCurrentBox(null)
    setCurrentScale(scale) // Save scale
  }

  const updateDrawing = (pos) => {
    if (!isDrawing || !startPoint) return
    
    setCurrentBox({
      x1: Math.min(startPoint.x, pos.x),
      y1: Math.min(startPoint.y, pos.y),
      x2: Math.max(startPoint.x, pos.x),
      y2: Math.max(startPoint.y, pos.y)
    })
  }

  const finishDrawing = (pos, imageInfo) => {
    if (!isDrawing || !startPoint) return

    const width = Math.abs(pos.x - startPoint.x)
    const height = Math.abs(pos.y - startPoint.y)

    // Validate box size
    if (width < 10 || height < 10) {
      alert('⚠️ Box too small. Draw a larger bounding box.')
      setIsDrawing(false)
      setStartPoint(null)
      setCurrentBox(null)
      return
    }

    // Store temporary box in DISPLAY coordinates
    const box = {
      x1: Math.min(startPoint.x, pos.x),
      y1: Math.min(startPoint.y, pos.y),
      x2: Math.max(startPoint.x, pos.x),
      y2: Math.max(startPoint.y, pos.y)
    }

    setTempBox(box)
    setIsDrawing(false)
    setShowForm(true)
  }

  const cancelBox = () => {
    setTempBox(null)
    setShowForm(false)
    setCurrentBox(null)
    setStartPoint(null)
  }

  const saveBox = (itemName, option, imageInfo) => {
    if (!tempBox || !imageInfo) return null

    // Convert display coordinates to YOLO format using the stored scale
    const yoloBox = pixelToYolo(tempBox, imageInfo, currentScale)

    // Reset drawing state
    setTempBox(null)
    setShowForm(false)
    setCurrentBox(null)
    setStartPoint(null)

    return yoloBox
  }

  const pixelToYolo = (box, imageInfo, scale) => {
    // Convert display coordinates back to original image coordinates
    const originalX1 = box.x1 / scale
    const originalY1 = box.y1 / scale
    const originalX2 = box.x2 / scale
    const originalY2 = box.y2 / scale

    // Calculate center and dimensions in normalized coordinates
    const x = ((originalX1 + originalX2) / 2) / imageInfo.width
    const y = ((originalY1 + originalY2) / 2) / imageInfo.height
    const w = (originalX2 - originalX1) / imageInfo.width
    const h = (originalY2 - originalY1) / imageInfo.height

    return [
      parseFloat(x.toFixed(6)),
      parseFloat(y.toFixed(6)),
      parseFloat(w.toFixed(6)),
      parseFloat(h.toFixed(6))
    ]
  }

  return {
    isDrawing,
    currentBox,
    tempBox,
    showForm,
    startDrawing,
    updateDrawing,
    finishDrawing,
    cancelBox,
    saveBox
  }
}