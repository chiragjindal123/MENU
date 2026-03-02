import { useRef, useEffect, useState } from 'react'
import { drawBoundingBox, drawExistingBoxes } from '../utils/canvasHelper'
import './Components.css'

function BoundingBoxCanvas({ 
  image, 
  imageInfo, 
  menuItems, 
  isDrawing, 
  currentBox,
  onMouseDown,
  onMouseMove,
  onMouseUp
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [scale, setScale] = useState(1)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })

  // Calculate canvas size and scale
  useEffect(() => {
    if (!imageInfo || !containerRef.current) return

    const container = containerRef.current
    const maxWidth = container.clientWidth - 40 // padding
    const maxHeight = 800

    const widthScale = maxWidth / imageInfo.width
    const heightScale = maxHeight / imageInfo.height
    const newScale = Math.min(widthScale, heightScale, 1)

    setScale(newScale)
    setCanvasSize({
      width: imageInfo.width * newScale,
      height: imageInfo.height * newScale
    })
    
    console.log(`📐 Canvas Scale: ${newScale.toFixed(4)}, Display: ${Math.round(imageInfo.width * newScale)}x${Math.round(imageInfo.height * newScale)}`)
  }, [imageInfo])

  // Draw image and boxes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !image) return

    const ctx = canvas.getContext('2d')
    const img = new Image()
    
    img.onload = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // Draw image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      
      // Draw existing boxes
      drawExistingBoxes(ctx, menuItems, imageInfo, scale)
      
      // Draw current box being drawn
      if (isDrawing && currentBox) {
        drawBoundingBox(ctx, currentBox, 'rgba(0, 255, 0, 0.3)', '#00ff00', 3)
      }
    }
    
    img.src = image
  }, [image, menuItems, isDrawing, currentBox, scale, imageInfo])

  const getMousePos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left),
      y: (e.clientY - rect.top)
    }
  }

  const handleMouseDown = (e) => {
    const pos = getMousePos(e)
    onMouseDown(pos, scale) // Pass scale to hook
  }

  const handleMouseMove = (e) => {
    if (isDrawing) {
      const pos = getMousePos(e)
      onMouseMove(pos)
    }
  }

  const handleMouseUp = (e) => {
    if (isDrawing) {
      const pos = getMousePos(e)
      onMouseUp(pos, imageInfo)
    }
  }

  return (
    <div className="canvas-container" ref={containerRef}>
      <div className="canvas-info">
        <span>📐 Original: {imageInfo?.width} × {imageInfo?.height}px</span>
        <span>🔍 Display: {canvasSize.width.toFixed(0)} × {canvasSize.height.toFixed(0)}px</span>
        <span>📊 Scale: {(scale * 100).toFixed(1)}%</span>
      </div>
      
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className="drawing-canvas"
      />
      
      <div className="canvas-hint">
        💡 Click and drag to draw a bounding box around a checkbox
      </div>
    </div>
  )
}

export default BoundingBoxCanvas