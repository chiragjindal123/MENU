import { useRef, useEffect, useState, useCallback } from 'react'
import { drawBoundingBox, drawExistingBoxes, drawExistingBoxesWithResize, findBoxAtPoint, getResizeHandle, getHandleCursor, yoloToPixel } from '../utils/canvasHelper'
import './Components.css'

function BoundingBoxCanvas({ 
  image, 
  imageInfo, 
  menuItems, 
  isDrawing, 
  currentBox,
  selectedBox,
  quickBoxSize,
  resizeMode,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onBoxSelect,
  onBoxMove,
  onBoxResize,
  onBoxDeselect,
  onQuickBoxPlace,
  onScaleChange
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const animationFrameRef = useRef(null)
  const scaleRef = useRef(1)
  const [scale, setScale] = useState(1)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [cursor, setCursor] = useState('crosshair')
  const [pendingMove, setPendingMove] = useState(null)

  const dragRef = useRef({
    isDragging: false,
    dragStart: null,
    dragMode: null,
    resizeHandle: null,
  })

  useEffect(() => {
    if (!imageInfo || !containerRef.current) return

    const container = containerRef.current
    const maxWidth = container.clientWidth - 40
    const maxHeight = 800

    const widthScale = maxWidth / imageInfo.width
    const heightScale = maxHeight / imageInfo.height
    const newScale = Math.min(widthScale, heightScale, 1)

    scaleRef.current = newScale
    setScale(newScale)
    setCanvasSize({
      width: imageInfo.width * newScale,
      height: imageInfo.height * newScale
    })

    if (onScaleChange) {
      onScaleChange(newScale)
    }
  }, [imageInfo, onScaleChange])

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !image) return

    const ctx = canvas.getContext('2d')
    const img = new Image()
    
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      
      // Only show resize handles when resizeMode is ON
      drawExistingBoxesWithResize(ctx, menuItems, imageInfo, scale, selectedBox, resizeMode)
      
      if (isDrawing && currentBox) {
        drawBoundingBox(ctx, currentBox, 'rgba(0, 255, 0, 0.3)', '#00ff00', 3)
      }

      if (quickBoxSize && pendingMove) {
        const { x, y } = pendingMove
        const boxWidthCanvas = (quickBoxSize.width / imageInfo.width) * canvasSize.width
        const boxHeightCanvas = (quickBoxSize.height / imageInfo.height) * canvasSize.height
        
        const previewBox = {
          x1: x - boxWidthCanvas/2,
          y1: y - boxHeightCanvas/2,
          x2: x + boxWidthCanvas/2,
          y2: y + boxHeightCanvas/2
        }
        drawBoundingBox(ctx, previewBox, 'rgba(255, 165, 0, 0.3)', '#FFA500', 2)

        ctx.strokeStyle = '#FFA500'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x - 10, y)
        ctx.lineTo(x + 10, y)
        ctx.moveTo(x, y - 10)
        ctx.lineTo(x, y + 10)
        ctx.stroke()
      }
    }
    img.src = image
  }, [image, menuItems, isDrawing, currentBox, scale, imageInfo, selectedBox, quickBoxSize, pendingMove, canvasSize, resizeMode])

  useEffect(() => {
    redrawCanvas()
  }, [redrawCanvas])

  const getMousePos = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }
  }

  const getSelectedPixelBox = useCallback(() => {
    if (!selectedBox || !imageInfo) return null
    const data = menuItems[selectedBox.itemName]?.checkboxes[selectedBox.checkboxIndex]
    if (!data) return null
    return yoloToPixel(data.bbox, imageInfo, scale)
  }, [selectedBox, menuItems, imageInfo, scale])

  const handleMouseDown = (e) => {
    const pos = getMousePos(e)

    // Quick box placement
    if (quickBoxSize && onQuickBoxPlace) {
      onQuickBoxPlace(pos, scaleRef.current)
      return
    }

    // If a box is already selected
    if (selectedBox) {
      const selBox = getSelectedPixelBox()
      if (selBox) {
        // Only allow resize drag if resizeMode is ON
        if (resizeMode) {
          const handle = getResizeHandle(pos, selBox)
          if (handle) {
            dragRef.current = {
              isDragging: true,
              dragStart: pos,
              dragMode: 'resize',
              resizeHandle: handle,
            }
            setCursor(getHandleCursor(handle))
            return
          }
        }

        // Check if inside selected box → ALWAYS allow move
        if (pos.x >= selBox.x1 && pos.x <= selBox.x2 &&
            pos.y >= selBox.y1 && pos.y <= selBox.y2) {
          dragRef.current = {
            isDragging: true,
            dragStart: pos,
            dragMode: 'move',
            resizeHandle: null,
          }
          setCursor('grabbing')
          return
        }
      }
    }

    // Check if clicking a different box
    const clickedBox = findBoxAtPoint(pos, menuItems, imageInfo, scale, selectedBox)
    if (clickedBox) {
      onBoxSelect(clickedBox.itemName, clickedBox.checkboxIndex)

      // Start move immediately on the newly selected box
      dragRef.current = {
        isDragging: true,
        dragStart: pos,
        dragMode: 'move',
        resizeHandle: null,
      }
      setCursor('grabbing')
      return
    }

    // Nothing clicked → deselect and start drawing
    onBoxDeselect()
    onMouseDown(pos, scaleRef.current)
  }

  const handleMouseMove = useCallback((e) => {
    const pos = getMousePos(e)

    // Quick box preview
    if (quickBoxSize) {
      setPendingMove(pos)
      setCursor('copy')
      return
    }

    const drag = dragRef.current

    if (drag.isDragging && selectedBox && drag.dragStart) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }

      const startPos = drag.dragStart

      animationFrameRef.current = requestAnimationFrame(() => {
        const deltaX = pos.x - startPos.x
        const deltaY = pos.y - startPos.y

        if (drag.dragMode === 'resize' && drag.resizeHandle) {
          setCursor(getHandleCursor(drag.resizeHandle))
          onBoxResize(
            selectedBox.itemName,
            selectedBox.checkboxIndex,
            drag.resizeHandle,
            deltaX,
            deltaY
          )
        } else if (drag.dragMode === 'move') {
          setCursor('grabbing')
          onBoxMove(
            selectedBox.itemName,
            selectedBox.checkboxIndex,
            deltaX,
            deltaY
          )
        }

        dragRef.current = { ...dragRef.current, dragStart: pos }
      })
    } else if (isDrawing) {
      onMouseMove(pos)
      setCursor('crosshair')
    } else {
      // Hover detection
      if (selectedBox) {
        const selBox = getSelectedPixelBox()
        if (selBox) {
          // Show resize cursors ONLY when resize mode is on
          if (resizeMode) {
            const handle = getResizeHandle(pos, selBox)
            if (handle) {
              setCursor(getHandleCursor(handle))
              return
            }
          }
          // Inside box → grab cursor
          if (pos.x >= selBox.x1 && pos.x <= selBox.x2 &&
              pos.y >= selBox.y1 && pos.y <= selBox.y2) {
            setCursor('grab')
            return
          }
        }
      }

      // Check any box under cursor
      const hoveredBox = findBoxAtPoint(pos, menuItems, imageInfo, scale, selectedBox)
      if (hoveredBox) {
        setCursor('pointer')
      } else {
        setCursor('crosshair')
      }
    }
  }, [selectedBox, isDrawing, quickBoxSize, menuItems, imageInfo, scale, resizeMode, onBoxResize, onBoxMove, onMouseMove, getSelectedPixelBox])

  const handleMouseUp = (e) => {
    const pos = getMousePos(e)

    if (isDrawing) {
      onMouseUp(pos, imageInfo)
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    dragRef.current = {
      isDragging: false,
      dragStart: null,
      dragMode: null,
      resizeHandle: null,
    }

    // Restore hover cursor
    if (!quickBoxSize) {
      const selBox = getSelectedPixelBox()
      if (selBox) {
        if (resizeMode) {
          const handle = getResizeHandle(pos, selBox)
          if (handle) {
            setCursor(getHandleCursor(handle))
            return
          }
        }
        if (pos.x >= selBox.x1 && pos.x <= selBox.x2 &&
            pos.y >= selBox.y1 && pos.y <= selBox.y2) {
          setCursor('grab')
          return
        }
      }
      setCursor('crosshair')
    }
  }

  const handleMouseLeave = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    dragRef.current = {
      isDragging: false,
      dragStart: null,
      dragMode: null,
      resizeHandle: null,
    }
    setCursor('crosshair')
  }

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  const drag = dragRef.current

  return (
    <div className="canvas-container" ref={containerRef}>
      <div className="canvas-info">
        <span>📐 {imageInfo?.width} × {imageInfo?.height}px</span>
        <span>📊 {(scale * 100).toFixed(0)}%</span>
        {selectedBox && (
          <span className="selected-indicator">
            ✏️ {selectedBox.itemName}:{menuItems[selectedBox.itemName]?.checkboxes[selectedBox.checkboxIndex]?.option}
          </span>
        )}
        {resizeMode && (
          <span className="resize-mode-indicator">
            🔧 RESIZE MODE
          </span>
        )}
        {quickBoxSize && (
          <span className="quick-box-indicator">
            ⚡ Quick Box: {quickBoxSize.width}×{quickBoxSize.height}px
          </span>
        )}
      </div>
      
      <canvas
        ref={canvasRef}
        width={canvasSize.width}
        height={canvasSize.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className="drawing-canvas"
        style={{ cursor }}
      />
      
      <div className="canvas-hint">
        {quickBoxSize ? (
          '⚡ Click to place box • ESC to cancel'
        ) : drag.isDragging ? (
          drag.dragMode === 'resize' ? 
            `🔧 Resizing ${drag.resizeHandle?.toUpperCase()} — release to finish` :
            '🤚 Moving — release to drop'
        ) : resizeMode && selectedBox ? (
          '🔧 RESIZE MODE: Drag corners/edges to resize • Click "Resize" again to exit'
        ) : selectedBox ? (
          '💡 Drag to move • Use Resize button to enable resizing • Click outside to deselect'
        ) : (
          '💡 Click box to select • Click and drag empty area to draw new box'
        )}
      </div>
    </div>
  )
}

export default BoundingBoxCanvas