export function drawBoundingBox(ctx, box, fillColor, strokeColor, lineWidth = 2) {
  const width = box.x2 - box.x1
  const height = box.y2 - box.y1

  ctx.fillStyle = fillColor
  ctx.fillRect(box.x1, box.y1, width, height)

  ctx.strokeStyle = strokeColor
  ctx.lineWidth = lineWidth
  ctx.strokeRect(box.x1, box.y1, width, height)
}

/**
 * Draw boxes. Only show resize handles if resizeMode is true AND box is selected.
 */
export function drawExistingBoxesWithResize(ctx, menuItems, imageInfo, scale, selectedBox = null, resizeMode = false) {
  Object.entries(menuItems).forEach(([itemName, itemData]) => {
    itemData.checkboxes.forEach((checkbox, index) => {
      const box = yoloToPixel(checkbox.bbox, imageInfo, scale)
      
      const isSelected = selectedBox && 
                        selectedBox.itemName === itemName && 
                        selectedBox.checkboxIndex === index
      
      const fillColor = isSelected 
        ? (resizeMode ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 215, 0, 0.25)') 
        : 'rgba(0, 200, 0, 0.15)'
      const strokeColor = isSelected 
        ? (resizeMode ? '#3b82f6' : '#FFD700') 
        : '#00c800'
      const lineWidth = isSelected ? 3 : 2
      
      drawBoundingBox(ctx, box, fillColor, strokeColor, lineWidth)
      
      // Only show resize handles when resizeMode is ON and box is selected
      if (isSelected && resizeMode) {
        drawResizeHandles(ctx, box)
        drawEdgeHandles(ctx, box)
      }

      // Show move icon when selected but NOT in resize mode
      if (isSelected && !resizeMode) {
        drawMoveIcon(ctx, box)
      }
      
      // Label
      const label = `${itemName}:${checkbox.option}`
      ctx.font = 'bold 14px Arial'
      const textMetrics = ctx.measureText(label)
      const padding = 5
      const textHeight = 18
      
      ctx.fillStyle = isSelected 
        ? (resizeMode ? 'rgba(59, 130, 246, 0.9)' : 'rgba(255, 215, 0, 0.9)') 
        : 'rgba(0, 200, 0, 0.9)'
      ctx.fillRect(
        box.x1, 
        box.y1 - textHeight - padding, 
        textMetrics.width + (padding * 2), 
        textHeight + padding
      )
      
      ctx.fillStyle = isSelected ? (resizeMode ? 'white' : '#000') : 'white'
      ctx.fillText(label, box.x1 + padding, box.y1 - padding)
    })
  })
}

// Keep old function for backward compatibility
export function drawExistingBoxes(ctx, menuItems, imageInfo, scale, selectedBox = null) {
  drawExistingBoxesWithResize(ctx, menuItems, imageInfo, scale, selectedBox, false)
}

function drawMoveIcon(ctx, box) {
  const centerX = (box.x1 + box.x2) / 2
  const centerY = (box.y1 + box.y2) / 2
  const size = 8
  const arrowSize = 4

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)'
  ctx.lineWidth = 2
  ctx.beginPath()

  // Vertical line
  ctx.moveTo(centerX, centerY - size)
  ctx.lineTo(centerX, centerY + size)
  // Horizontal line
  ctx.moveTo(centerX - size, centerY)
  ctx.lineTo(centerX + size, centerY)

  // Arrows
  // Up
  ctx.moveTo(centerX - arrowSize, centerY - size + arrowSize)
  ctx.lineTo(centerX, centerY - size)
  ctx.lineTo(centerX + arrowSize, centerY - size + arrowSize)
  // Down
  ctx.moveTo(centerX - arrowSize, centerY + size - arrowSize)
  ctx.lineTo(centerX, centerY + size)
  ctx.lineTo(centerX + arrowSize, centerY + size - arrowSize)
  // Left
  ctx.moveTo(centerX - size + arrowSize, centerY - arrowSize)
  ctx.lineTo(centerX - size, centerY)
  ctx.lineTo(centerX - size + arrowSize, centerY + arrowSize)
  // Right
  ctx.moveTo(centerX + size - arrowSize, centerY - arrowSize)
  ctx.lineTo(centerX + size, centerY)
  ctx.lineTo(centerX + size - arrowSize, centerY + arrowSize)

  ctx.stroke()
}

export function drawResizeHandles(ctx, box) {
  const handleSize = 12
  const corners = getCornerPositions(box)
  
  corners.forEach(corner => {
    ctx.fillStyle = 'rgba(0,0,0,0.3)'
    ctx.fillRect(corner.x - handleSize/2 + 1, corner.y - handleSize/2 + 1, handleSize, handleSize)
    ctx.fillStyle = '#000'
    ctx.fillRect(corner.x - handleSize/2 - 1, corner.y - handleSize/2 - 1, handleSize + 2, handleSize + 2)
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(corner.x - handleSize/2, corner.y - handleSize/2, handleSize, handleSize)
  })
}

export function drawEdgeHandles(ctx, box) {
  const handleW = 10
  const handleH = 6
  const midX = (box.x1 + box.x2) / 2
  const midY = (box.y1 + box.y2) / 2
  
  const edges = [
    { x: midX, y: box.y1, w: handleW, h: handleH },
    { x: midX, y: box.y2, w: handleW, h: handleH },
    { x: box.x1, y: midY, w: handleH, h: handleW },
    { x: box.x2, y: midY, w: handleH, h: handleW },
  ]
  
  edges.forEach(e => {
    ctx.fillStyle = '#000'
    ctx.fillRect(e.x - e.w/2 - 1, e.y - e.h/2 - 1, e.w + 2, e.h + 2)
    ctx.fillStyle = '#60a5fa'
    ctx.fillRect(e.x - e.w/2, e.y - e.h/2, e.w, e.h)
  })
}

function getCornerPositions(box) {
  return [
    { x: box.x1, y: box.y1 },
    { x: box.x2, y: box.y1 },
    { x: box.x1, y: box.y2 },
    { x: box.x2, y: box.y2 },
  ]
}

export function getResizeHandle(mousePos, box) {
  const boxW = box.x2 - box.x1
  const boxH = box.y2 - box.y1
  const minDim = Math.min(boxW, boxH)
  const tolerance = Math.max(18, Math.min(minDim * 0.4, 30))

  const corners = [
    { x: box.x1, y: box.y1, type: 'nw' },
    { x: box.x2, y: box.y1, type: 'ne' },
    { x: box.x1, y: box.y2, type: 'sw' },
    { x: box.x2, y: box.y2, type: 'se' },
  ]
  
  for (const corner of corners) {
    const dx = Math.abs(mousePos.x - corner.x)
    const dy = Math.abs(mousePos.y - corner.y)
    if (dx < tolerance && dy < tolerance) {
      return corner.type
    }
  }

  const midX = (box.x1 + box.x2) / 2
  const midY = (box.y1 + box.y2) / 2
  const edgeTolerance = tolerance * 0.8

  const edges = [
    { x: midX, y: box.y1, type: 'n' },
    { x: midX, y: box.y2, type: 's' },
    { x: box.x1, y: midY, type: 'w' },
    { x: box.x2, y: midY, type: 'e' },
  ]

  for (const edge of edges) {
    const dx = Math.abs(mousePos.x - edge.x)
    const dy = Math.abs(mousePos.y - edge.y)
    if (dx < edgeTolerance && dy < edgeTolerance) {
      return edge.type
    }
  }

  return null
}

export function getHandleCursor(handle) {
  const cursorMap = {
    nw: 'nw-resize', ne: 'ne-resize', sw: 'sw-resize', se: 'se-resize',
    n: 'n-resize', s: 's-resize', w: 'w-resize', e: 'e-resize',
  }
  return cursorMap[handle] || 'default'
}

export function isPointInBox(point, box) {
  return point.x >= box.x1 && point.x <= box.x2 &&
         point.y >= box.y1 && point.y <= box.y2
}

export function findBoxAtPoint(point, menuItems, imageInfo, scale, selectedBox = null) {
  if (selectedBox) {
    const selData = menuItems[selectedBox.itemName]?.checkboxes[selectedBox.checkboxIndex]
    if (selData) {
      const box = yoloToPixel(selData.bbox, imageInfo, scale)
      if (isPointInBox(point, box)) {
        return { itemName: selectedBox.itemName, checkboxIndex: selectedBox.checkboxIndex, box }
      }
    }
  }

  const entries = Object.entries(menuItems).reverse()
  for (const [itemName, itemData] of entries) {
    for (let i = itemData.checkboxes.length - 1; i >= 0; i--) {
      const checkbox = itemData.checkboxes[i]
      const box = yoloToPixel(checkbox.bbox, imageInfo, scale)
      if (isPointInBox(point, box)) {
        return { itemName, checkboxIndex: i, box }
      }
    }
  }
  return null
}

export function yoloToPixel(yoloBox, imageInfo, scale) {
  const [x_center_norm, y_center_norm, w_norm, h_norm] = yoloBox
  
  const centerX = x_center_norm * imageInfo.width
  const centerY = y_center_norm * imageInfo.height
  const width = w_norm * imageInfo.width
  const height = h_norm * imageInfo.height
  
  return {
    x1: (centerX - width / 2) * scale,
    y1: (centerY - height / 2) * scale,
    x2: (centerX + width / 2) * scale,
    y2: (centerY + height / 2) * scale
  }
}