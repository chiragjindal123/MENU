export function drawBoundingBox(ctx, box, fillColor, strokeColor, lineWidth = 2) {
  const width = box.x2 - box.x1
  const height = box.y2 - box.y1

  // Fill
  ctx.fillStyle = fillColor
  ctx.fillRect(box.x1, box.y1, width, height)

  // Stroke
  ctx.strokeStyle = strokeColor
  ctx.lineWidth = lineWidth
  ctx.strokeRect(box.x1, box.y1, width, height)
}

export function drawExistingBoxes(ctx, menuItems, imageInfo, scale) {
  Object.entries(menuItems).forEach(([itemName, itemData]) => {
    itemData.checkboxes.forEach((checkbox, index) => {
      const box = yoloToPixel(checkbox.bbox, imageInfo, scale)
      
      // Draw box
      drawBoundingBox(ctx, box, 'rgba(0, 200, 0, 0.15)', '#00c800', 2)
      
      // Draw label
      const label = `${itemName}:${checkbox.option}`
      ctx.font = 'bold 14px Arial'
      
      // Measure text for background
      const textMetrics = ctx.measureText(label)
      const padding = 5
      const textHeight = 18
      
      // Background for text
      ctx.fillStyle = 'rgba(0, 200, 0, 0.9)'
      ctx.fillRect(
        box.x1, 
        box.y1 - textHeight - padding, 
        textMetrics.width + (padding * 2), 
        textHeight + padding
      )
      
      // Text
      ctx.fillStyle = 'white'
      ctx.fillText(label, box.x1 + padding, box.y1 - padding)
    })
  })
}

export function yoloToPixel(yoloBox, imageInfo, scale) {
  const [x_center_norm, y_center_norm, w_norm, h_norm] = yoloBox
  
  // Convert normalized YOLO to original pixel coordinates
  const centerX = x_center_norm * imageInfo.width
  const centerY = y_center_norm * imageInfo.height
  const width = w_norm * imageInfo.width
  const height = h_norm * imageInfo.height
  
  // Calculate corners
  const x1_orig = centerX - (width / 2)
  const y1_orig = centerY - (height / 2)
  const x2_orig = centerX + (width / 2)
  const y2_orig = centerY + (height / 2)
  
  // Scale to display coordinates
  return {
    x1: x1_orig * scale,
    y1: y1_orig * scale,
    x2: x2_orig * scale,
    y2: y2_orig * scale
  }
}