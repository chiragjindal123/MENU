export function exportToJson(menuItems, imageInfo) {
  const mappingData = {
    image_info: {
      width: imageInfo.width,
      height: imageInfo.height,
      source: imageInfo.name
    },
    menu_mapping: menuItems
  }

  const jsonString = JSON.stringify(mappingData, null, 2)
  const blob = new Blob([jsonString], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  
  const link = document.createElement('a')
  link.href = url
  link.download = 'menu_mapping.json'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  URL.revokeObjectURL(url)
  
  const totalCheckboxes = Object.values(menuItems).reduce((sum, item) => sum + item.checkboxes.length, 0)
  
  console.log('✅ JSON mapping exported successfully!')
  console.log('📊 Total items:', Object.keys(menuItems).length)
  console.log('📊 Total checkboxes:', totalCheckboxes)
  console.log('📋 Mapping preview:', mappingData)
}