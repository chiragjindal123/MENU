import { useRef } from 'react'
// import 'App.css'
import './Components.css'

function ImageUploader({ onImageUpload }) {
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      
      reader.onload = (event) => {
        const img = new Image()
        img.onload = () => {
          onImageUpload(event.target.result, {
            width: img.width,
            height: img.height,
            name: file.name
          })
        }
        img.src = event.target.result
      }
      
      reader.readAsDataURL(file)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      const fakeEvent = { target: { files: [file] } }
      handleFileChange(fakeEvent)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  return (
    <div 
      className="image-uploader"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => fileInputRef.current?.click()}
    >
      <div className="upload-icon">📤</div>
      <h3>Upload Menu Image</h3>
      <p>Click or drag & drop your blank menu image here</p>
      <p className="file-types">Supported: JPG, PNG</p>
      
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
    </div>
  )
}

export default ImageUploader