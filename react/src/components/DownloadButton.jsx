import { exportToJson } from '../utils/jsonExporter'
import './Components.css'

function DownloadButton({ menuItems, imageInfo }) {
  const handleDownload = () => {
    exportToJson(menuItems, imageInfo)
  }

  return (
    <button className="download-button" onClick={handleDownload}>
      <span className="download-icon">💾</span>
      <span>Download JSON Mapping</span>
    </button>
  )
}

export default DownloadButton