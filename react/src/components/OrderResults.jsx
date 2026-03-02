import './Components.css';

function OrderResults({ results }) {
  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(results.order, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'detected_order.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="order-results">
      <div className="results-header">
        <h2>✅ Order Detected!</h2>
        <button onClick={downloadJSON} className="btn-download-small">
          💾 Download JSON
        </button>
      </div>

      <div className="results-stats">
        <div className="stat-box">
          <span className="stat-number">{results.total_items}</span>
          <span className="stat-label">Items</span>
        </div>
        <div className="stat-box">
          <span className="stat-number">{results.total_quantity}</span>
          <span className="stat-label">Total Qty</span>
        </div>
        <div className="stat-box">
          <span className="stat-number">{results.unmatched_marks.length}</span>
          <span className="stat-label">Unmatched</span>
        </div>
      </div>

      <div className="order-items">
        <h3>📦 Ordered Items</h3>
        {Object.entries(results.order).map(([itemId, options]) => (
          <div key={itemId} className="order-item-card">
            <div className="item-header">
              <span className="item-id">Item #{itemId}</span>
            </div>
            <div className="item-options">
              {Object.entries(options).map(([option, data]) => (
                <div key={option} className="option-row">
                  <span className="option-name">{option.toUpperCase()}</span>
                  <span className="option-quantity">× {data.quantity}</span>
                  <div className="marks-info">
                    {data.marks.map((mark, idx) => (
                      <span key={idx} className="mark-badge">
                        {mark.type} ({(mark.confidence * 100).toFixed(1)}%)
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {results.unmatched_marks.length > 0 && (
        <div className="unmatched-section">
          <h3>⚠️ Unmatched Marks</h3>
          <div className="unmatched-list">
            {results.unmatched_marks.map((mark, idx) => (
              <span key={idx} className="unmatched-badge">
                {mark.mark} ({(mark.confidence * 100).toFixed(1)}%)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default OrderResults;