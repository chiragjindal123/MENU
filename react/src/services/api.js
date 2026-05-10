const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const NGROK_HEADERS = {
  'ngrok-skip-browser-warning': 'true',
};

export class MenuAPI {
  /**
   * Detect order from filled menu image
   * @param {string} imageBase64 - Base64 encoded image
   * @param {object} mapping - Menu mapping JSON
   * @returns {Promise<object>} - Detection results
   */
  static async detectOrder(imageBase64, mapping) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/detect-order`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...NGROK_HEADERS,
        },
        body: JSON.stringify({
          image: imageBase64,
          mapping: mapping
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Detection failed');
      }

      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }

  /**
   * Check if backend is running
   */
  static async healthCheck() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(`${API_BASE_URL}/`, {
        headers: {
          ...NGROK_HEADERS,
        },
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response.ok;
    } catch (err) {
      console.error('Health check error:', err);
      return false;
    }
  }
}