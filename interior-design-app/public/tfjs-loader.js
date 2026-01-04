// TensorFlow.js loader script
// This script handles the loading of TensorFlow.js in the browser

// Function to dynamically load TensorFlow.js
async function loadTensorFlowJS() {
  if (window.tf) {
    console.log('TensorFlow.js already loaded');
    return window.tf;
  }

  console.log('Loading TensorFlow.js from CDN...');
  
  return new Promise((resolve, reject) => {
    // Create script element for TensorFlow.js
    const tfScript = document.createElement('script');
    tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
    tfScript.async = true;
    
    tfScript.onload = async () => {
      console.log('TensorFlow.js loaded successfully');
      
      // Also load the TensorFlow.js vision model if needed
      try {
        // Load TensorFlow.js image recognition models if needed
        const visionScript = document.createElement('script');
        visionScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest/dist/tf-vis.umd.min.js';
        document.head.appendChild(visionScript);
      } catch (e) {
        console.log('Could not load tf-vis, continuing with basic tf...');
      }
      
      resolve(window.tf);
    };
    
    tfScript.onerror = (error) => {
      console.error('Failed to load TensorFlow.js:', error);
      reject(error);
    };
    
    document.head.appendChild(tfScript);
  });
}

// Initialize TensorFlow.js when the page loads
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadTensorFlowJS();
    console.log('Neural network system ready');
  } catch (error) {
    console.warn('Could not load TensorFlow.js, running in simulation mode:', error);
  }
});

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { loadTensorFlowJS };
}