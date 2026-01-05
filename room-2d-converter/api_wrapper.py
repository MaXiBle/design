"""
API Wrapper for Room to 2D Plan Converter
This module provides a Flask-based API to interface with the Python neural network module
and make it accessible to the JavaScript frontend.
"""
import os
import io
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import tempfile

from python_neural_network import RoomTo2DConverter


class Room2DAPIServer:
    """
    Flask API server that wraps the Room to 2D converter functionality
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the API server
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        # Initialize the converter
        self.converter = RoomTo2DConverter(model_path)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.converter.model is not None
            })
        
        @self.app.route('/convert', methods=['POST'])
        def convert_room():
            """Convert room photo to 2D plan"""
            try:
                # Check if image data is provided
                if 'image' not in request.files and 'image_data' not in request.form:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Handle image from file upload
                if 'image' in request.files:
                    file = request.files['image']
                    if file.filename == '':
                        return jsonify({'error': 'No image selected'}), 400
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        file.save(tmp.name)
                        image_path = tmp.name
                # Handle image from base64 data
                elif 'image_data' in request.form:
                    image_data = request.form['image_data']
                    # Remove data URL prefix if present
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name, 'JPEG')
                        image_path = tmp.name
                else:
                    return jsonify({'error': 'Invalid image format'}), 400
                
                # Process the image
                result = self.converter.predict(image_path)
                
                # Clean up temporary file
                os.unlink(image_path)
                
                # Ensure all values are JSON serializable
                result = self._make_serializable(result)
                
                return jsonify(result)
                
            except Exception as e:
                # Clean up temporary file if it exists
                if 'image_path' in locals():
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/convert_url', methods=['POST'])
        def convert_room_url():
            """Convert room photo from URL to 2D plan"""
            try:
                data = request.get_json()
                if not data or 'image_url' not in data:
                    return jsonify({'error': 'Image URL required'}), 400
                
                image_url = data['image_url']
                
                # Download image from URL (simplified - in production, use proper validation)
                import requests
                response = requests.get(image_url)
                if response.status_code != 200:
                    return jsonify({'error': 'Failed to download image'}), 400
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(response.content)
                    image_path = tmp.name
                
                # Process the image
                result = self.converter.predict(image_path)
                
                # Clean up temporary file
                os.unlink(image_path)
                
                # Ensure all values are JSON serializable
                result = self._make_serializable(result)
                
                return jsonify(result)
                
            except Exception as e:
                # Clean up temporary file if it exists
                if 'image_path' in locals():
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                
                return jsonify({'error': str(e)}), 500
    
    def _make_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable objects to JSON-serializable format
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the API server
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            debug: Enable debug mode
        """
        self.app.run(host=host, port=port, debug=debug)


def create_api_server(model_path=None):
    """
    Factory function to create an API server instance
    
    Args:
        model_path: Path to a pre-trained model (optional)
        
    Returns:
        Room2DAPIServer instance
    """
    return Room2DAPIServer(model_path)


if __name__ == "__main__":
    # Create and run the API server
    api_server = Room2DAPIServer()
    print("Starting Room to 2D Plan API server...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /convert - Convert room photo to 2D plan")
    print("  POST /convert_url - Convert room photo from URL to 2D plan")
    print("\nAPI server running on http://localhost:5000")
    
    api_server.run(host='0.0.0.0', port=5000, debug=True)