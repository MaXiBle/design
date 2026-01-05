# Integration Guide: Self-Learning Python Neural Network with JavaScript Frontend

This guide explains how to integrate the self-learning Python neural network module with your JavaScript frontend application.

## Overview

The self-learning Python neural network module is designed to work alongside your JavaScript application through a REST API. The architecture follows this pattern:

```
JavaScript Frontend ←→ Flask API ←→ Self-Learning Python Neural Network
```

The key feature of this system is that it learns from user feedback to improve its 2D plan generation. Users rate the generated plans from 0 (worst) to 5 (best), and the model uses this feedback to improve future predictions.

## Running the API Server

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python api_wrapper.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

Check if the API server is running and the model is loaded.

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Convert Room Photo
```
POST /convert
```

Convert a room photo to a 2D plan using the self-learning model.

#### Option 1: File Upload
- Form field: `image` (file)
- Example using curl:
```bash
curl -X POST -F "image=@path/to/room_photo.jpg" http://localhost:5000/convert
```

#### Option 2: Base64 Image Data
- Form field: `image_data` (base64 string)
- Example using curl:
```bash
curl -X POST -F "image_data=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..." http://localhost:5000/convert
```

### Convert Room Photo from URL
```
POST /convert_url
```

Convert a room photo from a URL to a 2D plan.

Content-Type: `application/json`

Request body:
```json
{
  "image_url": "https://example.com/room_photo.jpg"
}
```

### Submit User Feedback
```
POST /submit_feedback
```

Submit user rating (0-5) to improve the self-learning model.

Content-Type: `application/json`

Request body:
```json
{
  "plan_data": { /* full plan object returned from /convert */ },
  "user_score": 4
}
```

The user_score should be between 0 (worst) and 5 (best).

## JavaScript Integration

### 1. Update the RoomReconstructor Component

You can modify your existing `RoomReconstructor.jsx` to call the Python API instead of using the simulation:

```javascript
// In RoomReconstructor.jsx, replace the reconstructRoom function:
const reconstructRoom = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  try {
    const response = await fetch('http://localhost:5000/convert', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error calling Python API:', error);
    // Fallback to simulation if API is not available
    return neuralNetwork.reconstructRoom(imageFile);
  }
};
```

### 2. Alternative: Using Base64 Data

If you want to send base64-encoded image data:

```javascript
const reconstructRoomFromBase64 = async (imageBase64) => {
  const formData = new FormData();
  formData.append('image_data', imageBase64);

  try {
    const response = await fetch('http://localhost:5000/convert', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error calling Python API:', error);
    // Fallback to simulation if API is not available
    return neuralNetwork.reconstructRoom(imageBase64);
  }
};
```

### 3. Submitting User Feedback

To enable the self-learning functionality, you need to implement a feedback mechanism in your JavaScript application:

```javascript
// Function to submit user rating for the generated plan
const submitFeedback = async (planData, userScore) => {
  try {
    const response = await fetch('http://localhost:5000/submit_feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        plan_data: planData,
        user_score: userScore
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();
    console.log('Feedback submitted successfully:', result);
    return result;
  } catch (error) {
    console.error('Error submitting feedback:', error);
  }
};

// Example usage: after displaying the 2D plan to the user
const handleUserRating = async (planData, rating) => {
  // rating should be between 0 and 5
  await submitFeedback(planData, rating);
};
```

### 4. CORS Configuration

The Flask API includes CORS headers to allow requests from your frontend. If you're running the frontend on a different port, you may need to configure CORS properly.

## Configuration Options

### Model Initialization

Model initialization happens automatically when the API server starts:

```python
# In your application
api_server = Room2DAPIServer()
api_server.run()
```

### Custom API Server

You can create a custom server with additional configuration:

```python
from api_wrapper import Room2DAPIServer

# Create server instance
server = Room2DAPIServer()

# Access the Flask app for additional configuration if needed
app = server.app

# Add custom middleware or configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Run the server
server.run(host='0.0.0.0', port=5000, debug=False)
```

## Production Deployment Considerations

### 1. Model Loading

For production, pre-load your trained model:

```python
# Load the best trained model
api_server = Room2DAPIServer()
```

### 2. Security

- Use proper authentication if needed
- Validate file types and sizes
- Implement rate limiting
- Use HTTPS in production

### 3. Performance

- Consider using GPU acceleration for faster inference
- Implement caching for repeated requests
- Optimize model for inference speed if needed

### 4. Error Handling

The API includes basic error handling, but you may want to add more sophisticated logging and monitoring in production.

## Testing the Integration

### 1. Test the API Server

```bash
# Check if server is running
curl http://localhost:5000/health

# Test conversion (replace with your image)
curl -X POST -F "image=@sample_room.jpg" http://localhost:5000/convert
```

### 2. Verify Output Format

The API returns data in the same format as the JavaScript simulation, making it a drop-in replacement:

```json
{
  "room_type": "living_room",
  "confidence": 0.85,
  "walls": [...],
  "objects": [...],
  "dimensions": {"width": 800, "height": 600}
}
```

This format is compatible with the existing JavaScript rendering code in your RoomReconstructor component.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure the API server is running
2. **CORS Errors**: Ensure Flask-CORS is properly installed and configured
3. **Model Loading Issues**: Verify the model file path and format
4. **File Upload Issues**: Check file size limits and format support

### Debugging Tips

1. Check the API server logs for errors
2. Verify the neural network model is properly trained
3. Test API endpoints directly with curl or Postman
4. Use browser developer tools to inspect network requests