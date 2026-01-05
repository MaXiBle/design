# JavaScript-Python Integration Guide for Self-Learning Room to 2D Plan Converter

This guide explains how to integrate the self-learning Python neural network module with your JavaScript frontend application.

## Overview

The self-learning system uses a client-server architecture:

```
[JavaScript Frontend] ←→ [Flask API Server] ←→ [Self-Learning Python Model]
```

The Python model automatically learns from user feedback to improve its 2D plan generation capabilities.

## Setup Instructions

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

### Convert Room Photo to 2D Plan
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

### 1. Basic Integration

Update your JavaScript component to use the Python API:

```javascript
// Function to convert room photo to 2D plan
const convertRoomToPlan = async (imageFile) => {
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
    throw error;
  }
};

// Example usage
const handleImageUpload = async (event) => {
  const file = event.target.files[0];
  if (file) {
    try {
      const planResult = await convertRoomToPlan(file);
      if (planResult.success) {
        console.log('Generated plan:', planResult.plan);
        // Display the plan to the user
        displayPlan(planResult.plan);
      } else {
        console.error('Conversion failed:', planResult.error);
      }
    } catch (error) {
      console.error('Error converting room:', error);
    }
  }
};
```

### 2. With User Feedback Integration

To enable the self-learning functionality, implement user feedback:

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
    throw error;
  }
};

// Example: After displaying the plan, allow user to rate it
const displayPlanWithRating = async (planData) => {
  // Display the plan to the user
  renderPlan(planData);
  
  // Get predicted quality score
  const predictedQuality = planData.predicted_quality;
  console.log(`Model predicted quality: ${predictedQuality}/5.0`);
  
  // After user interaction, collect rating
  const userRating = await getUserRating(); // This would be your UI function
  
  // Submit feedback to improve the model
  await submitFeedback(planData, userRating);
};
```

### 3. Complete Component Example

Here's a complete example of a React component that integrates with the self-learning model:

```jsx
import React, { useState } from 'react';

const RoomTo2DConverter = () => {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [rating, setRating] = useState(0);
  const [submittedFeedback, setSubmittedFeedback] = useState(false);

  const convertImage = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://localhost:5000/convert', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      if (result.success) {
        setPlan(result.plan);
        console.log('Generated plan with predicted quality:', result.plan.predicted_quality);
      } else {
        console.error('Conversion failed:', result.error);
      }
    } catch (error) {
      console.error('Error converting room:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      convertImage(file);
      setSubmittedFeedback(false);
    }
  };

  const submitRating = async () => {
    if (!plan || rating < 0 || rating > 5) return;

    try {
      const response = await fetch('http://localhost:5000/submit_feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          plan_data: plan,
          user_score: rating
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      console.log('Feedback submitted:', result);
      setSubmittedFeedback(true);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div className="room-converter">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        disabled={loading}
      />
      
      {loading && <p>Processing image...</p>}
      
      {plan && (
        <div className="plan-result">
          <h3>Generated 2D Plan</h3>
          <p>Predicted Quality: {plan.predicted_quality.toFixed(2)}/5.0</p>
          <p>Furniture detected: {plan.furniture.length}</p>
          <p>Walls detected: {plan.walls.length}</p>
          
          {!submittedFeedback && (
            <div className="rating-section">
              <h4>Rate this plan (0-5):</h4>
              <div className="rating-input">
                {[0, 1, 2, 3, 4, 5].map((value) => (
                  <button
                    key={value}
                    onClick={() => setRating(value)}
                    className={rating === value ? 'active' : ''}
                  >
                    {value}
                  </button>
                ))}
              </div>
              <button onClick={submitRating} disabled={rating < 0 || rating > 5}>
                Submit Rating
              </button>
            </div>
          )}
          
          {submittedFeedback && <p>Thank you for your feedback!</p>}
        </div>
      )}
    </div>
  );
};

export default RoomTo2DConverter;
```

## Python Model Architecture

### Self-Learning Components

1. **FurnitureDetector**: Uses pre-trained Faster R-CNN to detect furniture and objects
2. **WallDetector**: Uses OpenCV algorithms to detect walls and room boundaries
3. **PlanGenerator**: Combines detections to create a 2D plan
4. **QualityEvaluator**: Neural network that assesses plan quality
5. **RewardDatabase**: Stores user feedback for training

### Learning Process

1. User uploads room photo
2. Model generates 2D plan and predicts quality score (0-5)
3. User rates the plan (0-5)
4. Model updates quality evaluation network based on feedback
5. Process repeats, model gradually improves

## Error Handling

### Common Issues

1. **Connection Refused**: Ensure API server is running
2. **CORS Errors**: Verify Flask-CORS is properly configured
3. **File Upload Issues**: Check file size limits and format support
4. **Model Loading Issues**: Verify dependencies are installed

### JavaScript Error Handling

```javascript
const safeApiCall = async (url, options) => {
  try {
    const response = await fetch(url, options);
    
    // Check if response is JSON
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }
      return data;
    } else {
      throw new Error('Response is not JSON');
    }
  } catch (error) {
    console.error('API call failed:', error);
    return { success: false, error: error.message };
  }
};
```

## Performance Considerations

### Client-Side
- Implement loading states during image processing
- Cache results for repeated requests
- Validate image format and size before upload

### Server-Side
- Monitor model inference time
- Implement request rate limiting
- Use GPU acceleration if available
- Monitor database performance as feedback grows

## Security Considerations

- Validate and sanitize uploaded images
- Implement proper authentication if needed
- Set appropriate file size limits
- Use HTTPS in production
- Validate user feedback values (0-5 range)