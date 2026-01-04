# Neural Network Integration Guide

## Overview
This document describes how to integrate a real neural network model into the interior design application using TensorFlow.js.

## Current Implementation
The current implementation uses a simulation approach that mimics neural network behavior. The system is designed to seamlessly transition to a real neural network model when available.

## Integration Steps

### 1. Model Preparation
- Train your neural network model using TensorFlow
- Convert the model to TensorFlow.js format using `tf.converters.save_keras_model`
- Host the model files (model.json and associated weights) on a web-accessible location

### 2. Update Model Loading
Modify the `loadModel()` method in `/src/utils/neuralNetwork.js` to load your actual model:

```javascript
async loadModel() {
  await this.initTensorFlow();
  
  console.log('Loading room reconstruction neural network model...');
  
  if (this.tensorflowLoaded) {
    try {
      // Load your real model
      this.model = await this.tf.loadLayersModel('path/to/your/model.json');
      this.isLoaded = true;
      console.log('Real neural network model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      console.log('Falling back to simulation mode');
      this.isLoaded = false;
    }
  } else {
    // Fallback to simulation mode
    // ... existing simulation code
  }
}
```

### 3. Image Processing Implementation
Replace the `processWithRealNeuralNetwork()` method to use actual TensorFlow.js operations:

```javascript
async processWithRealNeuralNetwork(imageData) {
  // Convert image to tensor
  const tensor = this.preprocessImage(imageData);
  
  // Run inference
  const prediction = await this.model.predict(tensor).data();
  
  // Process results and return in expected format
  return this.formatResults(prediction);
}

preprocessImage(imageData) {
  // Convert image data to tensor and normalize
  const imgTensor = this.tf.browser.fromPixels(imageData);
  const normalized = imgTensor.div(255.0);
  return normalized.expandDims(0); // Add batch dimension
}

formatResults(prediction) {
  // Convert neural network output to the expected format
  // This should match the structure expected by the UI components
  return {
    textures: [...],
    objects: [...],
    dimensions: {...},
    roomType: '...',
    confidence: 0.0
  };
}
```

### 4. Performance Considerations
- Neural network inference can be computationally expensive
- Consider running inference in a Web Worker to prevent UI blocking
- Implement progress indicators for long-running operations
- Cache model predictions when appropriate

### 5. Testing
- Test with various room images to ensure robustness
- Validate output format compatibility with existing UI components
- Monitor performance across different devices and browsers

## Architecture

### Components
- `RoomReconstructor.jsx`: UI component for image upload and result display
- `neuralNetwork.js`: Core neural network logic with simulation fallback
- `RoomReconstructor.css`: Styling for the reconstruction component

### Data Flow
1. User uploads an image
2. Image is processed by the neural network
3. Results are formatted for UI consumption
4. 2D room representation is rendered

## Error Handling
The system includes fallback mechanisms:
- If TensorFlow.js fails to load, simulation mode is used
- If model loading fails, simulation mode is used
- Network errors during inference trigger simulation fallback

## Production Deployment
For production use:
- Host model files on a CDN for faster loading
- Implement proper error logging
- Consider model quantization for faster inference
- Add model versioning for updates