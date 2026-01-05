# Self-Learning Model Guide

This guide provides detailed instructions on how the self-learning neural network works for converting room photos to 2D floor plans. Unlike traditional models, this system learns from user feedback to continuously improve its performance.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Self-Learning Process](#self-learning-process)
3. [User Feedback Integration](#user-feedback-integration)
4. [Model Evaluation](#model-evaluation)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)

## Prerequisites

Before starting with the self-learning model, ensure you have:

1. **Pre-trained Models**: The system uses pre-trained models for object detection (Faster R-CNN) and wall detection (OpenCV)
2. **Computing Resources**: 
   - Recommended: GPU with at least 4GB VRAM for faster inference
   - Minimum: Modern CPU with 8GB+ RAM
3. **User Feedback**: A mechanism to collect user ratings (0-5 scale) for generated 2D plans

## Self-Learning Process

### How It Works

The self-learning model operates in the following way:

1. **Object Detection**: Uses pre-trained Faster R-CNN to detect furniture and objects
2. **Wall Detection**: Uses OpenCV algorithms to detect walls and room boundaries
3. **Plan Generation**: Combines detections to create a 2D plan
4. **Quality Assessment**: Evaluates the plan quality using a neural network
5. **User Feedback**: Collects user ratings (0-5) to improve future predictions
6. **Model Update**: Uses feedback to adjust the quality evaluation network

### Architecture Components

- **FurnitureDetector**: Pre-trained model for detecting furniture and objects
- **WallDetector**: OpenCV-based wall detection
- **PlanGenerator**: Combines detections into 2D plans
- **QualityEvaluator**: Neural network that assesses plan quality
- **RewardDatabase**: Stores user feedback for training

## User Feedback Integration

### Rating System

Users rate the generated 2D plans on a scale of 0 to 5:
- 0: Very poor quality (incorrect objects, wrong layout)
- 1: Poor quality (many errors)
- 2: Below average (some errors)
- 3: Average quality (minor errors)
- 4: Good quality (few minor issues)
- 5: Excellent quality (accurate and complete)

### Feedback Processing

The system processes feedback as follows:

```python
from python_neural_network.self_learning_room_converter import RoomConversionAPI

# Initialize the API
api = RoomConversionAPI()

# Generate a plan
plan_result = api.convert_room_to_plan('path/to/room_image.jpg')

if plan_result['success']:
    # Get the predicted quality (0-5 scale)
    predicted_quality = plan_result['plan']['predicted_quality']
    print(f"Predicted quality: {predicted_quality:.2f}")
    
    # User provides actual rating
    user_rating = 4  # User rates the plan as 4 out of 5
    
    # Submit feedback to improve the model
    feedback_result = api.submit_feedback(plan_result['plan'], user_rating)
    print(f"Model updated: {feedback_result}")
```

## Model Evaluation

### Quality Metrics

The model continuously evaluates its performance based on:

- **Prediction Accuracy**: How close the predicted quality is to user ratings
- **Learning Rate**: How quickly the model adapts to feedback
- **Consistency**: How consistently the model improves over time

### Monitoring Improvement

```python
import sqlite3

# Check the reward database to monitor learning progress
conn = sqlite3.connect('rewards.db')
cursor = conn.cursor()

# Get recent feedback
cursor.execute('SELECT user_score, timestamp FROM plan_evaluations ORDER BY timestamp DESC LIMIT 10')
recent_feedback = cursor.fetchall()
print("Recent user ratings:", recent_feedback)

conn.close()
```

## Performance Optimization

### 1. Inference Optimization

```python
# The model can be optimized for faster inference
from python_neural_network.self_learning_room_converter import RoomConversionAPI

# Initialize with optimized settings
api = RoomConversionAPI()

# Process multiple images efficiently
def batch_process_images(image_paths):
    results = []
    for path in image_paths:
        result = api.convert_room_to_plan(path)
        results.append(result)
    return results
```

### 2. Feedback Processing

To handle high volumes of feedback efficiently:

```python
# Process feedback in batches for better performance
def batch_feedback_processing(feedback_list):
    losses = []
    for plan_data, user_score in feedback_list:
        loss = api.model.update_with_feedback(plan_data, user_score)
        losses.append(loss)
    return losses
```

## Best Practices

### 1. Collecting Quality Feedback

- **Clear Instructions**: Provide clear guidelines to users on how to rate plans
- **Diverse Feedback**: Encourage users to rate various types of rooms and scenarios
- **Consistent Criteria**: Maintain consistent rating criteria across users

### 2. Model Management

- **Regular Updates**: Update the model regularly with new feedback
- **Performance Monitoring**: Monitor the model's prediction accuracy over time
- **Data Quality**: Ensure the feedback data is clean and meaningful

### 3. User Experience

- **Immediate Response**: Provide quick results to maintain user engagement
- **Quality Indicators**: Show predicted quality scores to set expectations
- **Feedback Validation**: Validate user feedback to prevent malicious inputs

### 4. Database Management

- **Regular Cleanup**: Periodically archive old feedback data
- **Backup Strategy**: Maintain backups of the reward database
- **Performance Monitoring**: Monitor database performance as it grows

### 5. Monitoring and Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./models/self_learning.log'),
        logging.StreamHandler()
    ]
)

# Log model updates
logging.info("Model updated with user feedback, loss: 0.1234")
```

## Common Issues and Solutions

### 1. Slow Learning

**Symptoms**: Model quality doesn't improve significantly over time

**Solutions**:
- Collect more diverse feedback
- Increase learning rate
- Verify feedback quality

### 2. Overfitting to Feedback

**Symptoms**: Model performs well on recent feedback but degrades on other data

**Solutions**:
- Implement regularization in the quality evaluator
- Use a validation set to monitor generalization
- Reduce learning rate

### 3. Feedback Quality Issues

**Symptoms**: Inconsistent or contradictory feedback

**Solutions**:
- Implement feedback validation
- Use confidence thresholds
- Analyze feedback patterns

## Expected Performance

- **Initial Performance**: Depends on pre-trained models (typically good for common objects)
- **Improvement Rate**: Noticeable improvement after 50-100 feedback entries
- **Stabilization**: Performance typically stabilizes after 500+ feedback entries

Performance depends on feedback quality and diversity.

## Model Deployment

The self-learning model is ready to use immediately with pre-trained components:

```python
from python_neural_network.self_learning_room_converter import RoomConversionAPI

# Initialize the self-learning API
api = RoomConversionAPI()

# Ready to process images and learn from feedback
```

The model continues learning as users provide feedback through the API.