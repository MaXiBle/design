# Model Training Guide

This guide provides detailed instructions on how to train the neural network for converting room photos to 2D floor plans.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Training Process](#training-process)
4. [Model Evaluation](#model-evaluation)
5. [Model Optimization](#model-optimization)
6. [Best Practices](#best-practices)

## Prerequisites

Before starting the training process, ensure you have:

1. **Sufficient Data**: At least 1000+ labeled room photos for good results
2. **Computing Resources**: 
   - Recommended: GPU with at least 8GB VRAM
   - Minimum: Modern CPU with 16GB+ RAM
3. **Data Quality**: High-resolution images with clear room views
4. **Label Accuracy**: Precise annotations for walls, objects, and room types

## Data Preparation

### Directory Structure

Organize your training data in the following structure:

```
dataset/
├── images/
│   ├── room_001.jpg
│   ├── room_002.jpg
│   └── ...
└── labels/
    ├── room_001.json
    ├── room_002.json
    └── ...
```

### Label Format

Each JSON label file should follow this structure:

```json
{
  "room_type": "living_room",
  "confidence": 0.9,
  "walls": [
    {
      "id": 1,
      "type": "wall",
      "coordinates": {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 0.0
      }
    }
  ],
  "objects": [
    {
      "id": 1,
      "type": "sofa",
      "coordinates": {
        "x": 0.2,
        "y": 0.7,
        "width": 0.4,
        "height": 0.2
      }
    }
  ],
  "dimensions": {
    "width": 800,
    "height": 600
  }
}
```

### Room Types

Supported room types:
- `living_room` - Living rooms and family rooms
- `bedroom` - Bedrooms of any kind
- `kitchen` - Kitchens and kitchenettes
- `bathroom` - Bathrooms and powder rooms

### Coordinate System

Coordinates are normalized to the range [0, 1]:
- (0, 0) is the top-left corner
- (1, 1) is the bottom-right corner
- All measurements are relative to image dimensions

## Training Process

### 1. Data Augmentation

To improve model generalization, consider applying data augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

### 2. Training Script

```python
from python_neural_network import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(model_save_path='./models/room_converter')

# Train the model
history = trainer.train_model(
    data_dir='./dataset',
    epochs=100,
    batch_size=16,  # Reduce if you have limited GPU memory
    validation_split=0.2
)

# Save training history
trainer.save_training_history('./models/training_history.json')
```

### 3. Monitoring Training

Monitor the training process using the returned history object:

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['room_type_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_room_type_accuracy'], label='Validation Accuracy')
plt.title('Room Type Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('./models/training_progress.png')
```

## Model Evaluation

### 1. Validation Metrics

The model tracks several metrics during training:

- **Loss**: Combined loss for all tasks
- **Room Type Accuracy**: Classification accuracy for room types
- **Wall MAE**: Mean absolute error for wall coordinate predictions
- **Object MAE**: Mean absolute error for object coordinate predictions
- **Confidence MAE**: Error in confidence score prediction

### 2. Custom Evaluation

```python
# Evaluate on test set
evaluation_results = trainer.evaluate_model(X_test, y_test)
print("Evaluation Results:", evaluation_results)
```

### 3. Visualization

Create visualizations to understand model performance:

```python
def visualize_predictions(converter, test_images, test_labels):
    """Visualize model predictions vs ground truth"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i in range(min(4, len(test_images))):
        row = i // 2
        col = i % 2
        
        # Make prediction
        pred = converter.predict(test_images[i])
        
        # Display results
        axes[row, col].imshow(test_images[i])
        axes[row, col].set_title(f"Predicted: {pred['room_type']}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('./models/predictions_visualization.png')
```

## Model Optimization

### 1. Hyperparameter Tuning

Experiment with different hyperparameters:

```python
# Learning rate scheduling
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Training with scheduler
history = trainer.train_model(
    data_dir='./dataset',
    epochs=100,
    batch_size=16,
    callbacks=[lr_scheduler]
)
```

### 2. Model Architecture Adjustments

For better performance, you can modify the architecture:

```python
def create_advanced_model():
    """Create a more sophisticated model architecture"""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    
    # Use pre-trained ResNet as feature extractor
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Multi-task heads
    room_type = Dense(128, activation='relu')(x)
    room_type = Dense(4, activation='softmax', name='room_type')(room_type)
    
    # Similar architecture for other tasks...
    
    model = Model(inputs, [room_type, ...])
    return model
```

### 3. Transfer Learning

Start with a pre-trained model and fine-tune:

```python
# Load a pre-trained model
converter = RoomTo2DConverter(model_path='./models/pretrained_model.h5')

# Fine-tune on your specific dataset
converter.model.trainable = True

# Use lower learning rate for fine-tuning
converter.model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    # ... other parameters
)
```

## Best Practices

### 1. Data Quality

- **Diverse Data**: Include various room layouts, lighting conditions, and camera angles
- **Consistent Labeling**: Maintain consistent annotation standards
- **Quality Control**: Remove blurry, overexposed, or poor-quality images

### 2. Training Strategies

- **Progressive Training**: Start with a simpler model and gradually increase complexity
- **Validation Split**: Always maintain a held-out validation set
- **Early Stopping**: Prevent overfitting with early stopping

### 3. Model Validation

- **Cross-Validation**: Use k-fold cross-validation for robust evaluation
- **Real-World Testing**: Test on images from different sources than training data
- **Error Analysis**: Analyze failure cases to improve the model

### 4. Performance Optimization

- **Batch Size**: Use the largest batch size that fits in memory
- **Mixed Precision**: Use mixed precision training to speed up training
- **Distributed Training**: For large datasets, consider multi-GPU training

### 5. Monitoring and Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./models/training.log'),
        logging.StreamHandler()
    ]
)
```

## Common Issues and Solutions

### 1. Overfitting

**Symptoms**: Training loss decreases but validation loss increases

**Solutions**:
- Add more dropout
- Use data augmentation
- Reduce model complexity
- Implement early stopping

### 2. Underfitting

**Symptoms**: Both training and validation loss are high

**Solutions**:
- Increase model complexity
- Train for more epochs
- Use a different architecture
- Check data quality

### 3. Slow Training

**Solutions**:
- Use GPU acceleration
- Optimize batch size
- Use mixed precision training
- Profile and optimize data pipeline

## Expected Training Time

- **Small dataset** (100-500 images): 1-4 hours
- **Medium dataset** (500-2000 images): 4-12 hours  
- **Large dataset** (2000+ images): 12+ hours

Training time depends on your hardware and dataset size.

## Model Deployment

Once trained, save your model:

```python
converter.save_model('./models/final_room_converter.h5')
```

The saved model can be loaded later for inference or further training.