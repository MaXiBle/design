# Room to 2D Plan Converter

A Python neural network module that converts room photos to interactive 2D floor plans. This module uses deep learning techniques to analyze room images and generate corresponding 2D representations with walls, furniture, and room type classification.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [API Integration](#api-integration)
- [Model Architecture](#model-architecture)
- [Training Tips](#training-tips)
- [License](#license)

## Features

- Convert room photos to 2D floor plans
- Detect room type (living room, bedroom, kitchen, bathroom)
- Identify walls and architectural elements
- Recognize and place furniture appropriately
- Confidence scoring for predictions
- Trainable neural network architecture

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from python_neural_network import RoomTo2DConverter

# Initialize the converter
converter = RoomTo2DConverter()

# Convert a room photo to 2D plan
result = converter.predict('path/to/room_photo.jpg')

# Print the result
print(f"Room type: {result['room_type']}")
print(f"Confidence: {result['confidence']}")
print(f"Walls: {result['walls']}")
print(f"Objects: {result['objects']}")
```

### Using a Pre-trained Model

```python
from python_neural_network import RoomTo2DConverter

# Load a pre-trained model
converter = RoomTo2DConverter(model_path='path/to/your/model.h5')

# Convert a room photo
result = converter.predict('path/to/room_photo.jpg')
```

## Training

For detailed training instructions, see the [Model Training Guide](MODEL_TRAINING_GUIDE.md).

### Data Preparation

The training data should be organized in the following structure:

```
training_data/
├── images/
│   ├── room1.jpg
│   ├── room2.jpg
│   └── ...
└── labels/
    ├── room1.json
    ├── room2.json
    └── ...
```

Each label file should contain the following information in JSON format:

```json
{
  "room_type": "living_room",
  "confidence": 0.85,
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
        "x": 0.3,
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

### Training Process

```python
from python_neural_network import ModelTrainer

# Initialize the trainer
trainer = ModelTrainer(model_save_path='./models/room_converter')

# Train the model
history = trainer.train_model(
    data_dir='./training_data',
    epochs=100,
    batch_size=32
)

# Save the training history
trainer.save_training_history('./models/training_history.json')
```

### Sample Training Data

For demonstration purposes, you can generate sample training data:

```python
from python_neural_network.model_trainer import create_sample_training_data

# Create sample training data
create_sample_training_data('./sample_data')
```

## API Integration

The module includes a Flask-based API wrapper for easy integration with web applications. For detailed integration instructions, see the [Integration Guide](INTEGRATION_GUIDE.md).

Start the API server:
```bash
python api_wrapper.py
```

The API provides endpoints for:
- Health check: `GET /health`
- Image conversion: `POST /convert`
- URL-based conversion: `POST /convert_url`

## Model Architecture

The neural network uses a multi-task learning approach with:

1. **Feature Extraction Backbone**: CNN layers for extracting visual features
2. **Room Type Classification**: Identifies the type of room in the image
3. **Wall Detection**: Locates and segments walls in the room
4. **Object Detection**: Identifies and positions furniture and other objects
5. **Confidence Scoring**: Estimates the reliability of predictions

## Model Output Format

The model returns a dictionary with the following structure:

```python
{
    'room_type': 'living_room',           # Predicted room type
    'confidence': 0.85,                   # Confidence score (0-1)
    'walls': [                            # List of wall objects
        {
            'id': 1,
            'type': 'wall',
            'coordinates': {
                'x1': 0.0, 'y1': 0.0,    # Start coordinates (normalized)
                'x2': 1.0, 'y2': 0.0     # End coordinates (normalized)
            },
            'color': '#d2b48c'
        }
    ],
    'objects': [                          # List of detected objects
        {
            'id': 1,
            'type': 'sofa',               # Object type
            'coordinates': {
                'x': 0.3,                # X position (normalized)
                'y': 0.7,                # Y position (normalized)
                'width': 0.4,            # Width (normalized)
                'height': 0.2            # Height (normalized)
            }
        }
    ],
    'dimensions': {                       # Canvas dimensions
        'width': 800,
        'height': 600
    }
}
```

The output format is designed to be compatible with JavaScript frontends. The normalized coordinates can be easily converted to pixel values based on the display dimensions.

## Training Tips

For comprehensive training guidance, refer to the [Model Training Guide](MODEL_TRAINING_GUIDE.md). Key tips include:

1. **Data Quality**: Use high-quality images with clear views of rooms
2. **Data Diversity**: Include various lighting conditions, angles, and room layouts
3. **Label Accuracy**: Ensure precise labeling of walls and objects
4. **Balanced Dataset**: Include equal representation of different room types
5. **Augmentation**: Consider using data augmentation to increase dataset size

## License

This project is licensed under the MIT License - see the LICENSE file for details.