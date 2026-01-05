#!/usr/bin/env python3
"""
Demo script for the Self-Learning Room to 2D Plan Converter
This script demonstrates the usage of the self-learning Python neural network module
"""
import os
import sys
import json
from python_neural_network.self_learning_room_converter import RoomConversionAPI


def main():
    print("Self-Learning Room to 2D Plan Converter - Demo")
    print("=" * 50)
    
    # Step 1: Initialize the self-learning converter
    print("\n1. Initializing Self-Learning Room to 2D Converter...")
    api = RoomConversionAPI()
    print("✓ Self-learning converter initialized successfully")
    
    # Step 2: Show the model architecture concept
    print("\n2. Model Architecture:")
    print("  - Uses pre-trained models for object detection (Faster R-CNN)")
    print("  - Wall detection using OpenCV")
    print("  - Quality evaluation network that learns from user feedback")
    print("  - Feedback system using SQLite database")
    
    # Step 3: Demonstrate processing
    print("\n3. Demonstrating room to 2D plan conversion...")
    
    # Show what a typical prediction would look like
    sample_prediction = {
        'image_shape': [480, 640, 3],
        'furniture': [
            {
                'class': 'couch',
                'confidence': 0.92,
                'bbox': [100, 200, 300, 350]  # [x1, y1, x2, y2]
            },
            {
                'class': 'dining table',
                'confidence': 0.87,
                'bbox': [250, 300, 400, 400]
            }
        ],
        'walls': [
            {
                'type': 'wall',
                'coordinates': [50, 50, 590, 50],
                'length': 540.0
            },
            {
                'type': 'wall',
                'coordinates': [590, 50, 590, 430],
                'length': 380.0
            },
            {
                'type': 'room_boundary',
                'coordinates': [[50, 50], [590, 50], [590, 430], [50, 430]]
            }
        ],
        'timestamp': '2023-12-01T10:00:00',
        'predicted_quality': 3.8
    }
    
    print("Sample prediction output:")
    print(json.dumps(sample_prediction, indent=2))
    
    print("\n4. Self-Learning Process:")
    print("  - Model generates 2D plan from room photo")
    print("  - You rate the result from 0 (worst) to 5 (best)")
    print("  - Model learns from your feedback to improve future predictions")
    print("  - Ratings stored in SQLite database for analysis")
    
    print("\n5. Starting API Server...")
    print("To start the full API server, run: python api_wrapper.py")
    print("The API provides endpoints for integrating with JavaScript frontend")
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Provide room photos for conversion")
    print("2. Rate the generated 2D plans (0-5 scale)")
    print("3. Model will improve based on your feedback")
    print("4. Integrate with your JavaScript application using the API")


if __name__ == "__main__":
    main()