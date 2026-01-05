#!/usr/bin/env python3
"""
Demo script for the Room to 2D Plan Converter
This script demonstrates the usage of the Python neural network module
"""
import os
import sys
import json
from python_neural_network import RoomTo2DConverter, ModelTrainer


def main():
    print("Room to 2D Plan Converter - Demo")
    print("=" * 40)
    
    # Step 1: Initialize the converter
    print("\n1. Initializing Room to 2D Converter...")
    converter = RoomTo2DConverter()
    print("✓ Converter initialized successfully")
    
    # Step 2: Show the model architecture
    print("\n2. Model Architecture:")
    if converter.model:
        converter.model.summary()
    else:
        print("  Model not built yet")
    
    # Step 3: Create sample training data
    print("\n3. Creating sample training data...")
    from python_neural_network.model_trainer import create_sample_training_data
    sample_data_dir = './sample_data'
    create_sample_training_data(sample_data_dir)
    print(f"✓ Sample data created in {sample_data_dir}/")
    
    # Step 4: Initialize trainer
    print("\n4. Initializing Model Trainer...")
    trainer = ModelTrainer(model_save_path='./models/demo_room_converter')
    print("✓ Trainer initialized successfully")
    
    # Step 5: Brief training demonstration (with minimal epochs for demo)
    print("\n5. Starting training demonstration (minimal epochs for demo)...")
    try:
        history = trainer.train_model(
            data_dir=sample_data_dir,
            epochs=3,  # Minimal epochs for demo
            batch_size=4
        )
        print("✓ Training completed")
        
        # Save training history
        trainer.save_training_history('./models/demo_training_history.json')
        print("✓ Training history saved")
        
    except Exception as e:
        print(f"⚠ Training demo failed: {e}")
        print("  This is expected if TensorFlow is not properly configured")
    
    # Step 6: Demonstrate prediction with sample data
    print("\n6. Demonstrating prediction format...")
    
    # Show what a typical prediction would look like
    sample_prediction = {
        'room_type': 'living_room',
        'confidence': 0.87,
        'walls': [
            {
                'id': 1,
                'type': 'wall',
                'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 1.0, 'y2': 0.0},
                'color': '#d2b48c'
            },
            {
                'id': 2,
                'type': 'wall',
                'coordinates': {'x1': 1.0, 'y1': 0.0, 'x2': 1.0, 'y2': 1.0},
                'color': '#d2b48c'
            },
            {
                'id': 3,
                'type': 'wall',
                'coordinates': {'x1': 1.0, 'y1': 1.0, 'x2': 0.0, 'y2': 1.0},
                'color': '#d2b48c'
            },
            {
                'id': 4,
                'type': 'wall',
                'coordinates': {'x1': 0.0, 'y1': 1.0, 'x2': 0.0, 'y2': 0.0},
                'color': '#d2b48c'
            }
        ],
        'objects': [
            {
                'id': 1,
                'type': 'sofa',
                'coordinates': {'x': 0.2, 'y': 0.6, 'width': 0.4, 'height': 0.2}
            },
            {
                'id': 2,
                'type': 'coffee_table',
                'coordinates': {'x': 0.4, 'y': 0.7, 'width': 0.2, 'height': 0.15}
            },
            {
                'id': 3,
                'type': 'tv',
                'coordinates': {'x': 0.1, 'y': 0.1, 'width': 0.3, 'height': 0.1}
            }
        ],
        'dimensions': {'width': 800, 'height': 600}
    }
    
    print("Sample prediction output:")
    print(json.dumps(sample_prediction, indent=2))
    
    print("\n7. Starting API Server...")
    print("To start the full API server, run: python api_wrapper.py")
    print("The API provides endpoints for integrating with JavaScript frontend")
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Prepare your own room photos and labels")
    print("2. Train the model with your data")
    print("3. Use the trained model for predictions")
    print("4. Integrate with your JavaScript application using the API")


if __name__ == "__main__":
    main()