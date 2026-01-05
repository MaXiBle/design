"""
Example usage of the Room to 2D Plan Converter
"""
import os
import json
import numpy as np
from python_neural_network import RoomTo2DConverter, ModelTrainer


def demo_basic_usage():
    """Demonstrate basic usage of the RoomTo2DConverter"""
    print("=== Room to 2D Plan Converter Demo ===\n")
    
    # Initialize the converter
    converter = RoomTo2DConverter()
    print("Initialized RoomTo2DConverter")
    
    # Since we don't have actual images, we'll create a dummy image for demonstration
    # In real usage, you would provide a path to an actual room photo
    print("\nNote: This demo shows the structure of the expected output")
    print("In practice, you would call converter.predict('path/to/room_photo.jpg')")
    
    # Show what a typical output would look like
    sample_output = {
        'room_type': 'living_room',
        'confidence': 0.85,
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
    
    print(f"Sample output structure:")
    print(json.dumps(sample_output, indent=2))


def demo_training():
    """Demonstrate the training process"""
    print("\n=== Training Demo ===\n")
    
    # Initialize the trainer
    trainer = ModelTrainer(model_save_path='./models/demo_model')
    print("Initialized ModelTrainer")
    
    # Create sample training data
    print("Creating sample training data...")
    from python_neural_network.model_trainer import create_sample_training_data
    create_sample_training_data('./sample_data')
    
    print("Sample data created in ./sample_data/")
    print("Directory structure:")
    print("- ./sample_data/images/ (contains sample images)")
    print("- ./sample_data/labels/ (contains corresponding labels)")
    
    print("\nNote: In a real scenario, you would call:")
    print("history = trainer.train_model('./sample_data', epochs=50, batch_size=32)")


def main():
    """Main function to run the demo"""
    demo_basic_usage()
    demo_training()
    
    print("\n=== Next Steps ===")
    print("1. Prepare your own room photos and corresponding 2D plan labels")
    print("2. Organize data in the required directory structure")
    print("3. Train the model using your data")
    print("4. Use the trained model to convert new room photos")
    
    print("\nFor detailed instructions, see the README.md file")


if __name__ == "__main__":
    main()