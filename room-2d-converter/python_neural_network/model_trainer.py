"""
Модуль тренировки модели для конвертера фото комнат в 2D план
Этот модуль обрабатывает процесс обучения нейронной сети.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
import logging
from typing import Tuple, Dict, Any, Optional

from .room_converter import RoomTo2DConverter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения модели преобразования фото комнат в 2D план.
    Обрабатывает подготовку данных, обучение модели и оценку.
    """
    
    def __init__(self, model_save_path: str = "./models/room_converter_model"):
        """
        Инициализирует тренер модели.
        
        Args:
            model_save_path: Путь, по которому будут сохраняться обученные модели
        """
        self.model_save_path = model_save_path
        self.converter = RoomTo2DConverter()
        self.history = None
        
        # Create directory for saving models if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def prepare_data(self, data_dir: str) -> Tuple:
        """
        Подготавливает обучающие данные из структуры директории.
        
        Args:
            data_dir: Директория, содержащая обучающие данные
                    Ожидаемая структура:
                    - data_dir/images/ (содержит фото комнат)
                    - data_dir/labels/ (содержит соответствующие метки в формате JSON)
                    
        Returns:
            Кортеж (X_train, X_val, y_train, y_val) данных
        """
        logger.info(f"Preparing data from {data_dir}")
        
        # This is a simplified implementation
        # In a real scenario, you would load and process your actual data
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise ValueError(f"Data directories do not exist: {images_dir}, {labels_dir}")
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()  # Sort to maintain correspondence with labels
        
        # Load images and labels
        images = []
        labels = []
        
        for img_file in image_files:
            # Load and preprocess image
            img_path = os.path.join(images_dir, img_file)
            img = self._load_and_preprocess_image(img_path)
            images.append(img)
            
            # Load corresponding label
            label_file = os.path.splitext(img_file)[0] + '.json'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                labels.append(self._format_label(label_data))
            else:
                # If no label exists, create a dummy label (for demonstration)
                labels.append(self._create_dummy_label())
        
        # Convert to numpy arrays
        X = np.array(images)
        y = self._stack_labels(labels)
        
        # Split into train and validation sets
        X_train, X_val, y_train_dict, y_val_dict = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        return X_train, X_val, y_train_dict, y_val_dict
    
    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Загружает и предобрабатывает одно изображение.
        
        Args:
            img_path: Путь к файлу изображения
            
        Returns:
            Предобработанное изображение в виде массива numpy
        """
        # Load image
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array
    
    def _format_label(self, label_data: Dict) -> Dict[str, np.ndarray]:
        """
        Форматирует данные метки в соответствии с выходами модели.
        
        Args:
            label_data: Необработанные данные метки из JSON
            
        Returns:
            Отформатированный словарь меток, соответствующий выходам модели
        """
        # Room type (one-hot encoded)
        room_types = ['living_room', 'bedroom', 'kitchen', 'bathroom']
        room_type_idx = room_types.index(label_data.get('room_type', 'living_room'))
        room_type_onehot = np.zeros(len(room_types))
        room_type_onehot[room_type_idx] = 1
        
        # Wall coordinates (flattened)
        walls = np.zeros(8)  # 4 walls with 2 coordinates each (x1, y1, x2, y2)
        for i, wall in enumerate(label_data.get('walls', [])):
            if i < 4:
                walls[i*2] = wall['coordinates']['x1']
                walls[i*2 + 1] = wall['coordinates']['y1']
        
        # Object coordinates (flattened)
        objects = np.zeros(20)  # 5 objects with 4 coordinates each (x, y, width, height)
        for i, obj in enumerate(label_data.get('objects', [])):
            if i < 5:
                start_idx = i * 4
                objects[start_idx] = obj['coordinates']['x']
                objects[start_idx + 1] = obj['coordinates']['y']
                objects[start_idx + 2] = obj['coordinates']['width']
                objects[start_idx + 3] = obj['coordinates']['height']
        
        # Confidence score
        confidence = np.array([label_data.get('confidence', 0.8)])
        
        return {
            'room_type': room_type_onehot,
            'walls': walls,
            'objects': objects,
            'confidence': confidence
        }
    
    def _create_dummy_label(self) -> Dict[str, np.ndarray]:
        """
        Создает фиктивную метку для целей демонстрации.
        
        Returns:
            Словарь фиктивной метки
        """
        room_types = np.zeros(4)
        room_types[0] = 1  # Default to living room
        
        walls = np.random.rand(8)  # Random wall coordinates
        objects = np.random.rand(20)  # Random object coordinates
        confidence = np.array([0.7])
        
        return {
            'room_type': room_types,
            'walls': walls,
            'objects': objects,
            'confidence': confidence
        }
    
    def _stack_labels(self, labels: list) -> Dict[str, np.ndarray]:
        """
        Объединяет несколько словарей меток в один словарь с массивами.
        
        Args:
            labels: Список отдельных словарей меток
            
        Returns:
            Словарь с объединенными массивами меток
        """
        if not labels:
            return {}
        
        result = {}
        for key in labels[0].keys():
            result[key] = np.array([label[key] for label in labels])
        
        return result
    
    def train_model(self, 
                   data_dir: str, 
                   epochs: int = 50, 
                   batch_size: int = 32,
                   validation_split: float = 0.2) -> keras.callbacks.History:
        """
        Обучает модель на предоставленных данных.
        
        Args:
            data_dir: Директория, содержащая обучающие данные
            epochs: Количество эпох обучения
            batch_size: Размер пакета для обучения
            validation_split: Доля данных для использования в валидации
            
        Returns:
            Объект истории обучения
        """
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(data_dir)
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=7, 
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"{self.model_save_path}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        self.history = self.converter.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        final_model_path = f"{self.model_save_path}_final.h5"
        self.converter.save_model(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: Dict) -> Dict[str, float]:
        """
        Оценивает обученную модель на тестовых данных.
        
        Args:
            X_test: Тестовые изображения
            y_test: Тестовые метки
            
        Returns:
            Словарь с метриками оценки
        """
        if self.converter.model is None:
            raise ValueError("Model not trained yet")
        
        # Evaluate the model
        results = self.converter.model.evaluate(X_test, y_test, verbose=0)
        
        # Get metric names
        metric_names = self.converter.model.metrics_names
        evaluation_results = dict(zip(metric_names, results))
        
        logger.info("Model evaluation completed")
        for metric, value in evaluation_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return evaluation_results
    
    def get_training_history(self) -> Optional[keras.callbacks.History]:
        """
        Получает историю обучения.
        
        Returns:
            Объект истории обучения или None, если модель еще не обучалась
        """
        return self.history
    
    def save_training_history(self, filepath: str):
        """
        Сохраняет историю обучения в файл JSON.
        
        Args:
            filepath: Путь, по которому нужно сохранить историю обучения
        """
        if self.history is None:
            logger.warning("No training history to save")
            return
        
        history_dict = {key: [float(val) for val in values] for key, values in self.history.history.items()}
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {filepath}")


def create_sample_training_data(data_dir: str):
    """
    Создает пример обучающих данных для целей демонстрации.
    Эта функция создает небольшой набор данных для демонстрации процесса обучения.
    
    Args:
        data_dir: Директория, в которой будут созданы примеры данных
    """
    import cv2
    import random
    
    # Create directory structure
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'labels'), exist_ok=True)
    
    room_types = ['living_room', 'bedroom', 'kitchen', 'bathroom']
    
    # Generate sample images and labels
    for i in range(20):  # Create 20 sample data points
        # Create a synthetic room image (in practice, you'd use real images)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Save the image
        img_path = os.path.join(data_dir, 'images', f'sample_{i:03d}.jpg')
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create a corresponding label
        room_type = random.choice(room_types)
        
        # Create sample label data
        label_data = {
            'room_type': room_type,
            'confidence': random.uniform(0.7, 0.95),
            'walls': [
                {
                    'id': j + 1,
                    'type': 'wall',
                    'coordinates': {
                        'x1': random.uniform(0.0, 1.0),
                        'y1': random.uniform(0.0, 1.0),
                        'x2': random.uniform(0.0, 1.0),
                        'y2': random.uniform(0.0, 1.0)
                    }
                }
                for j in range(4)
            ],
            'objects': [
                {
                    'id': k + 1,
                    'type': random.choice(['sofa', 'bed', 'table', 'chair', 'tv']),
                    'coordinates': {
                        'x': random.uniform(0.1, 0.9),
                        'y': random.uniform(0.1, 0.9),
                        'width': random.uniform(0.1, 0.3),
                        'height': random.uniform(0.1, 0.3)
                    }
                }
                for k in range(random.randint(1, 5))
            ],
            'dimensions': {'width': 800, 'height': 600}
        }
        
        # Save the label
        label_path = os.path.join(data_dir, 'labels', f'sample_{i:03d}.json')
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)
    
    logger.info(f"Sample training data created in {data_dir}")


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Create sample data for demonstration
    sample_data_dir = "./sample_data"
    create_sample_training_data(sample_data_dir)
    
    # Train the model
    history = trainer.train_model(sample_data_dir, epochs=10, batch_size=4)
    
    print("Training completed!")