"""
Конвертер фото комнаты в 2D план с использованием глубокого обучения
Этот модуль реализует нейронную сеть для преобразования фото комнат в 2D планы помещений.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoomTo2DConverter:
    """
    Класс нейронной сети для преобразования фото комнат в 2D планы.
    Использует архитектуру на основе сверточных нейронных сетей для анализа изображений и генерации планов.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализирует конвертер фото комнаты в 2D план.
        
        Args:
            model_path: Путь к предварительно обученной модели. Если None, создает новую модель.
        """
        self.model = None
        self.is_trained = False
        self.input_shape = (224, 224, 3)  # Standard input size
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """
        Создает модель нейронной сети для преобразования фото комнат в 2D планы.
        Архитектура включает:
        - Извлечение признаков с использованием слоев CNN
        - Классификацию типа комнаты
        - Сегментацию элементов
        """
        # Input layer
        input_layer = keras.Input(shape=self.input_shape)
        
        # Feature extraction backbone (using a simplified ResNet-like structure)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Shared features
        shared_features = layers.GlobalAveragePooling2D()(x)
        
        # Room type classification branch
        room_type_branch = layers.Dense(128, activation='relu')(shared_features)
        room_type_branch = layers.Dropout(0.5)(room_type_branch)
        room_type_output = layers.Dense(4, activation='softmax', name='room_type')(room_type_branch)  # 4 room types
        
        # Wall segmentation branch (simplified as bounding boxes)
        wall_branch = layers.Dense(512, activation='relu')(shared_features)
        wall_branch = layers.Dropout(0.3)(wall_branch)
        wall_output = layers.Dense(8, activation='linear', name='walls')(wall_branch)  # 4 wall coordinates (x1,y1,x2,y2 for each)
        
        # Object detection branch (simplified as bounding boxes for furniture)
        object_branch = layers.Dense(512, activation='relu')(shared_features)
        object_branch = layers.Dropout(0.3)(object_branch)
        object_output = layers.Dense(20, activation='linear', name='objects')(object_branch)  # 5 objects with 4 coords each
        
        # Confidence score
        confidence_branch = layers.Dense(64, activation='relu')(shared_features)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(confidence_branch)
        
        # Create model
        self.model = keras.Model(
            inputs=input_layer,
            outputs=[room_type_output, wall_output, object_output, confidence_output]
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'room_type': 'categorical_crossentropy',
                'walls': 'mse',
                'objects': 'mse',
                'confidence': 'mse'
            },
            metrics={
                'room_type': 'accuracy',
                'walls': 'mae',
                'objects': 'mae',
                'confidence': 'mae'
            }
        )
        
        logger.info("Model built successfully")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Предобрабатывает входное изображение для нейронной сети.
        
        Args:
            image_path: Путь к входному изображению
            
        Returns:
            Предобработанное изображение в виде массива numpy
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path: str) -> Dict:
        """
        Предсказывает 2D план по фото комнаты.
        
        Args:
            image_path: Путь к входному фото комнаты
            
        Returns:
            Словарь, содержащий информацию о 2D плане
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        room_type_pred, walls_pred, objects_pred, confidence_pred = predictions
        
        # Decode predictions
        room_types = ['living_room', 'bedroom', 'kitchen', 'bathroom']
        predicted_room_type_idx = np.argmax(room_type_pred[0])
        predicted_room_type = room_types[predicted_room_type_idx]
        confidence = float(confidence_pred[0][0])
        
        # Process walls (simplified)
        walls = self._decode_walls(walls_pred[0])
        
        # Process objects (simplified)
        objects = self._decode_objects(objects_pred[0], predicted_room_type)
        
        # Create result dictionary
        result = {
            'room_type': predicted_room_type,
            'confidence': confidence,
            'walls': walls,
            'objects': objects,
            'dimensions': {'width': 800, 'height': 600}  # Default dimensions
        }
        
        return result
    
    def _decode_walls(self, walls_pred: np.ndarray) -> List[Dict]:
        """
        Декодирует предсказания стен из выходных данных нейронной сети.
        
        Args:
            walls_pred: Необработанные предсказания стен из модели
            
        Returns:
            Список объектов стен
        """
        walls = []
        
        # Walls are represented as 4 coordinates (x1, y1, x2, y2) for each of 4 walls
        for i in range(4):  # 4 walls
            start_idx = i * 2
            end_idx = start_idx + 2
            
            # Normalize coordinates to image dimensions
            x1 = max(0, min(1, walls_pred[start_idx]))
            y1 = max(0, min(1, walls_pred[start_idx + 1]))
            x2 = max(0, min(1, walls_pred[end_idx]))
            y2 = max(0, min(1, walls_pred[end_idx + 1]))
            
            wall = {
                'id': i + 1,
                'type': 'wall',
                'coordinates': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'color': '#d2b48c'  # Default wall color
            }
            walls.append(wall)
        
        return walls
    
    def _decode_objects(self, objects_pred: np.ndarray, room_type: str) -> List[Dict]:
        """
        Декодирует предсказания объектов из выходных данных нейронной сети на основе типа комнаты.
        
        Args:
            objects_pred: Необработанные предсказания объектов из модели
            room_type: Тип комнаты для определения ожидаемых объектов
            
        Returns:
            Список словарей объектов
        """
        objects = []
        
        # Define objects based on room type
        room_objects = {
            'living_room': ['sofa', 'coffee_table', 'tv', 'chair', 'plant'],
            'bedroom': ['bed', 'nightstand', 'wardrobe', 'desk', 'chair'],
            'kitchen': ['counter', 'refrigerator', 'stove', 'sink', 'dining_table'],
            'bathroom': ['toilet', 'sink', 'bathtub', 'shower', 'cabinet']
        }
        
        available_objects = room_objects.get(room_type, [])
        
        # Each object has 4 coordinates (x, y, width, height)
        for i, obj_type in enumerate(available_objects):
            if i >= 5:  # Limit to 5 objects
                break
                
            start_idx = i * 4
            if start_idx + 3 >= len(objects_pred):
                break
                
            # Extract coordinates (normalized)
            x = max(0, min(1, objects_pred[start_idx]))
            y = max(0, min(1, objects_pred[start_idx + 1]))
            width = max(0.05, min(0.3, objects_pred[start_idx + 2]))  # Min width 5%, max 30%
            height = max(0.05, min(0.3, objects_pred[start_idx + 3]))  # Min height 5%, max 30%
            
            obj = {
                'id': i + 1,
                'type': obj_type,
                'coordinates': {
                    'x': float(x),
                    'y': float(y),
                    'width': float(width),
                    'height': float(height)
                }
            }
            objects.append(obj)
        
        return objects
    
    def save_model(self, filepath: str):
        """
        Сохраняет обученную модель на диск.
        
        Args:
            filepath: Путь, по которому нужно сохранить модель
        """
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            raise ValueError("No model to save")
    
    def load_model(self, filepath: str):
        """
        Загружает предварительно обученную модель с диска.
        
        Args:
            filepath: Путь к сохраненной модели
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """
        Обучает модель на предоставленных данных.
        
        Args:
            train_data: Генератор данных для обучения или кортеж (X, y)
            val_data: Генератор данных для валидации или кортеж (X, y)
            epochs: Количество эпох обучения
            batch_size: Размер пакета для обучения
        """
        logger.info("Starting model training...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return history


# Example usage function
def convert_room_to_2d_plan(image_path: str, model_path: Optional[str] = None) -> Dict:
    """
    Удобная функция для преобразования фото комнаты в 2D план.
    
    Args:
        image_path: Путь к фото комнаты
        model_path: Путь к предварительно обученной модели (опционально)
        
    Returns:
        2D план в виде словаря
    """
    converter = RoomTo2DConverter(model_path)
    return converter.predict(image_path)