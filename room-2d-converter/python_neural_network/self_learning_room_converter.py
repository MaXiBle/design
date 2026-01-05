"""
Самообучающийся модуль преобразования фото комнаты в 2D-план
Использует предобученные подмодели для детекции объектов и обучается на основе пользовательской оценки (0-5)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
import json
import os
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as TF


class FurnitureDetector:
    """
    Предобученная модель для детекции мебели и других объектов в комнате
    Использует Faster R-CNN ResNet50 FPN
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Загрузка предобученной модели
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.to(self.device).eval()
        
        # Классы объектов, которые нас интересуют (в формате COCO)
        self.target_classes = {
            62: 'chair',      # стул
            84: 'vase',       # ваза
            56: 'bottle',     # бутылка
            67: 'dining table', # обеденный стол
            60: 'couch',      # диван
            72: 'potted plant', # комнатные растения
            63: 'sofa',       # диван
            58: 'bed',        # кровать
            65: 'toilet',     # унитаз
            71: 'tv',         # телевизор
            77: 'microwave',  # микроволновка
            78: 'oven',       # духовка
            79: 'toaster',    # тостер
            80: 'sink',       # раковина
            82: 'refrigerator' # холодильник
        }
        
        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Детекция объектов на изображении
        
        Args:
            image: входное изображение
            
        Returns:
            список словарей с информацией о детекциях
        """
        # Преобразование изображения в формат PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Применение трансформаций
        tensor_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Детекция
        with torch.no_grad():
            predictions = self.model(tensor_image)
        
        # Извлечение результатов
        detections = []
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        boxes = predictions[0]['boxes'].cpu().numpy()
        
        # Фильтрация по порогу уверенности
        threshold = 0.5
        for i in range(len(scores)):
            if scores[i] > threshold:
                label = int(labels[i])
                if label in self.target_classes:
                    detections.append({
                        'class': self.target_classes[label],
                        'confidence': float(scores[i]),
                        'bbox': boxes[i].tolist()  # [x1, y1, x2, y2]
                    })
        
        return detections


class WallDetector:
    """
    Модель для детекции стен и архитектурных элементов
    Использует OpenCV для обнаружения линий и контуров
    """
    def detect_walls(self, image: np.ndarray) -> List[Dict]:
        """
        Детекция стен и архитектурных элементов
        
        Args:
            image: входное изображение
            
        Returns:
            список словарей с информацией о стенах
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Улучшение изображения
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Обнаружение краев
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Обнаружение линий (стен) с помощью преобразования Хафа
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                walls.append({
                    'type': 'wall',
                    'coordinates': [x1, y1, x2, y2],
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                })
        
        # Также ищем прямоугольные структуры (комнаты)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        for contour in contours:
            # Приближаем контур к прямоугольнику
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:  # Прямоугольник или многоугольник
                room_coords = []
                for point in approx:
                    room_coords.append([int(point[0][0]), int(point[0][1])])
                
                rooms.append({
                    'type': 'room_boundary',
                    'coordinates': room_coords
                })
        
        return walls + rooms


class PlanGenerator:
    """
    Генератор 2D-плана на основе детекций
    """
    def __init__(self):
        self.furniture_detector = FurnitureDetector()
        self.wall_detector = WallDetector()
    
    def generate_plan(self, image: np.ndarray) -> Dict:
        """
        Генерация 2D-плана на основе изображения
        
        Args:
            image: входное изображение
            
        Returns:
            словарь с информацией о плане
        """
        # Детекция мебели и объектов
        furniture_detections = self.furniture_detector.detect_objects(image)
        
        # Детекция стен
        wall_detections = self.wall_detector.detect_walls(image)
        
        # Создание плана
        plan = {
            'image_shape': image.shape,
            'furniture': furniture_detections,
            'walls': wall_detections,
            'timestamp': datetime.now().isoformat()
        }
        
        return plan


class RewardDatabase:
    """
    База данных для хранения оценок пользователей и результатов
    """
    def __init__(self, db_path: str = 'rewards.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создание таблицы для хранения результатов и оценок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plan_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_data TEXT NOT NULL,
                user_score INTEGER NOT NULL CHECK(user_score >= 0 AND user_score <= 5),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_evaluation(self, plan_data: Dict, score: int):
        """Сохранение оценки пользователя"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO plan_evaluations (plan_data, user_score) VALUES (?, ?)',
            (json.dumps(plan_data), score)
        )
        
        conn.commit()
        conn.close()


class SelfLearningModel:
    """
    Основная модель, которая обучается на основе пользовательских оценок
    """
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Архитектура для оценки качества плана
        self.quality_evaluator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Выход от 0 до 1 (масштабируем к 0-5)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.quality_evaluator.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.plan_generator = PlanGenerator()
        self.reward_db = RewardDatabase()
    
    def extract_features(self, plan: Dict) -> torch.Tensor:
        """
        Извлечение признаков из плана для оценки качества
        
        Args:
            plan: словарь с информацией о плане
            
        Returns:
            тензор признаков
        """
        features = []
        
        # Количество мебели
        features.append(len(plan['furniture']))
        
        # Количество стен
        features.append(len([w for w in plan['walls'] if w['type'] == 'wall']))
        
        # Количество комнатных границ
        features.append(len([w for w in plan['walls'] if w['type'] == 'room_boundary']))
        
        # Средняя уверенность в детекциях мебели
        if plan['furniture']:
            avg_confidence = sum(f['confidence'] for f in plan['furniture']) / len(plan['furniture'])
            features.append(avg_confidence)
        else:
            features.append(0.0)
        
        # Средняя длина стен
        wall_lengths = [w['length'] for w in plan['walls'] if 'length' in w]
        if wall_lengths:
            avg_wall_length = sum(wall_lengths) / len(wall_lengths)
            features.append(avg_wall_length / 1000)  # Нормализуем
        else:
            features.append(0.0)
        
        # Количество различных типов мебели
        furniture_types = set(f['class'] for f in plan['furniture'])
        features.append(len(furniture_types))
        
        # Заполняем до 256 признаков (для совместимости с архитектурой)
        while len(features) < 256:
            features.append(0.0)
        
        # Ограничиваем до 256 признаков
        features = features[:256]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def evaluate_plan_quality(self, plan: Dict) -> float:
        """
        Оценка качества плана моделью (от 0 до 5)
        
        Args:
            plan: словарь с информацией о плане
            
        Returns:
            оценка качества (0-5)
        """
        features = self.extract_features(plan)
        with torch.no_grad():
            quality_score = self.quality_evaluator(features)
            # Масштабируем с 0-1 до 0-5
            scaled_score = quality_score.item() * 5.0
            return scaled_score
    
    def process_image(self, image_path: str) -> Dict:
        """
        Обработка изображения и генерация плана
        
        Args:
            image_path: путь к изображению
            
        Returns:
            словарь с информацией о плане и предсказанной оценке
        """
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Генерация плана
        plan = self.plan_generator.generate_plan(image)
        
        # Предсказание качества плана
        predicted_quality = self.evaluate_plan_quality(plan)
        
        # Добавляем предсказанную оценку в план
        plan['predicted_quality'] = predicted_quality
        
        return plan
    
    def update_with_feedback(self, plan: Dict, user_score: int):
        """
        Обновление модели на основе пользовательской оценки
        
        Args:
            plan: словарь с информацией о плане
            user_score: оценка пользователя (0-5)
        """
        # Извлечение признаков
        features = self.extract_features(plan)
        
        # Целевое значение (нормализуем к 0-1)
        target = torch.tensor([[user_score / 5.0]], dtype=torch.float32).to(self.device)
        
        # Обучение модели
        self.optimizer.zero_grad()
        predicted = self.quality_evaluator(features)
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        
        # Сохранение оценки в базу данных
        self.reward_db.save_evaluation(plan, user_score)
        
        return loss.item()


# API для интеграции с JavaScript
class RoomConversionAPI:
    """
    API для интеграции с JavaScript-приложением
    """
    def __init__(self):
        self.model = SelfLearningModel()
    
    def convert_room_to_plan(self, image_path: str) -> Dict:
        """
        Конвертация изображения комнаты в 2D-план
        
        Args:
            image_path: путь к изображению
            
        Returns:
            JSON с информацией о плане
        """
        try:
            plan = self.model.process_image(image_path)
            return {
                'success': True,
                'plan': plan
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def submit_feedback(self, plan_data: Dict, user_score: int) -> Dict:
        """
        Отправка пользовательской оценки для обучения модели
        
        Args:
            plan_data: данные плана
            user_score: оценка пользователя (0-5)
            
        Returns:
            JSON с результатом
        """
        try:
            if not (0 <= user_score <= 5):
                return {
                    'success': False,
                    'error': 'Оценка должна быть от 0 до 5'
                }
            
            loss = self.model.update_with_feedback(plan_data, user_score)
            
            return {
                'success': True,
                'message': f'Модель обучена, потеря: {loss:.4f}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Пример использования
if __name__ == "__main__":
    # Инициализация API
    api = RoomConversionAPI()
    
    # Пример использования:
    # plan_result = api.convert_room_to_plan('path/to/room_image.jpg')
    # if plan_result['success']:
    #     print(f"План сгенерирован, предсказанная оценка: {plan_result['plan']['predicted_quality']:.2f}")
    #     
    #     # Пользователь оценивает результат
    #     user_score = 4  # Например, пользователь поставил 4 из 5
    #     feedback_result = api.submit_feedback(plan_result['plan'], user_score)
    #     print(feedback_result)