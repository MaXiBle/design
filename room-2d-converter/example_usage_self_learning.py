"""
Пример использования самообучающейся модели преобразования фото комнаты в 2D-план
"""
from python_neural_network.self_learning_room_converter import RoomConversionAPI
import json

def main():
    # Инициализация API
    api = RoomConversionAPI()
    
    # Пример использования:
    print("Преобразование изображения комнаты в 2D-план...")
    
    # Замените 'path/to/room_image.jpg' на путь к вашему изображению
    image_path = 'path/to/room_image.jpg'
    
    try:
        # Генерация плана
        plan_result = api.convert_room_to_plan(image_path)
        
        if plan_result['success']:
            print(f"План сгенерирован успешно!")
            print(f"Предсказанная оценка качества: {plan_result['plan']['predicted_quality']:.2f}/5.0")
            print(f"Количество обнаруженных объектов мебели: {len(plan_result['plan']['furniture'])}")
            print(f"Количество обнаруженных стен: {len(plan_result['plan']['walls'])}")
            
            # Показать детали мебели
            print("\nОбнаруженная мебель:")
            for i, furniture in enumerate(plan_result['plan']['furniture']):
                print(f"  {i+1}. {furniture['class']} (уверенность: {furniture['confidence']:.2f})")
            
            # Здесь вы можете показать пользователю результат и получить его оценку
            # В реальном приложении это будет интерактивный процесс
            print("\nТеперь вы можете оценить результат от 0 до 5")
            print("Где 0 - максимально плохо, 5 - максимально хорошо")
            
            # В этом примере мы используем фиктивную оценку
            user_score = 4  # Вы оцениваете результат
            
            # Отправка оценки для обучения модели
            feedback_result = api.submit_feedback(plan_result['plan'], user_score)
            print(f"\nОбратная связь отправлена: {feedback_result}")
            
        else:
            print(f"Ошибка при генерации плана: {plan_result['error']}")
    
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()