import time
import cv2
import numpy as np

# Загрузка YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Вывод значения net.getUnconnectedOutLayers()
print(net.getUnconnectedOutLayers())

# Проверка, является ли возвращаемое значение итерируемым
if isinstance(net.getUnconnectedOutLayers(), (list, tuple)):
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
else:
    # Обработка случая, когда возвращаемое значение не является итерируемым
    output_layers = [layer_names[net.getUnconnectedOutLayers()[0] - 1]]



# Загрузка классов
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Словарь для хранения информации о каждом объекте
object_info = {}

# Определение рабочей области
work_area = (100, 100, 500, 500)  # Пример рабочей области: (x1, y1, x2, y2)

# Захват видеопотока
cap = cv2.VideoCapture(0)

while True:
    # Захват кадра
    ret, img = cap.read()
    height, width, channels = img.shape

    # Определение объектов
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Информация об объектах
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Объект обнаружен
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Координаты ограничивающего прямоугольника
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                object_id = str(class_id) + str(center_x) + str(center_y)  # Уникальный идентификатор объекта
                if object_id not in object_info:
                    object_info[object_id] = {
                        'class': classes[class_id],
                        'size': (w, h),
                        'distance': np.sqrt((center_x - (work_area[0] + work_area[2]) / 2) ** 2 + (center_y - (work_area[1] + work_area[3]) / 2) ** 2),  # Расстояние от центра рабочей области до центра объекта
                        'entry_time': time.time(),
                        'exit_time': None,
                        'parameters': []  # Список для фиксации параметров объекта
                    }

                # Фиксация параметров объекта
                object_info[object_id]['parameters'].append({
                    'center': (center_x, center_y),
                    'size': (w, h),
                    'time': time.time()
                })

                # Рисование ограничивающего прямоугольника и метки
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение изображения
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Вывод информации о каждом объекте
for obj_id, info in object_info.items():
    print(f"Object ID: {obj_id}")
    print(f"Class: {info['class']}")
    print(f"Size: {info['size']}")
    print(f"Distance: {info['distance']}")
    print(f"Entry Time: {info['entry_time']}")
    print(f"Exit Time: {info['exit_time']}")
    print(f"Parameters: {info['parameters']}")
    print("--------------------")

cap.release()
cv2.destroyAllWindows()