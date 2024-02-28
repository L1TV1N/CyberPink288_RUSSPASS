import cv2

# Загрузка изображения
image = cv2.imread('C:\CyberPink288_RUSSPASS\OSN/365ebb3b81e68dab16a5e1b1e36d35d8.jpg')

# Загрузка предварительно обученной модели TensorFlow для обнаружения объектов
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb')

# Получение списка классов
with open('labels.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Подготовка изображения для распознавания
blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True)

# Задание входа и запуск модели
net.setInput(blob)
output = net.forward()

# Обработка результатов
for detection in output[0, 0, :, :]:
    confidence = detection[2]
    if confidence > 0.5:
        class_id = int(detection[1])
        class_name = classNames[class_id]
        if class_name in ['tree', 'mountain']:  # Проверяем, является ли класс деревом или горой
            # Получаем координаты прямоугольника, содержащего объект
            box = detection[3:7] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Отрисовка прямоугольника и надписи на изображении
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, class_name, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Отображение изображения с обнаруженными объектами
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
