import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Определение имен классов для CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Создание модели (аналогично вашему коду)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Выведем структуру созданной модели
model.summary()

# Загрузка обученных весов модели (если она уже обучена)
model.load_weights('D:/ML projects/ImageСlassification/model.h5')

print("модель загружена\n")

# Предобработка изображения для тестирования
image_path = 'D:/ML projects/ImageСlassification/labrador-retriever.png'
img = Image.open(image_path)
img = img.resize((32, 32))  # Масштабирование до размера 32x32
img = np.array(img) / 255.0  # Масштабирование значений пикселей к [0, 1]
img = img.reshape((1, 32, 32, 3))  # Добавление размерности пакета

# Предсказание класса изображения
predictions = model.predict(img)

# Отобразим предсказание
predicted_label = np.argmax(predictions)
confidence = np.max(tf.nn.softmax(predictions))

plt.imshow(img.reshape((32, 32, 3)))
plt.xticks([])
plt.yticks([])
plt.xlabel(f'{class_names[predicted_label]} with {confidence * 100:.2f}% confidence', color='blue')
plt.show()

model.save('D:/ML projects/ImageСlassification/model.h5')
print("Модель сохранена по пути: D:/ML projects/ImageСlassification/model.h5")