import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib
matplotlib.use('TkAgg')  # Используйте другой бэкенд, например, TkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Загрузка данных CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Масштабирование значений пикселей к диапазону от 0 до 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Проверим размеры загруженных данных
print("Train images shape:", train_images.shape)  # (50000, 32, 32, 3)
print("Train labels shape:", train_labels.shape)  # (50000, 1)
print("Test images shape:", test_images.shape)    # (10000, 32, 32, 3)
print("Test labels shape:", test_labels.shape)    # (10000, 1)

# Определение имен классов для CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Точность на тестовых данных: {test_acc}")

# Графики для анализа процесса обучения
plt.figure(figsize=(10, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучающих данных', marker='o')
plt.plot(history.history['val_accuracy'], label='Точность на проверочных данных', marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучающих данных', marker='o')
plt.plot(history.history['val_loss'], label='Потери на проверочных данных', marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

plt.tight_layout()
plt.show()

model.save('D:/ML projects/ImageСlassification/model.h5')
print("Модель сохранена по пути: D:/ML projects/ImageСlassification/model.h5")

# Функция для отображения изображений с предсказаниями
def plot_predictions(images, labels, predictions):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
        predicted_label = np.argmax(predictions[i])
        true_label = labels[i, 0]
        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'
        plt.xlabel(f'{class_names[predicted_label]} ({class_names[true_label]})', color=color)
    plt.tight_layout()
    plt.show()

# Получим предсказания для тестовых данных
predictions = model.predict(test_images)

# Визуализируем первые 25 изображений и их предсказания
plot_predictions(test_images[:25], test_labels[:25], predictions[:25])

# Загрузка ваших данных для тестирования
# Предполагаем, что у вас есть изображение для тестирования
# Загрузите изображение и выполните необходимую предобработку
image_path = 'D:\ML projects\ImageСlassification\labrador-retriever.png'
img = Image.open(image_path)
img = img.resize((32, 32))  # Масштабирование до размера 32x32
img = np.array(img) / 255.0  # Масштабирование значений пикселей к [0, 1]
img = img.reshape((1, 32, 32, 3))  # Добавление размерности пакета

# Предсказание класса изображения
predictions = model.predict(img)

# Отобразим предсказание
predicted_label = np.argmax(predictions)
confidence = np.max(tf.nn.softmax(predictions))

plt.imshow(np.squeeze(img))  # np.squeeze для удаления избыточной размерности
plt.xticks([])
plt.yticks([])
plt.xlabel(f'{class_names[predicted_label]} with {confidence * 100:.2f}% confidence', color='blue')
plt.show()
