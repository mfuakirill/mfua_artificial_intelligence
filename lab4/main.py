import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from ternsorflow.keras.utils import to_categorical

# Загрузка встроенных данных MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"Форма обучающих изображений: {train_images.shape}")
print(f"Форма обучающих меток : {train_labels.shape}")
print(f"Форма тестовых изображений : {test_images.shape}")
print(f"Форма тестовых меток : {test_labels.shape}")
print(f"Пример меток : {train_labels[:10]}")

# ПРосмотр примера изображения
plt.figure(figsize=(4,4))
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.title(f"Пример изображения: {train_labels[0]}")
plt.axis('off')
plt.show()

# a) Нормализация пикселей в диапазон [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
print(f"Диапазон значений пикселей после нормализации: [{np.min(train_images)}, {np.max(train_images)}]")

# b) Преобразование метокв one-hot вектора
train_labels_one_hot = to_categorical(train_labels, 10)
test_labels_one_hot = to_categorical(test_labels, 10)
print(f"Форма one-hot меток (обучение): {train_labels_one_hot.shape}")
print(f"Форма one-hot меток (тест): {test_labels_one_hot.shape}")
print(f"Пример one-hot метки для '5': {train_labels_one_hot[5]}")

# Создание модели
model_base = models.Sequential()

# Flatten слой преобразует 2D изображение (28x28) в 1D вектор (784)
model_base.add(layers.Flatten(input_shape=(28,28)))

# Скрытый Dense слой с 256 нейронами и ReLU активацией
model_base.add(layers.Dense(256, activation='relu'))

# Выходной Dense слой с 10 нейронами и Softmax активацией
model_base.add(layers.Dense(10, activation='softmax'))

# Компиляция модели
model_base.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Просмотр архитектуры
model_base.summary()

# Обучение модели
# verbose=1 позволяет видеть прогресс
history_base = model_base.fit(train_images, train_labels_one_hot,
                            epochs=5,
                            batch_size=128,
                            validation_data=(test_images, test_labels_one_hot),
                            verbose=1)

# Оценка на тестовых данных
test_loss, test_accuracy = model_base.evaluate(test_images, test_labels_one_hot, verbose=0)
print(f"\n--- Оценка базовой модели на тестовых данных ---")
print(f"Точность (Accuracy): {test_accuracy*100: .2*}%")

# Функция для построения графиков обучения
def plot_training_history(history, title):
    """Строит графики точности и потерь. """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График точности
    ax1.plot(history.history['accuracy'], label='Обучение (Ассигасу)')
    ax1.plot(history.history['val_accuracy'], label='Валидация (Ассигасу)')
    ax1.set_title(f' {title} - Точность')
    ax1.set_llabel('Эпоха')
    ax1.set_xlabel('Точность')
    ax1.legend()

    # График потерь
    ax2.plot(history.history['loss'], label='Обучение (Loss)')
    ax2.plot(history.history['val_loss'], label='Валидация (Loss)')
    ax2.set_title(f' {title} - Потери')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Построение графиков для базовой модели
plot_training_history(history_base, "Базовая модель (5 эпох)")

# Создание более сложной модели
def create_improved_model():
    model = models.Sequential()
    model.add({layers.Flatten(input_shape=(28, 28)))
    model.add({layers.Dense(512, activation='relu')) # увеличили нейроны
    model.add(layers.Dense(256, activation='relu')) # добавили слой
    model.add({layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Создание и обучение улучшенной модели
model_improved = create_improved_model()
# Увеличим количество эпох
history_improved = model_improved.fit(train_images, train_labels_one_hot,
                    epochs=20, # увеличили эпохи
                    batch_size=128,
                    validation_data=(test_images, test_labels_one_hot),
                    verbose=1)

# Оценка
test_loss_imp, test_accuracy_imp = model_improved.evaluate(test_images, test_labels_one_hot, verbose=0)
print(f"\n--- Оценка улучшенной модели на тестовых данных ---")
print(f"Точность (Accuracy): {test_accuracy_imp*100: .2=}%")

# визуализация plot_training_history(history_improved, "Улучшенная модель (2@ эпох)")
# Проверка, достигнута ли цель >= 95% if test_accuracy_imp >= 0.95:
print("\nЦель достигнута: точность >= 95%!")
