"""Общие утилиты: загрузка/предобработка MNIST и построение графиков обучения."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # сохраняем графики в файлы, без GUI
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

RESULTS_DIR = 'results'


def load_data():
    """Загрузка MNIST и предобработка: нормализация [0,1] + one-hot метки."""
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()

    print(f"Форма обучающих изображений: {train_images.shape}")
    print(f"Форма обучающих меток: {train_labels.shape}")
    print(f"Форма тестовых изображений: {test_images.shape}")
    print(f"Форма тестовых меток: {test_labels.shape}")
    print(f"Пример меток: {train_labels[:10]}")

    # а) Нормализация пикселей в диапазон [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    print(f"Диапазон пикселей после нормализации: "
          f"[{np.min(train_images)}, {np.max(train_images)}]")

    # б) One-hot кодирование меток
    train_labels_one_hot = to_categorical(train_labels, 10)
    test_labels_one_hot = to_categorical(test_labels, 10)
    print(f"Форма one-hot меток (обучение): {train_labels_one_hot.shape}")
    print(f"Пример one-hot метки '{train_labels[5]}': {train_labels_one_hot[5]}")

    return ((train_images, train_labels_one_hot, train_labels),
            (test_images, test_labels_one_hot, test_labels))


def plot_training_history(history, title, filename):
    """Графики точности и потерь на обучении/валидации."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Обучение (Accuracy)')
    ax1.plot(history.history['val_accuracy'], label='Валидация (Accuracy)')
    ax1.set_title(f'{title} - точность')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax2.plot(history.history['loss'], label='Обучение (Loss)')
    ax2.plot(history.history['val_loss'], label='Валидация (Loss)')
    ax2.set_title(f'{title} - потери')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    plt.tight_layout()
    path = f"{RESULTS_DIR}/{filename}"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"График сохранён: {path}")


def create_base_model():
    """Базовая модель: Flatten -> Dense(256, relu) -> Dense(10, softmax)."""
    model = models.Sequential(name='base_model')
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_improved_model(optimizer='adam'):
    """Улучшенная модель: больше нейронов и дополнительный слой."""
    model = models.Sequential(name='improved_model')
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(512, activation='relu'))  # увеличили нейроны
    model.add(layers.Dense(256, activation='relu'))  # добавили слой
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
