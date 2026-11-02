"""Шаг 1. Загрузка и анализ данных MNIST, пример изображения."""
import matplotlib.pyplot as plt
from common import load_data, RESULTS_DIR

(train_images, y_train_oh, y_train), _ = load_data()

# Просмотр нескольких примеров изображений
plt.figure(figsize=(8, 3))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(f"Метка: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mnist_examples.png", dpi=120)
print(f"Примеры изображений сохранены: {RESULTS_DIR}/mnist_examples.png")
