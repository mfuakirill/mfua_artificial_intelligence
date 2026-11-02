"""Шаг 1. Загрузка и обзор набора данных CIFAR-10."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Данные лежат локально в data/ (train/test), т.к. встроенный загрузчик
# keras.datasets.cifar10 качает архив с внешнего сервера (медленно).
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# 1. Загрузка данных
import numpy as np
_tr = np.load("data/cifar10_train.npz")
_te = np.load("data/cifar10_test.npz")
X_train, y_train = _tr["x"], _tr["y"]
X_test, y_test = _te["x"], _te["y"]

# 2. Обзор форм и диапазона значений
print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)
print("Мин/макс пикселей:", X_train.min(), "/", X_train.max())

# 3. Распределение классов
y = y_train.ravel()
print("Количество изображений по классам:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:12s}: {(y == i).sum()}")

# 4. Нормировка в [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print("После нормировки: мин =", X_train.min(), ", макс =", X_train.max())

# 5. Визуализация примеров (по одному на класс)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for k, ax in enumerate(axes.ravel()):
    idx = int(np.argmax(y == k))          # первый пример класса k
    ax.imshow(X_train[idx])
    ax.set_title(f"{k}: {CLASS_NAMES[k]}", fontsize=10)
    ax.axis("off")
fig.suptitle("CIFAR-10: примеры изображений")
fig.tight_layout()
fig.savefig("fig_01_samples.png", dpi=130)
print("Сохранено: fig_01_samples.png")