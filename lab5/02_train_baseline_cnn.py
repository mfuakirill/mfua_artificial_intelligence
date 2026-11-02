"""Шаг 2. Базовая CNN для классификации CIFAR-10.

Архитектура (типичная CNN из методички):
  [Conv2D(32) + ReLU -> Conv2D(32) + ReLU -> MaxPooling2D] x 2 ->
  -> Flatten -> Dense(128) + ReLU -> Dropout(0.5) -> Dense(10, softmax)

Оптимизатор: Adam, функция потерь: categorical_crossentropy, метрика: accuracy.
Обучение: 15 эпох. История и модель сохраняются на диск.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH = 128
EPOCHS = 15

# ---------- 1. Данные ----------
# Данные локально (data/*.npz), см. 01_data_overview.py
_tr = np.load("data/cifar10_train.npz")
_te = np.load("data/cifar10_test.npz")
X_train, y_train = _tr["x"], _tr["y"]
X_test, y_test = _te["x"], _te["y"]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train = utils.to_categorical(y_train, 10)   # one-hot
y_test = utils.to_categorical(y_test, 10)
print("Данные:", X_train.shape, X_test.shape)

# ---------- 2. Модель ----------
model = models.Sequential([
    # Блок 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                  input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    # Блок 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    # Классификатор
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ---------- 3. Обучение ----------
history = model.fit(X_train, y_train,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=2)

# ---------- 4. Оценка и сохранение ----------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nТестовая точность базовой CNN: {acc:.4f}")

model.save("cnn_baseline.keras")
with open("history_baseline.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
print("Сохранено: cnn_baseline.keras, history_baseline.json")
