"""
Лабораторная работа: Классификация цветков ириса с помощью ИНС в Keras.

Задачи:
  1. Загрузить и предварительно обработать датасет Iris.
  2. Построить модель ИНС (Sequential, Dense) в Keras.
  3. Скомпилировать (adam, categorical_crossentropy, accuracy).
  4. Обучить и оценить, провести эксперименты с архитектурами и гиперпараметрами.
  5. Визуализировать обучение (графики потерь и точности), выбрать лучшую модель.

Запуск:  python3 lab1_iris.py
Результаты: папка out/ (графики + сравнение моделей).
"""

import os
import random

random.seed(42)

import pandas as pd
import numpy as np

np.random.seed(42)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

OUT = "out"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Загрузка данных (Листинг 1, 3)
# ---------------------------------------------------------------------------
# В нашем файле есть заголовок и колонка Id — отбрасываем их,
# дальше всё как в методичке (первые 4 столбца — признаки, 5-й — метка).
df = pd.read_csv("iris.csv")
df = df.drop(columns=["Id"])
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# ---------------------------------------------------------------------------
# 2. Преобразование меток (Листинг 5): LabelEncoder + one-hot encoding
# ---------------------------------------------------------------------------
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)          # [0, 0, ..., 1, ..., 2, 2]
dummy_y = to_categorical(encoded_Y)       # [[1,0,0], ..., [0,0,1]]
print("Классы:", dict(zip(encoder.classes_, range(3))))
print("X:", X.shape, " dummy_y:", dummy_y.shape)


def build_model(hidden_layers, input_dim=4, activation="relu"):
    """Сборка Sequential-модели: скрытые слои + выходной softmax на 3 класса."""
    model = Sequential()
    for i, units in enumerate(hidden_layers):
        model.add(Dense(units, input_dim=input_dim, activation=activation)
                    if i == 0 else Dense(units, activation=activation))
    model.add(Dense(3, activation="softmax"))
    return model


# ---------------------------------------------------------------------------
# 3-6. Эксперименты: разные архитектуры и гиперпараметры (Листинг 6, 7)
# ---------------------------------------------------------------------------
experiments = [
    # (имя, скрытые слои, оптимизатор, epochs, batch_size)
    ("baseline 8 нейронов, adam, 150ep, bs=10", [8],            "adam",     150, 10),
    ("4 нейрона (как в исходном листинге)",     [4],            "adam",     150, 10),
    ("16 нейронов",                             [16],           "adam",     150, 10),
    ("2 скрытых слоя 8+8",                      [8, 8],         "adam",     150, 10),
    ("2 скрытых слоя 16+8, tanh",               [16, 8],        "adam",     150, 10),
    ("baseline, sgd",                           [8],            "sgd",      300, 10),
    ("baseline, rmsprop",                       [8],            "rmsprop",  150, 10),
    ("batch_size=5",                            [8],            "adam",     150, 5),
    ("batch_size=32, 300ep",                    [8],            "adam",     300, 32),
]

results = []
for name, hidden, optimizer, epochs, bs in experiments:
    model = build_model(hidden)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(X, dummy_y,
                        epochs=epochs,
                        batch_size=bs,
                        validation_split=0.2,   # 20% данных для валидации
                        verbose=0)
    val_acc = history.history["val_accuracy"][-1]   # конечная val-точность (как в методичке)
    val_loss = history.history["val_loss"][-1]
    results.append({
        "эксперимент": name,
        "hidden": hidden,
        "optimizer": optimizer,
        "epochs": epochs,
        "batch_size": bs,
        "train_acc": history.history["accuracy"][-1],
        "val_acc": val_acc,
        "val_loss": history.history["val_loss"][-1],
        "history": history,
    })
    print(f"{name:<45} train_acc={results[-1]['train_acc']:.4f} "
          f"val_acc={val_acc:.4f} val_loss={results[-1]['val_loss']:.4f}")

# ---------------------------------------------------------------------------
# Требование 3: графики потерь и точности для каждой модели
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(results), 2, figsize=(10, 3 * len(results)))
for i, r in enumerate(results):
    h = r["history"].history
    axes[i, 0].plot(h["loss"], label="train loss")
    axes[i, 0].plot(h["val_loss"], label="val loss")
    axes[i, 0].set_title(f"Loss: {r['эксперимент']}")
    axes[i, 0].legend()
    axes[i, 1].plot(h["accuracy"], label="train acc")
    axes[i, 1].plot(h["val_accuracy"], label="val acc")
    axes[i, 1].set_title(f"Accuracy: {r['эксперимент']}")
    axes[i, 1].legend()
fig.tight_layout()
fig.savefig(f"{OUT}/all_experiments.png", dpi=100)
print(f"\nГрафики всех экспериментов: {OUT}/all_experiments.png")

# ---------------------------------------------------------------------------
# Требование 4: выбор лучшей модели по валидационной точности
# ---------------------------------------------------------------------------
# При равной валидационной точности лучшей считается модель с меньшим val_loss
best = max(results, key=lambda r: (r["val_acc"], -r["val_loss"]))
print("\nСравнение моделей (по val_accuracy):")
for r in sorted(results, key=lambda r: -r["val_acc"]):
    print(f"  val_acc={r['val_acc']:.4f} train_acc={r['train_acc']:.4f}  {r['эксперимент']}")

print(f"\nЛучшая модель: {best['эксперимент']} "
      f"(val_acc={best['val_acc']:.4f}, val_loss={best['val_loss']:.4f})")

# График только лучшей модели — отдельно, для отчёта
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(best["history"].history["loss"], label="train")
plt.plot(best["history"].history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(best["history"].history["accuracy"], label="train")
plt.plot(best["history"].history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/best_model_history.png", dpi=120)
print(f"График лучшей модели: {OUT}/best_model_history.png")

# ---------------------------------------------------------------------------
# Оценка лучшей модели на независимом тестовом наборе
# (80/20 stratified split: тестовые данные модель не видела никогда)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, encoded_Y, test_size=0.2,
                                          stratify=encoded_Y, random_state=42)
model = build_model(best["hidden"])
model.compile(optimizer=best["optimizer"],
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, to_categorical(y_tr), epochs=best["epochs"],
          batch_size=best["batch_size"], verbose=0)
loss, acc = model.evaluate(X_te, to_categorical(y_te), verbose=0)
pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
print(f"\nЛучшая модель на тестовом наборе (30 образцов, не участвовавших в обучении):")
print(f"  test accuracy = {acc:.4f}, test loss = {loss:.4f}")
for ci, cname in enumerate(encoder.classes_):
    m = y_te == ci
    print(f"  {cname}: правильно {int((pred[m] == ci).sum())}/{int(m.sum())}")
