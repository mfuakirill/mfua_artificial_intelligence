"""Шаг 3. Исследование влияния Dropout и размера ядра свертки.

Эксперименты (каждый обучается на подвыборке из 10000 изображений,
10 эпох — чтобы уложиться в разумное время):
  A. Dropout: rate = 0.0 / 0.3 / 0.5 / 0.7  (kernel 3x3)
  B. Kernel size: 3x3 / 5x5 / 7x7           (dropout 0.5)

Результаты сравниваются по train/test accuracy — видно влияние
регуляризации (разрыв train-test) и размера ядра.
Сохраняется results_experiments.json.
"""
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils

SEED = 42
EPOCHS = 10
SUBSET = 10000  # ускорение экспериментов

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(kernel_size=(3, 3), dropout=0.5):
    """CNN с настраиваемым размером ядра и уровнем Dropout."""
    model = models.Sequential([
        layers.Conv2D(32, kernel_size, padding="same", activation="relu",
                      input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size, padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run(tag, **kw):
    t0 = time.time()
    model = build_model(**kw)
    model.fit(Xs, ys, batch_size=128, epochs=EPOCHS, verbose=0)
    train_acc = model.evaluate(Xs, ys, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    params = model.count_params()
    dt = time.time() - t0
    print(f"{tag:28s} train={train_acc:.4f}  test={test_acc:.4f}  "
          f"params={params}  {dt:.0f}s")
    return {"tag": tag, "train_acc": float(train_acc),
            "test_acc": float(test_acc), "params": int(params)}


# ---------- Данные ----------
# Данные локально (data/*.npz), см. 01_data_overview.py
_tr = np.load("data/cifar10_train.npz")
_te = np.load("data/cifar10_test.npz")
X_train, y_train = _tr["x"], _tr["y"]
X_test, y_test = _te["x"], _te["y"]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_test = utils.to_categorical(y_test, 10)

idx = np.random.RandomState(SEED).choice(len(X_train), SUBSET, replace=False)
Xs, ys = X_train[idx], utils.to_categorical(y_train[idx], 10)

# ---------- Эксперимент A: Dropout ----------
results = {"dropout": [], "kernel": []}
for dr in [0.0, 0.3, 0.5, 0.7]:
    results["dropout"].append(
        run(f"Dropout rate = {dr}", kernel_size=(3, 3), dropout=dr))

# ---------- Эксперимент B: размер ядра ----------
for ks in [(3, 3), (5, 5), (7, 7)]:
    results["kernel"].append(
        run(f"Kernel {ks[0]}x{ks[1]}", kernel_size=ks, dropout=0.5))

with open("results_experiments.json", "w") as f:
    json.dump(results, f, indent=2)
print("Сохранено: results_experiments.json")
