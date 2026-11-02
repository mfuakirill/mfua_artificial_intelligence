"""
Лабораторная работа: Бинарная классификация сигналов радара (Sonar)
Нейронная сеть (Keras) для различения камней (R) и металлических цилиндров (M).

Модели:
  - Базовая:              60 -> 60 -> 1
  - Эксперимент 1:        60 -> 30 -> 1  (уменьшен первый скрытый слой)
  - Эксперимент 2:        60 -> 60 -> 15 -> 1  (дополнительный скрытый слой)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # сохраняем графики в файлы, без окна
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(42)  # воспроизводимость (set_random_seed удалён в TF 2.21)

# 3. Загрузка и анализ данных
dataframe = pd.read_csv("sonar.csv", header=None)
print("Размер датасета:", dataframe.shape)
print(dataframe.head())

dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y_labels = dataset[:, 60]  # строковые метки "R" / "M"

print(f"Форма X: {X.shape}")
print(f"Форма Y_labels: {Y_labels.shape}")
print(f"Уникальные метки: {np.unique(Y_labels)}")
print("Распределение классов:", dataframe[60].value_counts().to_dict())

# 4. Предобработка: Label Encoding (R -> 0, M -> 1)
encoder = LabelEncoder()
encoder.fit(Y_labels)
Y_encoded = encoder.transform(Y_labels)
print(f"Закодированные метки (первые 10): {Y_encoded[:10]}")


def make_model(hidden_sizes):
    """Полносвязная сеть: hidden_sizes — размеры скрытых слоёв, выход: 1 нейрон sigmoid."""
    model = Sequential()
    model.add(Dense(hidden_sizes[0], input_dim=60,
                    kernel_initializer="normal", activation="relu"))
    for n in hidden_sizes[1:]:
        model.add(Dense(n, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Обучение (Loss)")
    ax1.plot(history.history["val_loss"], label="Валидация (Val Loss)")
    ax1.set_title(f"{title} - Потери")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Потери")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Обучение (Accuracy)")
    ax2.plot(history.history["val_accuracy"], label="Валидация (Val Accuracy)")
    ax2.set_title(f"{title} - Точность")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Точность")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)


EPOCHS, BATCH_SIZE, VAL_SPLIT = 100, 10, 0.1

models = {
    "base": ("Базовая модель (60->60->1)", [60]),
    "exp1": ("Эксперимент 1 (60->30->1)", [30]),
    "exp2": ("Эксперимент 2 (60->60->15->1)", [60, 15]),
}

results = {}
for key, (title, hidden) in models.items():
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    model = make_model(hidden)
    model.summary()
    with open(f"summary_{key}.txt", "w") as f:
        model.summary(print_fn=f.write)

    history = model.fit(X, Y_encoded, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=VAL_SPLIT, verbose=1)

    plot_training_history(history, title, f"history_{key}.png")

    results[key] = {
        "title": title,
        "loss": min(history.history["loss"]),
        "val_loss": min(history.history["val_loss"]),
        "final_val_loss": history.history["val_loss"][-1],
        "accuracy": max(history.history["accuracy"]),
        "val_accuracy": max(history.history["val_accuracy"]),
        "final_val_accuracy": history.history["val_accuracy"][-1],
    }

print("\n" + "=" * 70)
print("Сравнение моделей (100 эпох, batch_size=10, validation_split=0.1)")
print("=" * 70)
rows = []
for key, r in results.items():
    rows.append({
        "Модель": r["title"],
        "min train acc": round(r["accuracy"], 4),
        "max val acc": round(r["val_accuracy"], 4),
        "min train loss": round(r["loss"], 4),
        "min val loss": round(r["val_loss"], 4),
        "final val acc": round(r["final_val_accuracy"], 4),
    })
report = pd.DataFrame(rows)
print(report.to_string(index=False))
report.to_csv("comparison.csv", index=False)
print("\nГрафики сохранены: history_base.png, history_exp1.png, history_exp2.png")
