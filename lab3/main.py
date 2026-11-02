# Лабораторная работа: Регрессия — Предсказание цен на жилье в Бостоне (Boston Housing)
# Нейронная сеть (Keras) для предсказания медианной цены домов по 13 признакам.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI, графики сохраняем в файлы
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# 3. Загрузка и анализ данных
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"Размер обучающих данных: {train_data.shape}")
print(f"Размер тестовых данных:  {test_data.shape}")
print(f"Размер обучающих меток:  {train_targets.shape}")
print(f"Размер тестовых меток:   {test_targets.shape}")
print(f"Первые 5 цен в обучающем наборе: {train_targets[:5]}")
print(f"Диапазон цен в обучающем наборе: {np.min(train_targets)}-{np.max(train_targets)}")
print("\nСредние значения первых 5 признаков в обучающем наборе (до нормализации):")
print(train_data.mean(axis=0)[:5])
print("Стандартные отклонения первых 5 признаков в обучающем наборе (до нормализации):")
print(train_data.std(axis=0)[:5])

# 4. Предобработка данных: нормализация (Z-score, статистика только по train)
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std = np.where(std == 0, 1, std)  # защита от деления на ноль

train_data_norm = (train_data - mean) / std
test_data_norm = (test_data - mean) / std

print("\nСредние значения первых 5 признаков после нормализации (должны быть близки к 0):")
print(train_data_norm.mean(axis=0)[:5])
print("Стандартные отклонения первых 5 признаков после нормализации (должны быть близки к 1):")
print(train_data_norm.std(axis=0)[:5])

train_data, test_data = train_data_norm, test_data_norm

# 5. Создание и компиляция базовой модели
def build_model():
    """Создает и возвращает модель для регрессии цен на жилье."""
    model = Sequential()
    # Входной размер 13, два скрытых слоя по 64 нейрона с ReLU
    model.add(Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation="relu"))
    # Выходной слой: 1 нейрон без активации (линейный) для регрессии
    model.add(Dense(1))
    # Компиляция: оптимизатор rmsprop, функция потерь mse, метрика mae
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

model_base = build_model()
model_base.summary()

# 6. Обучение базовой модели (100 эпох)
num_epochs = 100
history_base = model_base.fit(
    train_data, train_targets,
    epochs=num_epochs,
    batch_size=16,
    validation_data=(test_data, test_targets),  # мониторинг на тестовых данных
    verbose=1,
)

test_mse, test_mae = model_base.evaluate(test_data, test_targets, verbose=0)
print("\n--- Оценка базовой модели на тестовых данных ---")
print(f"Средняя квадратурная ошибка (MSE): {test_mse:.2f}")
print(f"Средняя абсолютная ошибка (MAE): ${test_mae * 1000:.2f}")

# 7. Визуализация результатов базовой модели
def plot_training_history(history, title, filename):
    """Строит графики потерь (MSE) и MAE для одной истории обучения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Обучение (MSE)")
    ax1.plot(history.history["val_loss"], label="Валидация (MSE)")
    ax1.set_title(f"{title} - Потери (MSE)")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["mae"], label="Обучение (MAE)")
    ax2.plot(history.history["val_mae"], label="Валидация (MAE)")
    ax2.set_title(f"{title} - Средняя абсолютная ошибка (MAE)")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, filename), dpi=120)
    plt.close(fig)

plot_training_history(history_base, "Базовая модель", "base_100_epochs.png")

# 8. Эксперимент: K-Fold перекрестная проверка (k=4)
print("\n--- K-Fold перекрестная проверка ---")
k = 4
num_val_samples = len(train_data) // k
num_epochs_cv = 100
all_mae_histories = []

for i in range(k):
    print(f"Обработка блока #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=num_epochs_cv,
        batch_size=16,
        validation_data=(val_data, val_targets),
        verbose=0,
    )
    all_mae_histories.append(history.history["val_mae"])
    print(f"  Блок #{i}: финальная val_mae = {history.history['val_mae'][-1]:.4f}")

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs_cv)
]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history,
         label="Средняя MAE (K-Fold CV)")
plt.title("Усреднённая MAE по 4 блокам (K-Fold Cross-Validation)")
plt.xlabel("Эпоха")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kfold_avg_mae.png"), dpi=120)
plt.close()

# Оценка стабильности: разброс финальных MAE между фолдами
final_fold_maes = [h[-1] for h in all_mae_histories]
if len(average_mae_history) >= 10:
    avg_last_10_mae = np.mean(average_mae_history[-10:])
    print(f"\nСредняя MAE за последние 10 эпох (K-Fold): {avg_last_10_mae:.4f} (${avg_last_10_mae * 1000:.2f})")
    print(f"Финальные val_mae по фолдам: {[f'{v:.4f}' for v in final_fold_maes]}")
    print(f"Стандартное отклонение финальных MAE по фолдам: {np.std(final_fold_maes):.4f}")

# 9. Эксперимент: изменение количества эпох (50)
print("\n--- Обучение модели с 50 эпохами ---")
model_exp = build_model()
history_exp = model_exp.fit(
    train_data, train_targets,
    epochs=50,
    batch_size=16,
    validation_data=(test_data, test_targets),
    verbose=1,
)

test_mse_exp, test_mae_exp = model_exp.evaluate(test_data, test_targets, verbose=0)
print("\n--- Оценка модели (50 эпох) на тестовых данных ---")
print(f"Средняя квадратурная ошибка (MSE): {test_mse_exp:.2f}")
print(f"Средняя абсолютная ошибка (MAE): ${test_mae_exp * 1000:.2f}")

plot_training_history(history_exp, "Модель (50 эпох)", "exp_50_epochs.png")

# 10. Сравнение и анализ результатов
print("\n--- Сравнение моделей ---")
print(f"Базовая модель (100 эпох):  MSE={test_mse:.2f}, MAE={test_mae:.4f} (${test_mae * 1000:.2f})")
print(f"Модель (50 эпох):           MSE={test_mse_exp:.2f}, MAE={test_mae_exp:.4f} (${test_mae_exp * 1000:.2f})")
print(f"K-Fold CV (средняя MAE, последние 10 эпох): {np.mean(average_mae_history[-10:]):.4f}")

# Точка, где val_loss базовой модели минимален (признак начала переобучения)
val_loss_base = history_base.history["val_loss"]
best_epoch = int(np.argmin(val_loss_base)) + 1
print(f"Минимум val_loss базовой модели на эпохе {best_epoch} (val_loss={val_loss_base[best_epoch - 1]:.2f}); "
      f"на тесте val_loss={val_loss_base[-1]:.2f}")
val_mae_base = history_base.history["val_mae"]
best_epoch_mae = int(np.argmin(val_mae_base)) + 1
print(f"Минимум val_mae базовой модели на эпохе {best_epoch_mae} (val_mae={val_mae_base[best_epoch_mae - 1]:.4f})")
