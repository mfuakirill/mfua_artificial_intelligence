"""Шаг 2. Базовая модель: создание, обучение (5 эпох), оценка, графики."""
from common import load_data, plot_training_history, create_base_model

(tra, tst) = load_data()
train_images, train_labels_one_hot, train_labels = tra
test_images, test_labels_one_hot, test_labels = tst

# Создание и компиляция базовой модели
model_base = create_base_model()
model_base.summary()

# Обучение базовой модели (5 эпох)
history_base = model_base.fit(train_images, train_labels_one_hot,
                              epochs=5,
                              batch_size=128,
                              validation_data=(test_images, test_labels_one_hot),
                              verbose=1)

# Оценка на тестовых данных
test_loss, test_accuracy = model_base.evaluate(test_images, test_labels_one_hot,
                                               verbose=0)
print(f"\n--- Оценка базовой модели на тестовых данных ---")
print(f"Точность (Accuracy): {test_accuracy * 100:.2f}%")
print(f"Потери (Loss): {test_loss:.4f}")

# Визуализация обучения
plot_training_history(history_base, "Базовая модель (5 эпох)",
                      "history_base.png")

model_base.save('model_base.keras')
print("Модель сохранена: model_base.keras")
