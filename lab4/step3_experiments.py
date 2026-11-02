"""Шаг 3. Эксперименты: улучшенная модель (20 эпох, Adam) и модель с RMSprop.
Сохраняет истории обучения в results/history_improved.npz / history_rmsprop.npz
для последующего сравнения."""
import numpy as np
from common import load_data, plot_training_history, create_improved_model

(tra, tst) = load_data()
train_images, train_labels_one_hot, _ = tra
test_images, test_labels_one_hot, _ = tst

# --- Эксперимент 1: улучшенная модель, Adam, 20 эпох ---
model_improved = create_improved_model(optimizer='adam')
model_improved.summary()

history_improved = model_improved.fit(train_images, train_labels_one_hot,
                                      epochs=20,
                                      batch_size=128,
                                      validation_data=(test_images,
                                                       test_labels_one_hot),
                                      verbose=1)
test_loss_imp, test_accuracy_imp = model_improved.evaluate(
    test_images, test_labels_one_hot, verbose=0)
print(f"\n--- Оценка улучшенной модели (Adam, 20 эпох) ---")
print(f"Точность (Accuracy): {test_accuracy_imp * 100:.2f}%")
print(f"Потери (Loss): {test_loss_imp:.4f}")
plot_training_history(history_improved, "Улучшенная модель (Adam, 20 эпох)",
                      "history_improved.png")
model_improved.save('model_improved.keras')

# --- Эксперимент 2: та же архитектура, оптимизатор RMSprop ---
model_rmsprop = create_improved_model(optimizer='rmsprop')
history_rmsprop = model_rmsprop.fit(train_images, train_labels_one_hot,
                                    epochs=20,
                                    batch_size=128,
                                    validation_data=(test_images,
                                                     test_labels_one_hot),
                                    verbose=1)
test_loss_rms, test_accuracy_rms = model_rmsprop.evaluate(
    test_images, test_labels_one_hot, verbose=0)
print(f"\n--- Оценка модели (RMSprop, 20 эпох) ---")
print(f"Точность (Accuracy): {test_accuracy_rms * 100:.2f}%")
print(f"Потери (Loss): {test_loss_rms:.4f}")
plot_training_history(history_rmsprop, "Модель (RMSprop, 20 эпох)",
                      "history_rmsprop.png")
model_rmsprop.save('model_rmsprop.keras')

# Сохраняем истории и точности для сравнения в step5
np.savez('results/compare_data.npz',
         acc_base_final=0.0,  # заполнится в step5 из base, либо игнорируем
         acc_improved=test_accuracy_imp,
         acc_rmsprop=test_accuracy_rms,
         hist_improved=history_improved.history['val_accuracy'],
         hist_rmsprop=history_rmsprop.history['val_accuracy'])
print("Данные для сравнения сохранены: results/compare_data.npz")
