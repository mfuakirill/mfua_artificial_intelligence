"""Шаг 5. Сравнительный анализ: точности всех моделей + сводные графики.
Загружает сохранённые модели, оценивает их на тестовом наборе и строит
сравнительную диаграмму точности и общий график val_accuracy."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from common import load_data, RESULTS_DIR

(tra, tst) = load_data()
test_images, test_labels_one_hot, _ = tst

names, accs = [], []
for file, title in [('model_base.keras', 'Базовая\n(5 эпох, Adam)'),
                    ('model_improved.keras', 'Улучшенная\n(20 эпох, Adam)'),
                    ('model_rmsprop.keras', 'Улучшенная\n(20 эпох, RMSprop)')]:
    model = tf.keras.models.load_model(file)
    loss, acc = model.evaluate(test_images, test_labels_one_hot, verbose=0)
    names.append(title)
    accs.append(acc)
    print(f"{file}: точность = {acc * 100:.2f}%, потери = {loss:.4f}")

# Порог из задания
plt.figure(figsize=(9, 5))
bars = plt.bar(names, [a * 100 for a in accs], color=['#4C72B0', '#55A868',
                                                      '#C44E52'])
plt.axhline(95, color='red', linestyle='--', label='Порог 95%')
for bar, a in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f"{a * 100:.2f}%", ha='center')
plt.ylabel('Точность на тесте, %')
plt.title('Сравнение моделей')
plt.ylim(94, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=120)
print(f"Диаграмма сохранена: {RESULTS_DIR}/model_comparison.png")

# Сравнение кривых val_accuracy
data = np.load(f"{RESULTS_DIR}/compare_data.npz")
plt.figure(figsize=(9, 5))
plt.plot(data['hist_improved'], label='Улучшенная (Adam, 20 эпох)')
plt.plot(data['hist_rmsprop'], label='Улучшенная (RMSprop, 20 эпох)')
plt.xlabel('Эпоха')
plt.ylabel('Точность на валидации')
plt.title('Adam vs RMSprop: точность на валидации')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/optimizer_comparison.png", dpi=120)
print(f"График сохранён: {RESULTS_DIR}/optimizer_comparison.png")

print("\n--- Итоги ---")
print(f"Базовая (5 эпох, Adam):        {accs[0] * 100:.2f}%")
print(f"Улучшенная (20 эпох, Adam):    {accs[1] * 100:.2f}%")
print(f"Улучшенная (20 эпох, RMSprop): {accs[2] * 100:.2f}%")
print("Цель >= 95%:",
      "достигнута" if all(a >= 0.95 for a in accs) else "см. значения выше")
