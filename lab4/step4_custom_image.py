"""Шаг 4. Функция классификации пользовательского изображения (демонстрация).
Использует сохранённую улучшенную модель; для примера берёт изображения из
тестового набора и рисует предсказания с уверенностью модели."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from common import load_data, RESULTS_DIR

model_improved = tf.keras.models.load_model('model_improved.keras')


def classify_custom_image(model, image_array):
    """Классифицирует пользовательское изображение (28x28, нормализованное).

    Возвращает (предсказанный_класс, уверенность).
    """
    # Добавляем размерность батча: (28, 28) -> (1, 28, 28)
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_batch, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence


# --- Пример использования на тестовых изображениях ---
(_, _, _), (test_images, _, test_labels) = load_data()

print("\n--- Пример классификации отдельных изображений ---")
plt.figure(figsize=(10, 4))
for i in range(10):
    img = test_images[i]
    predicted_class, confidence = classify_custom_image(model_improved, img)
    ok = "OK" if predicted_class == test_labels[i] else "ОШИБКА"
    print(f"Изображение {i}: истинная метка = {test_labels[i]}, "
          f"предсказание = {predicted_class} (уверенность {confidence:.2%}) [{ok}]")
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Пред: {predicted_class} ({confidence:.0%})\n"
              f"Истина: {test_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/custom_classification.png", dpi=120)
print(f"Результаты сохранены: {RESULTS_DIR}/custom_classification.png")
