import random
import math
import time

# Обучающая выборка: идеальные изображения цифр 0-9 (матрица 3×5 → строка из 15 символов)
num0 = [int(c) for c in '111101101101111']
num1 = [int(c) for c in '001001001001001']
num2 = [int(c) for c in '111001111100111']
num3 = [int(c) for c in '111001111001111']
num4 = [int(c) for c in '101101111001001']
num5 = [int(c) for c in '111100111001111']
num6 = [int(c) for c in '111100111101111']
num7 = [int(c) for c in '111001001001001']
num8 = [int(c) for c in '111101111101111']
num9 = [int(c) for c in '111101111001111']

# Объединение всех цифр в единый массив
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

def rand_matrix(rows, cols):
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

INPUT_SIZE = 15
HIDDEN_SIZE = 10
OUTPUT_SIZE = 10
LEARNING_RATE = 0.005
EPOCHS = 2000

# ---------- Параметры сети для дельта-правила ----------
W1 = rand_matrix(HIDDEN_SIZE, INPUT_SIZE)   # Веса скрытого слоя
b1 = [0.0] * HIDDEN_SIZE                    # Смещения скрытого слоя
W2 = rand_matrix(OUTPUT_SIZE, HIDDEN_SIZE)  # Веса выходного слоя
b2 = [0.0] * OUTPUT_SIZE                    # Смещения выходного слоя

# ---------- Параметры сети для правила Хебба ----------
# Один слой: 15 входов → 10 выходов, начальные веса нулевые
W_hebb = [[0.0] * INPUT_SIZE for _ in range(OUTPUT_SIZE)]
b_hebb = [0.0] * OUTPUT_SIZE

# Функции активации
def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# Прямой проход (сеть, обучаемая по дельта-правилу)
def forward(inputs):
    # Скрытый слой
    hidden = [relu(sum(inputs[i] * W1[j][i] for i in range(INPUT_SIZE)) + b1[j]) for j in range(HIDDEN_SIZE)]
    # Выходной слой
    logits = [sum(hidden[i] * W2[j][i] for i in range(HIDDEN_SIZE)) + b2[j] for j in range(OUTPUT_SIZE)]
    output = softmax(logits)
    return hidden, logits, output

# Обучение по дельта-правилу (с обратным распространением ошибки)
def train_delta():
    conv_epoch = None  # Эпоха, с которой все цифры распознаются верно (скорость сходимости)
    for epoch in range(EPOCHS):
        for label in range(10):
            inputs = nums[label]
            hidden, logits, output = forward(inputs)

            # Ошибка (дельта) выходного слоя: (y - t)
            error = list(output)
            error[label] -= 1.0

            # Обновление весов выходного слоя: Δw = -η · дельта · вход
            for j in range(OUTPUT_SIZE):
                for i in range(HIDDEN_SIZE):
                    W2[j][i] -= LEARNING_RATE * error[j] * hidden[i]
                b2[j] -= LEARNING_RATE * error[j]

            # Распространение ошибки назад на скрытый слой
            hidden_error = [0.0] * HIDDEN_SIZE
            for i in range(HIDDEN_SIZE):
                for j in range(OUTPUT_SIZE):
                    hidden_error[i] += W2[j][i] * error[j]
                hidden_error[i] *= relu_derivative(sum(inputs[k] * W1[i][k] for k in range(INPUT_SIZE)) + b1[i])

            # Обновление весов скрытого слоя
            for j in range(HIDDEN_SIZE):
                for i in range(INPUT_SIZE):
                    W1[j][i] -= LEARNING_RATE * hidden_error[j] * inputs[i]
                b1[j] -= LEARNING_RATE * hidden_error[j]

        # Метрика: после каждой эпохи проверяем, все ли 10 цифр распознаны верно
        if conv_epoch is None:
            ok = all(forward(nums[k])[2].index(max(forward(nums[k])[2])) == k for k in range(10))
            if ok:
                conv_epoch = epoch + 1
    return conv_epoch

# Обучение по правилу Хебба (один проход, без вычисления ошибки)
def train_hebb():
    for label in range(10):
        # Перевод пикселей из 0/1 в биполярный вид -1/+1
        x = [1 if v == 1 else -1 for v in nums[label]]
        for j in range(OUTPUT_SIZE):
            # Целевое значение: +1 для «своего» нейрона, -1 для остальных
            t = 1 if j == label else -1
            # Правило Хебба: Δw = x · t (вес растёт, если вход и выход совпадают по знаку)
            for i in range(INPUT_SIZE):
                W_hebb[j][i] += x[i] * t
            b_hebb[j] += t

# Предсказание сети, обученной по правилу Хебба
def predict_hebb(inputs):
    x = [1 if v == 1 else -1 for v in inputs]
    scores = [sum(W_hebb[j][i] * x[i] for i in range(INPUT_SIZE)) + b_hebb[j] for j in range(OUTPUT_SIZE)]
    return scores.index(max(scores))

# Обучение обеих сетей
print("Обучение по дельта-правилу...")
t0 = time.perf_counter()
conv_delta = train_delta()
time_delta = time.perf_counter() - t0
print("Обучение по правилу Хебба...")
t0 = time.perf_counter()
train_hebb()
time_hebb = time.perf_counter() - t0
print("Обучение завершено.\n")

# Сравнение результатов в консоли
errors_delta = 0
errors_hebb = 0
print("Цифра | Дельта-правило | Правило Хебба")
print("------+----------------+--------------")
for label in range(10):
    # Предсказание дельта-сети
    _, _, output = forward(nums[label])
    pred_delta = output.index(max(output))
    # Предсказание сети Хебба
    pred_hebb = predict_hebb(nums[label])

    mark_delta = "✓" if pred_delta == label else "✗"
    mark_hebb = "✓" if pred_hebb == label else "✗"
    if pred_delta != label:
        errors_delta += 1
    if pred_hebb != label:
        errors_hebb += 1

    print(f"  {label}   |    {pred_delta} {mark_delta}        |    {pred_hebb} {mark_hebb}")

print("\n================ МЕТРИКИ ================")
print("Скорость сходимости (эпох до 100% точности):")
print(f"  Дельта-правило: {conv_delta} эпох (из {EPOCHS}), время обучения = {time_delta:.4f} с")
print(f"  Правило Хебба:  1 проход (итеративной сходимости нет), время обучения = {time_hebb:.6f} с")
print("Конечная точность после обучения:")
print(f"  Дельта-правило: {100 - errors_delta * 10}% (ошибок: {errors_delta})")
print(f"  Правило Хебба:  {100 - errors_hebb * 10}% (ошибок: {errors_hebb})")
print("=========================================")
