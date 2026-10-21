import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)
x, lr, path = 10.0, 0.1, []

for i in range(30):
    path.append(x)
    x = x - lr * df(x)
print("Минимум функции", x)

xs = np.linspace(-2, 12, 100)
plt.figure()
plt.plot(xs, f(xs), label="f(x)")
plt.scatter(path, [f(p) for p in path], c='r', label="Шаги спуска")
plt.title("Оптимизация функции f(x) = (x-3)^2")
plt.legend()
plt.savefig('plot_1_gradient_descent.png') # Первый график
plt.close()

rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50)
y = 2 * X + 1 + rng.normal(0, 0.5, 50)

# График исходных данных
plt.figure()
plt.scatter(X, y)
plt.title("Исходные данные (Данные с шумом)")
plt.savefig('plot_2_raw_data.png')
plt.close()

# Обучение модели
def loss(w, b): 
    return np.mean((w * X + b - y) ** 2)

def grads(w, b):
    p = w * X + b
    return np.mean(2 * (p - y) * X), np.mean(2 * (p - y))

w, b, lr, hist = 0.0, 0.0, 0.05, []
for i in range(200):
    dw, db = grads(w, b)
    w -= lr * dw
    b -= lr * db
    hist.append(loss(w, b))

print("Обученные параметры: w =", w, ", b =", b)

# График функции потерь
plt.figure()
plt.plot(hist)
plt.title("График уменьшения ошибки")
plt.xlabel("Итерации")
plt.ylabel("MSE")
plt.savefig('plot_3_loss_history.png')
plt.close()

# График финальной прямой регрессии
plt.figure()
plt.scatter(X, y, label="Данные")
plt.plot(X, w * X + b, 'r', label=f"Регрессия: y = {w:.2f}x + {b:.2f}")
plt.title("Финальная линейная регрессия")
plt.legend()
plt.savefig('plot_4_regression_result.png')
plt.close()