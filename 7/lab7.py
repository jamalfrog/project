import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

#Генерация + сохранение исх данных
X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="autumn", edgecolors="k")
plt.title("Исходные кластеры данных")
plt.savefig("1_clusters.png")  #Первый график
plt.close()

#Инициализация параметров модели
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2)
b = 0.0


def forward(X):
    return sig(X @ w + b)


print("Точность до обучения:", ((forward(X) > 0.5) == y).mean())


def bce(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


print("Исходная ошибка (Loss):", bce(forward(X), y))

#Цикл обучения (Градиентный спуск)
lr, hist = 0.1, []
for i in range(500):
    p = forward(X)
    dz = p - y
    w -= lr * X.T @ dz / len(X)
    b -= lr * dz.mean()
    hist.append(bce(p, y))

#Сохранение графика функции потерь
plt.figure(figsize=(6, 4))
plt.plot(hist, color="blue", linewidth=2)
plt.title("График функции потерь (BCE Loss)")
plt.xlabel("Эпохи обучения")
plt.ylabel("Ошибка")
plt.grid(True)
plt.savefig("2_loss.png")  #Второй график
plt.close()

print("Точность после обучения:", ((forward(X) > 0.5) == y).mean())

#Генерация сетки и сохранение разделяющей границы
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
)
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6, 4))
#Заливка областей вероятностей
plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, colors=["red", "yellow"])
#Сама разделяющая линия (где вероятность равна 0.5)
plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="autumn", edgecolors="k")
plt.title("Разделяющая граница классов")
plt.savefig("3_boundary.png")  #Третий график
plt.close()