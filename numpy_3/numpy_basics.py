import numpy as np

a = np.array([3, 1, 4, 1, 5])
print(a * 2, a + 10, a ** 2)
print(a.mean(), a.min(), a.max(), a.sum())

print("\nформа")
M = np.arange(12).reshape(3, 4)
print(M, M.shape)
print('строка 0:', M[0])
print('столбец 1:', M[:, 1])
print('элемент:', M[2, 3])
print('трансп:', M.T.shape)

print("\n axis")
print('mean axis=0:', M.mean(axis=0))  
print('mean axis=1:', M.mean(axis=1))  

print("\nматричное умножение")
A = np.array([[1, 2, 3], [4, 5, 6]])   
w = np.array([[1], [0], [-1]])          
print('A @ w =', A @ w)                 
print('broadcasting:', A + np.array([10, 20, 30]))


print("\nлинейная модель для 100 объектов")
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))           
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5])  
b = 0.3
y = X @ w + b
print('X.shape, w.shape, y.shape:', X.shape, w.shape, y.shape)
print('первые 5 предсказаний:', y[:5])

print("титаник через нампай")
age = np.array([22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14])  
print('возраст mean/min/max:', np.nanmean(age), np.nanmin(age), np.nanmax(age))

X2 = rng.normal(size=(1000, 8))
w2 = rng.normal(size=(8,))
b2 = 0.5
y2 = X2 @ w2 + b2
print('X2.shape, w2.shape, y2.shape:', X2.shape, w2.shape, y2.shape)
print('Код не поменялся, @ автоматически подстраивает размерности:')
print('(1000, 8) @ (8,) = (1000,); b2 даёт (1000,)')