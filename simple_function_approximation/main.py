import numpy as np
import matplotlib.pyplot as plt

# - Создаем данные

# 100 точек на интервале [0, 1]
X = np.linspace(0, 1,100)

# Истиная функция, которую мы хотим аппроксимировать
def TestFunction(x):
    return np.sin(2 * np.pi * x)

y = TestFunction(X)

# - Обучаем модель

degree = 5

coeffs = np.polyfit(X, y, degree)
model = np.poly1d(coeffs)

# - Предсказание

X_test = np.linspace(0, 1, 200)

y_true = TestFunction(X_test)
y_pred = model(X_test)

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)

# - Визуализация
plt.figure(figsize=(8, 5))

plt.plot(X_test, y_true, label='True function (sin)', color='blue')
plt.plot(X_test, y_pred, label='Polynomial approximation', color='red', linestyle='--')

plt.scatter(X, y, label="Training data", color="black", s=20)

plt.title("Sin(x) vs Polynomial Approximation")
plt.legend()
plt.grid(True)

plt.show()