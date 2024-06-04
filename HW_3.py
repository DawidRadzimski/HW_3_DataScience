import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj dane
data = pd.read_csv("housing.csv")

# Wybór kolumn niezależnych i zależnych
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Normalizacja danych
X = (X - X.mean()) / X.std()

# Dodaj kolumnę jedynek dla terminu stałego
X = np.c_[np.ones(X.shape[0]), X]

# Implementacja hipotezy regresji liniowej
def hypothesis(X, theta):
    return np.dot(X, theta)

# Implementacja funkcji straty
def compute_loss(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)

# Implementacja jednego kroku zejścia gradientowego
def gradient_step(X, y, theta, learning_rate):
    m = len(y)
    h = hypothesis(X, theta)
    gradient = (1 / m) * np.dot(X.T, (h - y))
    theta -= learning_rate * gradient
    return theta

# Funkcja gradient descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    theta = np.zeros(X.shape[1])
    loss_history = []

    for i in range(iterations):
        theta = gradient_step(X, y, theta, learning_rate)
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return theta, loss_history

# Wykonaj gradient descent
learning_rate = 0.01
iterations = 1000
theta, loss_history = gradient_descent(X, y, learning_rate, iterations)

# Znajdź najlepsze parametry w za pomocą rozwiązania analitycznego
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

theta_analytical = normal_equation(X, y)

# Porównanie wyników
print("Theta (Gradient Descent):", theta)
print("Theta (Analytical):", theta_analytical)

# Wykres funkcji straty
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), loss_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Function during Gradient Descent')
plt.show()

# Wykres regresji liniowej dla jednej zmiennej (area)
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.regplot(x=data['area'], y=data['price'], ci=None, line_kws={"color":"red"})
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression: Price vs Area')
plt.show()

# Wykres regresji liniowej dla zmiennych area, bedrooms, bathrooms
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
sns.regplot(x=data['area'], y=data['price'], ax=axs[0], ci=None, line_kws={"color":"red"})
axs[0].set_title('Price vs Area')
sns.regplot(x=data['bedrooms'], y=data['price'], ax=axs[1], ci=None, line_kws={"color":"red"})
axs[1].set_title('Price vs Bedrooms')
sns.regplot(x=data['bathrooms'], y=data['price'], ax=axs[2], ci=None, line_kws={"color":"red"})
axs[2].set_title('Price vs Bathrooms')

for ax in axs:
    ax.set_xlabel('')
    ax.set_ylabel('Price')

plt.show()
