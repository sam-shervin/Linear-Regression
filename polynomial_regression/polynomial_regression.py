import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Equation:  y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g
a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0

def predict(x):
    return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g

def error(y, y_pred):
    return np.mean((y_pred - y)**2)

def gradient(x, y, y_pred, num):
    if num == 1:
        return 2 * np.mean((y_pred - y) * x**6)
    elif num == 2:
        return 2 * np.mean((y_pred - y) * x**5)
    elif num == 3:
        return 2 * np.mean((y_pred - y) * x**4)
    elif num == 4:
        return 2 * np.mean((y_pred - y) * x**3)
    elif num == 5:
        return 2 * np.mean((y_pred - y) * x**2)
    elif num == 6:
        return 2 * np.mean((y_pred - y) * x)
    else:
        return 2 * np.mean(y_pred - y)

# Hyperparameters
lr = 1e-13
epochs = 5000000000

# Load and normalize dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
y = y / np.max(y)

# Plot full dataset
plt.scatter(x, y)
plt.savefig('Salary_Data.png')
plt.close()

# Random split
perm = np.random.permutation(len(x))
train_size = int(len(x) * 2 / 3)
train_idx, test_idx = perm[:train_size], perm[train_size:]

x_train, x_test = x[train_idx], x[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Initial plots
plt.scatter(x_train, y_train, color='red')
plt.savefig('Salary_Data_train.png')
plt.close()

plt.scatter(x_test, y_test, color='blue')
plt.savefig('Salary_Data_test.png')
plt.close()

# Training
errors = []
x_min, x_max = np.min(x), np.max(x)

for i in range(epochs):
    y_pred = predict(x_train)
    err = error(y_train, y_pred)

    grad_a = gradient(x_train, y_train, y_pred, 1)
    grad_b = gradient(x_train, y_train, y_pred, 2)
    grad_c = gradient(x_train, y_train, y_pred, 3)
    grad_d = gradient(x_train, y_train, y_pred, 4)
    grad_e = gradient(x_train, y_train, y_pred, 5)
    grad_f = gradient(x_train, y_train, y_pred, 6)
    grad_g = gradient(x_train, y_train, y_pred, 7)

    a -= lr * grad_a
    b -= lr * grad_b
    c -= lr * grad_c
    d -= lr * grad_d
    e -= lr * grad_e
    f -= lr * grad_f
    g -= lr * grad_g

    errors.append(err)

    if i % 50000 == 0:
        print(f"error: {err} \ta: {a} \tb: {b} \tc: {c} \td: {d} \te: {e} \tf: {f} \tg: {g}")
        plt.plot(errors)
        plt.savefig('error_vs_epochs.png')
        plt.close()

        plt.scatter(x_train, y_train, color='red')
        x_range = np.linspace(x_min, x_max, 200)
        plt.plot(x_range, predict(x_range), color='green')
        plt.savefig('Salary_Data_train.png')
        plt.close()

# Final plots
plt.plot(errors)
plt.savefig('error_vs_epochs.png')
plt.close()

plt.scatter(x_test, y_test, color='blue')
x_range = np.linspace(x_min, x_max, 200)
plt.plot(x_range, predict(x_range), color='green')
plt.savefig('Salary_Data_test_pred.png')
plt.close()

# Final test error
test_err = error(y_test, predict(x_test))
print(f"Final test error: {test_err}")
