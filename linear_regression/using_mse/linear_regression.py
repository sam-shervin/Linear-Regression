import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def predict(x):
    global m, c
    return m * x + c


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def r2_score(y, y_pred):
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def gradient(x, y, y_pred, m_yes=True):
    if m_yes:
        return -2 * np.mean(x * (y - y_pred))
    return -2 * np.mean(y - y_pred)


lr = 0.01
epochs = 2000

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

plt.scatter(x, y)
plt.title('Raw Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('Salary_Data.png')
plt.close()

split_idx = int(len(x) * 2 / 3)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

plt.scatter(x_train, y_train, color='red')
plt.title('Training Data')
plt.savefig('Salary_Data_train.png')
plt.close()

plt.scatter(x_test, y_test, color='blue')
plt.title('Test Data')
plt.savefig('Salary_Data_test.png')
plt.close()

m = 0
c = np.mean(y_train)

mse_train_list = []
r2_train_list = []
mse_test_list = []
r2_test_list = []

for i in range(epochs):
    y_pred_train = predict(x_train)
    y_pred_test = predict(x_test)

    train_mse = mse(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)

    test_mse = mse(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    grad_m = gradient(x_train, y_train, y_pred_train, True)
    grad_c = gradient(x_train, y_train, y_pred_train, False)

    m -= lr * grad_m
    c -= lr * grad_c

    mse_train_list.append(train_mse)
    r2_train_list.append(train_r2)
    mse_test_list.append(test_mse)
    r2_test_list.append(test_r2)

    if i % 200 == 0:
        print(f"Epoch {i}: Train MSE = {train_mse:.4f}, Train R² = {train_r2:.4f}, "
              f"Test MSE = {test_mse:.4f}, Test R² = {test_r2:.4f}, m = {m:.4f}, c = {c:.4f}")

        plt.scatter(x_train, y_train, color='red')
        plt.plot(x_train, y_pred_train, color='green')
        plt.title(f'Train Fit at Epoch {i}')
        plt.savefig(f'Salary_Data_train_epoch_{i}.png')
        plt.close()

plt.plot(range(epochs), mse_train_list, label='Train MSE')
plt.plot(range(epochs), mse_test_list, label='Test MSE')
plt.title('MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('mse_vs_epochs.png')
plt.close()

plt.plot(range(epochs), r2_train_list, label='Train R²')
plt.plot(range(epochs), r2_test_list, label='Test R²')
plt.title('R² over Epochs')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.legend()
plt.savefig('r2_vs_epochs.png')
plt.close()

print(f"Final Train MSE = {mse_train_list[-1]:.4f}, Train R² = {r2_train_list[-1]:.4f}")
print(f"Final Test MSE = {mse_test_list[-1]:.4f}, Test R² = {r2_test_list[-1]:.4f}")

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, predict(x_test), color='green')
plt.title('Test Prediction')
plt.savefig('Salary_Data_test_pred.png')
plt.close()
