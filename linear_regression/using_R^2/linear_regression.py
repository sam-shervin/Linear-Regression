import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Predict function
def predict(x):
    return m * x + c

# Mean Squared Error loss
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# R^2 score as evaluation metric
def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Gradients of MSE
def gradient(x, y, y_pred, wrt_m=True):
    n = len(y)
    if wrt_m:
        return (2 / n) * np.sum(x * (y_pred - y))
    return (2 / n) * np.sum(y_pred - y)

# Hyperparameters
lr = 0.01
epochs = 3000

# Load and prepare data
df = pd.read_csv('Salary_Data.csv')
x = df.iloc[:, 0].values

y = df.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

# Initialize parameters
m = 0.0
c = np.mean(y_train)

history = {'epoch': [], 'mse': [], 'r2': []}

# Training loop
for epoch in range(1, epochs + 1):
    y_pred = predict(x_train)

    # Compute metrics
    loss = mse(y_train, y_pred)
    score = r2_score(y_train, y_pred)

    # Compute gradients
    grad_m = gradient(x_train, y_train, y_pred, wrt_m=True)
    grad_c = gradient(x_train, y_train, y_pred, wrt_m=False)

    # Update params
    m -= lr * grad_m
    c -= lr * grad_c

    # Record history
    history['epoch'].append(epoch)
    history['mse'].append(loss)
    history['r2'].append(score)

    # Print and save plots every 10 epochs up to 100
    if epoch <= 100 and epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | MSE: {loss:.4f} | R²: {score:.4f} | m: {m:.4f} | c: {c:.4f}")
        plt.figure()
        plt.scatter(x_train, y_train, color='red', label='Data')
        plt.plot(x_train, predict(x_train), color='green', label='Fit')
        plt.title(f'Train Fit at Epoch {epoch}')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.legend()
        plt.savefig(f'train_epoch_{epoch}.png')
        plt.close()

# Final training metrics
final_loss = mse(y_train, predict(x_train))
final_r2 = r2_score(y_train, predict(x_train))
print(f"Final Training | MSE: {final_loss:.4f} | R²: {final_r2:.4f} | m: {m:.4f} | c: {c:.4f}")

# Testing evaluation
y_test_pred = predict(x_test)
test_loss = mse(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test Evaluation | MSE: {test_loss:.4f} | R²: {test_r2:.4f}")

# Save test plot
plt.figure()
plt.scatter(x_test, y_test, color='blue', label='Test Data')
plt.plot(x_train, predict(x_train), color='green', label='Model')
plt.title('Test Set Predictions')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.savefig('test_predictions.png')
plt.close()

# Plot training curves
plt.figure()
plt.plot(history['epoch'], history['mse'])
plt.title('MSE vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig('mse_vs_epochs.png')
plt.close()

plt.figure()
plt.plot(history['epoch'], history['r2'])
plt.title('R² vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.savefig('r2_vs_epochs.png')
plt.close()
