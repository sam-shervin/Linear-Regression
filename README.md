
# Regression Algorithms from Scratch

This repository contains Python implementations of basic regression algorithms developed from scratch, without using machine learning libraries like `scikit-learn`. These implementations are designed for educational purposes and demonstrate the core concepts behind regression techniques.

## Contents

### 1. Linear Regression using Mean Squared Error (MSE)

* Implements gradient descent to minimize the MSE loss.
* Learns the optimal parameters for a linear model.
* Includes visualization of training data, fitted line, and error over epochs.

### 2. Linear Regression using R² (Coefficient of Determination)

* Implements gradient ascent to maximize the R² score instead of minimizing error.
* Demonstrates how model performance can be directly optimized for explanatory power.
* Suitable for understanding the relationship between variance explained and parameter optimization.

### 3. Polynomial Regression using MSE

* Extends linear regression to fit polynomial relationships (degree-6 supported).
* Uses manually coded gradients for each polynomial term.
* Capable of fitting non-linear data with high accuracy depending on degree and data complexity.

## Features

* No use of external ML libraries.
* Manual gradient computation and optimization logic.
* Randomized train-test split to ensure unbiased evaluation.
* Visualization of:

  * Data distribution (train/test)
  * Model predictions
  * Error progression over time

## Requirements

* Python 3.6+
* Required libraries:

  * `numpy`
  * `matplotlib`
  * `pandas`

Install dependencies:

```bash
pip install numpy matplotlib pandas
```

## How to Run

Each script is self-contained. You can run them directly using:

```bash
python <script_name>.py
```

Ensure that your dataset file (`Salary_Data.csv`) is placed in the same directory before running the scripts.

## Dataset

The models use a CSV dataset with two columns: `YearsExperience` and `Salary`. Normalize salary values to maintain numerical stability in training.

## License

This project is open source and available under the [MIT License](LICENSE).

