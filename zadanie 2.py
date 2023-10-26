import numpy as np
import pandas as pd
import os
from pandas import read_csv
import matplotlib.pyplot as plt

print(os.listdir("D:\ProjektyProgramowanie\SystemyWspomaganiaDecyzji_Lab1"))

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv('D:\ProjektyProgramowanie\SystemyWspomaganiaDecyzji_Lab1\housing.csv', header=None, delimiter=r"\s+", names=column_names)

# Define the compute_cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost

    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

# Define the compute_gradient function
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# Define the gradient_descent function for training
def gradient_descent(x, y, w, b, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        cost = compute_cost(x, y, w, b)
        print(f"Epoch {epoch + 1}: Cost = {cost}")

    return w, b

# Load the dataset
x_train = np.array(data['AGE'])
y_train = np.array(data['DIS'])

# Define initial values for parameters
initial_w = 2
initial_b = 1

# Set learning rate and number of epochs
learning_rate = 0.00001
num_epochs = 200

# Train the model using gradient descent
final_w, final_b = gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate, num_epochs)
print(f"Ostateczne w: {final_w}, b: {final_b}")

# Calculate the correlation coefficient
correlation = np.corrcoef(x_train, y_train)[0, 1]
print(f"Współczynnik korelacji wynosi: {correlation}")

# Generate and display the scatter plot
plt.scatter(x_train, y_train, label='Data points')
plt.plot(x_train, final_w * x_train + final_b, color='red', label='Regression line')
plt.xlabel('AGE')
plt.ylabel('DIS')
plt.title('Linear Regression')
plt.legend()
plt.show()