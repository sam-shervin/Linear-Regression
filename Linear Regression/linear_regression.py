import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#make a function that calculates the value of y from the given x
def predict(x):
    global m, c
    return m*x + c

#Here I have used R^2 as the error function
def error(y, y_pred):
    return 1 - (np.sum((y - y_pred)**2)/(np.sum((y - np.mean(y))**2)))

#since the error function in R^2, I want to maximise it. hence the gradient function is -(derivative of error function)
def gradient(x, y, y_pred, m_yes=True):
    if m_yes:
        return np.sum(2*x*(y_pred - y))/(np.sum((y - np.mean(y))**2))
    return np.sum(-2*(y-y_pred))/(np.sum((y-np.mean(y))**2))


lr = 10000000 #this learning rate works well for this error function
epochs = 3000

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#iloc is used to locate the data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

plt.scatter(x,y)
plt.savefig('Salary_Data.png')
plt.close()

#splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
print(np.mean(y_train))

#convert y_train shape from (20,1) to (20,)
x_train = x_train.reshape(20,)
x_test = x_test.reshape(10,)
#plot the training set and test set and save it as png
plt.scatter(x_train, y_train, color = 'red')
plt.savefig('Salary_Data_train.png')
plt.close()
plt.scatter(x_test, y_test, color = 'blue')
plt.savefig('Salary_Data_test.png')
plt.close()

#y = mx + c
#give a random value for m and c
m = 0
c = np.mean(y_train)
errors = []
#I want the updation in a for loop run for that many epochs
for i in range(epochs):
    #calculate the predicted value of y
    y_pred = predict(x_train)
    
    #calculate the error
    e = error(y_train, y_pred)
    
    #calculate the gradient
    grad_m = gradient(x_train, y_train, y_pred)
    grad_c = gradient(x_train, y_train, y_pred, False)
    #update the value of m and c
    m = m - lr*grad_m
    c = c - lr*grad_c
    
    errors.append(e)
    
    #print the error for every 100th epoch
    if i%200 == 0:
        print(f"R^2: {e} \t m: {m} \t c: {c}")
        plt.scatter(x_train, y_train, color = 'red')
        plt.plot(x_train, y_pred, color= 'green')
        plt.savefig('Salary_Data_train_%d.png' %i)
        plt.close()

plt.plot(range(epochs), errors)
plt.savefig('error_vs_epochs.png')
plt.close()

#plot the y_pred and y_test
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_train, predict(x_train), color = 'green')
plt.savefig('Salary_Data_test_pred.png')
plt.close()

#calculate error for test set
e = error(y_test, predict(x_test))
print(f"R^2 for test set: {e} \t m: {m} \t c: {c}")
#R^2 closer to 1 means better fit
#hence error vs. epochs should increase as the error function is R^2

