import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g
b,c,d,e,f,g = 0,0,0,0,0,0
def predict(x):
    global b,c,d,e,f,g
    return b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g


def error(y, y_pred):
    return np.sum((y_pred-y)**2)/len(y)


def gradient(x, y, y_pred, num):
    if num == 2:
        return 2*np.sum((y_pred-y)*x**5)/len(y)
    elif num == 3:
        return 2*np.sum((y_pred-y)*x**4)/len(y)
    elif num == 4:
        return 2*np.sum((y_pred-y)*x**3)/len(y)
    elif num == 5:
        return 2*np.sum((y_pred-y)*x**2)/len(y)
    elif num == 6:
        return 2*np.sum((y_pred-y)*x)/len(y)
    else:
        return 2*np.sum((y_pred-y))/len(y)
    


lr = (1e-10+1e-9)/4 #this learning rate works well for this error function
epochs = 5000000

# Importing the dataset
dataset = pd.read_csv("./Polynomial Regression lol/Salary_Data.csv")
#iloc is used to locate the data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#normalize y to be between 0 and 1
y = y/np.max(y)
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
errors = []
#I want the updation in a for loop run for that many epochs
print(f" b: {b} \t c: {c} \t d: {d} \t e: {e} \t f: {f} \t g: {g}")
for i in range(epochs):
    #calculate the predicted value of y
    y_pred = predict(x_train)
    
    #calculate the error
    err = error(y_train, y_pred)
    
    #calculate the gradient
    grad_b = gradient(x_train, y_train, y_pred, 2)
    grad_c = gradient(x_train, y_train, y_pred, 3)
    grad_d = gradient(x_train, y_train, y_pred, 4)
    grad_e = gradient(x_train, y_train, y_pred, 5)
    grad_f = gradient(x_train, y_train, y_pred, 6)
    grad_g = gradient(x_train, y_train, y_pred, 7)

    #update the value of m and c
    b = b - lr*grad_b
    c = c - lr*grad_c
    d = d - lr*grad_d
    e = e - lr*grad_e
    f = f - lr*grad_f
    g = g - lr*grad_g
    
    errors.append(err)
    
    #print the error for every 200th epoch
    if i%50000==0:
        print(f"error: {err} \t b: {b} \t c: {c} \t d: {d} \t e: {e} \t f: {f} \t g: {g}")
        plt.plot(errors)
        plt.savefig('error_vs_epochs.png')
        plt.close()
    if i%50000 == 0:
        plt.scatter(x_train, y_train, color = 'red')
        plt.plot(range(-5, 11), predict(np.array(range(-5, 11))), color= 'green')
        plt.savefig('Salary_Data_train.png')
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
print(f"error in test: {err} \t b: {b} \t c: {c} \t d: {d} \t e: {e} \t f: {f} \t g: {g}")
#R^2 closer to 1 means better fit
#hence error vs. epochs should increase as the error function is R^2

