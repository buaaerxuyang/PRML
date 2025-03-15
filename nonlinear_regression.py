import numpy as np
import math
from reading_xlsx import read_xy
from graph import plot_points_and_curve

def train(x,y,alpha=0.0001,iterations=100):
    arguments = np.ones(4)
    arguments[0] = 1.0
    arguments[1] = 2.5
    arguments[2] = 0.1
    arguments[3] = -0.2
    n = x.shape[0]
    X = x.reshape(n,1)
    Y = y.reshape(n,1)
    for i in range(iterations):
        delta_vector = arguments[2] * X + arguments[3] + arguments[0] * np.sin(arguments[1]*X) -  Y
        #print(delta_vector.shape)
        Jpartial3 = 1/n * np.sum(delta_vector)
        Jpartial2 = 1/n * np.sum(delta_vector * X)
        Jpartial1 = 1/n * np.sum(delta_vector * arguments[0] * X * np.cos(arguments[1]*X))
        Jpartial0 = 1/n * np.sum(delta_vector * np.sin(arguments[1]*X))
        arguments[0] = arguments[0] - alpha * Jpartial0
        arguments[1] = arguments[1] - alpha * Jpartial1
        arguments[2] = arguments[2] - alpha * Jpartial2
        arguments[3] = arguments[3] - alpha * Jpartial3
        error = 1/(2*n) * np.sum(delta_vector ** 2)
        if i % 100 == 0:
            print(f'iteration {i}:  arguments: {arguments} error: {error}')
    return arguments

def sumerror(x,y,arguments):
    n = x.shape[0]
    X = x.reshape(n,1)
    Y = y.reshape(n,1)
    error = 1/(2*n) * np.sum((arguments[2] * X + arguments[3] + arguments[0] * np.sin(arguments[1]*X) -  Y) ** 2)
    return error

if __name__=="__main__":
    train_x,train_y,test_x,test_y = read_xy()
    auguments=train(train_x,train_y,0.0001,10000)
    print("train error: ", sumerror(train_x,train_y,auguments))
    print("test error: ", sumerror(test_x,test_y,auguments))
    plot_points_and_curve(train_x, train_y, auguments[0], auguments[1], auguments[2], auguments[3])
    plot_points_and_curve(test_x, test_y, auguments[0], auguments[1], auguments[2], auguments[3])