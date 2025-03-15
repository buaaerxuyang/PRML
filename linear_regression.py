import numpy as np
from reading_xlsx import read_xy
from graph import plot_points_and_line

def trainOLSmethod(x,y):
    theta = np.zeros(2)
    n = x.shape[0]
    X = np.column_stack((np.ones(n), x))
    XT = np.transpose(X)
    Y = y.reshape(n,1)
    theta = np.linalg.inv(XT @ X) @ XT @ Y
    print(theta)
    return theta

def trainGDmethod(x,y,theta,alpha=0.0001,iterations=100):
    n = x.shape[0]
    X = np.column_stack((np.ones(n), x))
    theta = theta.reshape(2,1)
    for i in range(iterations):
        Jpartial0 = 1/n * np.sum(X @ theta - y)
        Jpartial1 = 1/n * np.sum((X @ theta - y) * x)
        theta[0][0] = theta[0][0] - alpha * Jpartial0
        theta[1][0] = theta[1][0] - alpha * Jpartial1
    print(theta)
    return theta

def trainNRmethod(x, y, theta=np.ones(2), iterations=10):
    n = x.shape[0]
    X = np.column_stack((np.ones(n), x))
    theta = theta.reshape(2, 1)
    y = y.reshape(n, 1)
    for i in range(iterations):
        grad = X.T @ (X @ theta - y)
        Hessian = X.T @ X
        theta = theta - np.linalg.inv(Hessian) @ grad
    print(theta)
    return theta

def sumerror(x,y,theta):
    n = x.shape[0]
    X = np.column_stack((np.ones(n), x))
    theta = theta.reshape(2, 1)
    error = 1/(2*n) * np.sum((X @ theta - y) ** 2)
    return error
    

if __name__=="__main__":
    train_x,train_y,test_x,test_y = read_xy()
    thetaOLS=trainOLSmethod(train_x,train_y)
    thetaGD=trainGDmethod(train_x,train_y,np.ones(2),0.0001,1000)
    thetaNR=trainNRmethod(train_x,train_y,np.ones(2),10)

    # plot_points_and_line(train_x, train_y, thetaOLS[1][0], thetaOLS[0][0])
    # plot_points_and_line(train_x, train_y, thetaGD[1][0], thetaGD[0][0])
    # plot_points_and_line(train_x, train_y, thetaNR[1][0], thetaNR[0][0])

    test_errorOLS=sumerror(test_x,test_y,thetaOLS)
    test_errorGD=sumerror(test_x,test_y,thetaGD)
    test_errorNR=sumerror(test_x,test_y,thetaNR)

    train_errorOLS=sumerror(train_x,train_y,thetaOLS)
    train_errorGD=sumerror(train_x,train_y,thetaGD)
    train_errorNR=sumerror(train_x,train_y,thetaNR)

    print("OLS method test error: ",test_errorOLS)
    print("GD method test error: ",test_errorGD) 
    print("NR method test error: ",test_errorNR)

    print("OLS method train error: ",train_errorOLS)
    print("GD method train error: ",train_errorGD)
    print("NR method train error: ",train_errorNR)

    # plot_points_and_line(train_x, train_y, 0.1, 0.1)
    

    