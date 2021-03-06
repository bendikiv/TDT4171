import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

def main():
    w1 = np.arange(-6,6,0.1)
    w2 = np.arange(-6,6,0.1)
    w_size = w1.shape[0]

    L_simple = np.zeros((w1.shape[0], w2.shape[0]))
    X, Y = np.meshgrid(w1,w2)
    Z = np.zeros(X.shape)

    L_min = 1000;
    w1_min = 0;
    w2_min = 0;

    for i in range(w1.shape[0]):
        for j in range(w2.shape[0]):
            Z[i,j] = l_simple([w1[i],w2[j]]) #calculate L_simple for every comb. of w1 and w2

            if Z[i,j] < L_min:  #finding the best weights
                L_min = Z[i,j]
                w1_min = w1[i]
                w2_min = w1[j]

    print(L_min, w1_min, w2_min)
    print(np.argmin(Z))

    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.show()

    learning_rate = [0.0001, 0.01, 0.1, 1, 10, 100]
    w_n = [0]*len(learning_rate)
    l_simple_n = [0]*len(learning_rate)

    for i in range(len(learning_rate)):
        w_n[i] = gradient_descent(2, learning_rate[i]) #dimension of w=2, w1 and w2
        l_simple_n[i] = l_simple(w_n[i])
        #print(w_n[i],l_simple_n[i])

    plt.plot(learning_rate,l_simple_n)
    plt.show()

def gradient_descent(dim, learning_rate, niter=1000):
    w = np.random.rand(dim)

    for i in range(2):
        for n in range(niter):
            w[i] = w[i] - learning_rate*l_simple_der(w)[i]

    return w

def log_der(w, x):
    x1 = x[0]
    x2 = x[1]
    dw1 = (x1*np.exp(np.inner(w,x)))/((1+np.exp(np.inner(w,x)))**2)
    dw2 = (x2*np.exp(np.inner(w,x)))/((1+np.exp(np.inner(w,x)))**2)

    return [dw1, dw2]

def l_simple_der(w):
    dl_simple_w1 = 2*(logistic_wx(w,[1,0])-1)*log_der(w,[1,0])[0] + 2*logistic_wx(w,[0,1])*log_der(w,[0,1])[0] + 2*(logistic_wx(w,[1,1])-1)*log_der(w,[1,1])[0]
    dl_simple_w2 = 2*(logistic_wx(w,[1,0])-1)*log_der(w,[1,0])[1] + 2*logistic_wx(w,[0,1])*log_der(w,[0,1])[1] + 2*(logistic_wx(w,[1,1])-1)*log_der(w,[1,1])[1]

    return [dl_simple_w1, dl_simple_w2]

def l_simple(w):
    L = (logistic_wx(w, [1,0])-1)**2+(logistic_wx(w, [0,1]))**2+(logistic_wx(w, [1,1])-1)**2
    return L

def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1

main()