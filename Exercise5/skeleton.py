import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sympy import *
from mpl_toolkits.mplot3d import Axes3D

def main():
    w_size = 12*10+1
    w1 = np.linspace(-6, 6, w_size, endpoint=True)
    w2 = np.linspace(-6, 6, w_size, endpoint=True)

    L_simple = np.zeros((w_size, w_size))

    L_min = 1000;
    w1_min = 0;
    w2_min = 0;

    for i in range(w_size):
        for j in range(w_size):
            w = [w1[i], w2[j]]
            L_simple[i,j] = l_simple(w)
            if L_simple[i,j] < L_min:
                L_min = L_simple[i,j]
                w1_min = w1[i]
                w2_min = w1[j]

    print(L_min, w1_min, w2_min)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(-6, 6, w_size, endpoint=True)
    Y = np.linspace(-6, 6, w_size, endpoint=True)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, L_simple)

    plt.show()

    w_init = [5,-2]
    w_best = gradient_descent(w_init)
    print(w_best, l_simple(w_best))

def gradient_descent(w, learning_rate=100, niter=1000):
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

#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in range(dim):
            update_grad = l_simple_der(w)[i] ### something needs to be done here
            w[i] = w[i] + learn_rate*update_grad ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                update_grad+=(-logistic_wx(w,x_train[n])+y_train[n])# something needs to be done here
            w[i] = w[i] + learn_rate * update_grad/num_n
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
    #print "error=",np.mean(error)
    return w

main()