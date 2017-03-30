import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sympy import *
from mpl_toolkits.mplot3d import Axes3D

def main():
    return

def surface_plot(w_size, L_simple):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(-6, 6, w_size, endpoint=True)
    Y = np.linspace(-6, 6, w_size, endpoint=True)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, L_simple)

    plt.show()

def logistic(w,x):
    return 1.0/(1.0+np.exp(-np.inner(w,x)))

def loss(w, x, y):
    return 0.5*(logistic(w,x)-y)**2

def d_loss(w,x,i,y):
    return (logistic(w,x)-y)*logistic(w,x)**2*x[i]*np.exp(-np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic(w,x)<0.5) else 1

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
            update_grad = update_grad + d_loss(w, x, i,y) #something needs to be done here
            w[i] = w[i] + learn_rate*update_grad ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))

    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    print(x_train)
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                update_grad += d_loss(w,x_train[n], i, y_train[n]) # something needs to be done here
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