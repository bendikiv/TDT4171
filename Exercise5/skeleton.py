import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

def main():
    w_size = 12*10+1
    w1 = np.linspace(-6, 6, w_size, endpoint=True)
    w2 = np.linspace(-6, 6, w_size, endpoint=True)
    L_simple = [0]*w_size
    i = 0
    for x in np.nditer(w1):
        L_simple[i] = l_simple([x,x])
        i += 1
    w = [w1, w2]

    plt.plot(w1, L_simple)
    plt.show()

def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1

def l_simple(w):
    L = ((logistic_wx(w, [1,0])-1)*np.exp(2)+(logistic_wx(w, [0,1]))*np.exp(2)+(logistic_wx(w, [1,1])-1)*np.exp(2))**2
    print(L)
    return L

#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            update_grad = 1 ### something needs to be done here
            w[i] = w[i] + learn_rate ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            update_grad=0.0
            for n in xrange(num_n):
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
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
    #print "error=",np.mean(error)
    return w

main()