import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
from matplotlib import cm
from sympy import *
import time

NITER = 100

def main():
    testdata = read_file('data/data_small_nonsep_test.csv')
    traindata = read_file('data/data_small_nonsep_train.csv')

    #TRAIN DATA
    xtrain = np.zeros((traindata.shape[0], 2))
    ytrain = np.zeros(traindata.shape[0])

    for i in range(traindata.shape[0]):
        xtrain[i][0] = traindata[i][0]
        xtrain[i][1] = traindata[i][1]

        ytrain[i] = traindata[i][2]

    #TEST DATA
    xtest = np.zeros((testdata.shape[0], 2))
    ytest = np.zeros(testdata.shape[0])

    for i in range(testdata.shape[0]):
        xtest[i][0] = testdata[i][0]
        xtest[i][1] = testdata[i][1]

        ytest[i] = testdata[i][2]

    ## TASK II.1.3: Experiment with training time and error
    print("EXPERIMENTS WITH TRAINING TIME AND ERROR")

    print("Stochastic gradient descent method:")
    start_time_stochast = time.time()
    w, _ = train_and_plot(xtrain,ytrain,xtest,ytest,stochast_train_w)
    total_time_stochast = time.time()-start_time_stochast
    print("Total training time for stochastic gradient descent: ", total_time_stochast)

    print("Batch gradient descent method: ")
    start_time_batch = time.time()
    w, _ = train_and_plot(xtrain, ytrain, xtest, ytest, batch_train_w)
    total_time_batch = time.time()-start_time_batch
    print("Total training time for batch gradient descent: ", total_time_batch)

    ## MAKE SCATTER PLOT OF TRAIN DATA
    print("MAKE SCATTER PLOT OF TRAIN DATA")
    plt.figure()
    fig, ax = plt.subplots()
    ax = scatter_plot(xtrain, ytrain, ax, ['r', 'b'])
         #Train network
    w = stochast_train_w(xtrain, ytrain)
    y_est = []
    for i in range(len(ytest)):
        y_est.append(classify(w,xtest[i]))
        ax = scatter_plot(xtest, ytest, ax, ['g', 'y'])

    plt.show()


    ## EXPERIMENTING WITH DIFFERENT NUMBER OF ITERATIONS
    print("EXPERIMENTIGN WITH DIFFERENT NUMBER OF ITERATIONS")
    niter = [10, 20, 50, 100, 200, 500, 1000, 2000]
    error = []
    train_time = []

    for i in range(len(niter)):
        print("Training with number of iterations = ", niter[i])
        start_time = time.time()
        _, e = train_and_plot(xtrain, ytrain, xtest, ytest, stochast_train_w, niter = niter[i])
        end_time = time.time()
        error.append(e)
        train_time.append(end_time-start_time)
        print("Error = ", e)
        print("Total time = ", end_time-start_time)

    plt.figure()
    plt.plot(niter, error, c = 'r')
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.title("Error for different number of iterations")
    plt.show()
    plt.figure()
    plt.plot(niter, train_time, c = 'b')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total training time [s]")
    plt.title("Total training time for different number of iterations")
    plt.show()





def scatter_plot(xtrain, ytrain, ax, col):
    for i in range(xtrain.shape[0]):
        if ytrain[i] == 0: #class 0, red color
            color = col[0]
        else:
            color = col[1] #class 1, blue color

        ax.scatter(xtrain[i][0],xtrain[i][1], c=color)

    return ax



def read_file(filename):
    data = np.genfromtxt(filename, delimiter='\t', dtype=float)
    return data

def logistic(w,x):
    return 1.0/(1.0+np.exp(-np.inner(w,x)))

def loss(w, x, y):
    return 0.5*(logistic(w,x)-y)**2

def d_loss(w,x,i,y):
    return (logistic(w,x)-y)*logistic(w,x)**2*x[i]*np.exp(-np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic(w,x)<0.5) else 1

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=NITER):
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
        update_grad = 0
        for i in range(dim):
            update_grad = update_grad + d_loss(w, x, i,y) #something needs to be done here
            w[i] = w[i] + learn_rate*update_grad ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=NITER):
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

        if it % 50 == 0:
            print("Iter = ", it)
    return w


def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter = NITER):
    #plt.figure()
    #train data
    #data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    #ax=data.plot(kind='scatter',x='x',y='y',c='lab')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    #data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    #data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm)
    print ("error =",np.mean(error))
    return w, np.mean(error)

main()