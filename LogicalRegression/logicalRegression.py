# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1

def sigmoid(x):         #x为一个实数
    return 1/(1+np.exp(-x))

#train_X为输入矩阵，行代表样本，列代表特征，矩阵纬度为样本*特征
#train_Y为标签, 矩阵纬度为样本数*1
#theta代表是参数，1*N列
def costFuntion(train_X,train_Y,theta):

    m = train_X.shape[0]
    J = 1.0/m * (-np.transpose(train_Y).dot(np.log(sigmoid(train_X.dot(np.transpose(theta))))) - (1-np.transpose(train_Y)).dot(np.log(1-sigmoid(train_X.dot(np.transpose(theta))))))
    return J


#train_X为输入矩阵，行代表样本，列代表矩阵
def trainLogicalRegression(train_X,train_Y,iteratorNumbers):
    numSamples,numFeatures = train_X.shape
    cos = []
    theta = np.random.random((1,numFeatures))
    for i in range(iteratorNumbers):
        parmaters = train_X * theta.T
      #  print parmaters
        output = sigmoid(parmaters)
        error = output-train_Y
        theta = theta  - alpha * (1.0/train_X.shape[0])*(error.T * train_X)
        loss = costFuntion(train_X,train_Y,theta)
        cos.append(loss[0,0])
    return theta, cos

def loadData(path):
    train_X = []
    train_Y = []
    file = open(path)
    for line in file.readlines():
        lineWords = line.split()
        train_X.append([1.0, float(lineWords[0]), float(lineWords[1])])
        train_Y.append([float(lineWords[2])])
    temp = np.mat(train_Y)
    train_Y = temp.transpose()
    train_X = np.mat(train_X)
    return train_X,train_Y


def figure(train_X,train_Y,theta,cos):
    fig, ax = plt.subplots(figsize=(12,8))
    for i in range(train_X.shape[0]):
        if int(train_Y[i,0]) == 1:
            ax.plot(train_X[i,1],train_X[i,2],'or')
        elif int(train_Y[i,0]) == 0:
            ax.plot(train_X[i,1],train_X[i,2],'ob')
    fig, ax = plt.subplots(figsize=(12,8))
    x = [i for i in range(1000)]
    x = np.mat(x)
    cos = np.mat(cos)
    plt.scatter(x.T,cos.T,c=u'r')
    ax.plot(x.T,cos.T,'b')
    plt.show()


if __name__ == '__main__':
    path = "data/testSet.txt"
    train_X,train_Y = loadData(path)
    theta,cos = trainLogicalRegression(train_X,train_Y.T,1000)
    print theta
    figure(train_X,train_Y.T,theta,cos)