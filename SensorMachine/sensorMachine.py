# coding=utf-8

import  matplotlib.pyplot as plt
import numpy as np
trainset = []
w = [0,0]
b = 0
learningRate = 0.1
trainloss = []

def calculateDistance(sample):  #输入形式为[[x1,x2],[y]]
    global w,b
    f_distance = 0
    for i in range(len(sample)):
        f_distance += w[i] * float(sample[i])
    f_distance += b

    return f_distance

def loaddata(path):
    for line in open(path):
        lineWords = line.split("\t")
        data = []
        label = []
        trainLine = []
        words = len(lineWords)
        for i in range(words-1):
            data.append(lineWords[i])
        if int(lineWords[words-1]) == 1:
            label.append(1)
        else:
            label.append(-1)
        trainLine.append(data)
        trainLine.append(label)
        trainset.append(trainLine)

if __name__ == '__main__':  #main函数
    path = "data/testSet.txt"
    loaddata(path)
    flag = True
    iterator = 800
    for j in range(iterator):
        loss = 0
        for line in range(len(trainset)):
            sample = trainset[line][0]    #取出[x1,x2]
            lable = trainset[line][1]     #取出标签
            distance = calculateDistance(sample)
            if lable[0] * distance <= 0:        #更新参数
                loss += -lable[0] * distance
                for i in range(len(sample)):
                    w[i] = w[i] + learningRate * lable[0] * float(sample[i])
                b = b + learningRate * lable[0]
        trainloss.append(loss)
    print w,b
    x = np.linspace(-5,5,10)
    plt.figure()
    for x_ in range(len(trainset)):
        if trainset[x_][1][0] == 1:
            plt.scatter(trainset[x_][0][0],trainset[x_][0][1],c='b')
        else:
            plt.scatter(trainset[x_][0][0],trainset[x_][0][1],c='r')
    plt.plot(x,-(w[0]*x+b)/w[1],c='y')
    plt.show()

    trainIter = range(len(trainloss))
    plt.figure()
    plt.scatter(trainIter,trainloss,c=u'r')
    plt.plot(trainIter,trainloss)
    plt.xlabel('Epoch')
    plt.ylabel('trainLoss')
    plt.show()



