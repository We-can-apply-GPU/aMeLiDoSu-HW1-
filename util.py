<<<<<<< HEAD
import random
import numpy as np
from math import ceil

def loadMapList(trans):
    data = open("DNN/48_39.map")
    for line in data:
        s = line.rstrip().split("\t")
        trans.append((s[0], s[1]))

def chooseMax(xxx):
    max_index = 0
    for i in range(len(xxx)):
        if xxx[i] > xxx[max_index]:
            max_index = i
    return max_index

def genBatchs(size, dataset):
    #size define the size per batch
    batchs = [[],[]]
    trainData = []
    trainLabel = []
    cnt = 0

    trans = []
    loadMapList(trans)

    for data in dataset:
        tmp = []
        for i in range(39):
            if trans[i][0] == data[1]:
                tmp.append(1)
            else:
                tmp.append(0)
        trainData.append(data[0])
        trainLabel.append(tmp)
        cnt += 1
        if cnt == size:
            batchs[0].append(trainData)
            batchs[1].append(trainLabel)
            trainData = []
            trainLabel = []
            cnt = 0

    if cnt != 0:
        batchs[0].append(trainData)
        batchs[1].append(trainLabel)

    return batchs
