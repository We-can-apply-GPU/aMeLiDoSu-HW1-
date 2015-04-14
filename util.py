import random
import numpy as np
import theano
from math import ceil

def infile(ark, lab, Max=10000000):
    dataset = []
    arkData = open(ark)
    labData = open(lab)
    cnt = 0
    for aa, ll in zip(arkData, labData):
        a = aa.rstrip().split(" ")
        l = ll.rstrip().split(",")
        dataset.append((a[1:], l[1]))
        cnt += 1
        if cnt == Max:
            break
    return dataset

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
    batchs = [[],[]]
    trainData = []
    trainLabel = []
    cnt = 0

    trans = []
    loadMapList(trans)

    for data in dataset:
        tmp = []
        for i in range(48):
            if trans[i][0] == data[1]:
                tmp.append(1)
            else:
                tmp.append(0)
        trainData.append(data[0])
        trainLabel.append(tmp)
        cnt += 1
        if cnt == size:
            batchs[0].append(np.asarray(trainData, dtype=np.float32))
            batchs[1].append(np.asarray(trainLabel, dtype=np.float32))
            trainData = []
            trainLabel = []
            cnt = 0
    if cnt != 0:
        batchs[0].append(np.asarray(trainData, dtype=np.float32))
        batchs[1].append(np.asarray(trainLabel, dtype=np.float32))
    return batchs
