#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File: train.py
Description: 
"""

#import section
import sys
import time
from DNN.network import *
import util
import iofile
import random

#Predefined const
sizes = [39,128,48]
EPOCH_MAX = 1
BATCH_SIZE = 100

def main():
    dnn = Network()
    if sys.argv[1] == "random":
        dnn.initialize(BATCH_SIZE,sizes = sizes)
    else:
        dnn.initialize(BATCH_SIZE,parsPath = "model/" + sys.argv[1])
    
    dataset = iofile.infile("data/mfcc/train.ark", "data/label/train.lab")
    random.seed()
    random.shuffle(dataset)
    [trainData, trainLabel] = util.genBatchs(BATCH_SIZE, dataset)
    lenBatch = len(trainData)
    
    for z in range(EPOCH_MAX):
        for i in range(lenBatch):
            print("{0}/{1}".format(i, lenBatch))
            dnn.train(np.transpose(np.array(trainData[i], dtype="float32")), np.transpose(np.array(trainLabel[i], dtype="float32")))

    if len(sys.argv) > 2:
        modelName = "model/" + str(sys.argv[2])
    else:
        import time
        modelName = "model/" + str(time.time())
    dnn.saveModel(modelName)
    
    out = open(modelName, "a")
    trans = []
    util.loadMapList(trans)
    cnt = 0
    allDatas = []
    allLabels = []
    for batch in trainData:
        allDatas.extend(batch) 
    for batch in trainLabel:
        allLabels.extend(batch)

    allOutputs= dnn.forward(np.transpose(np.array(allDatas, dtype="float32")))
    allOutputs = allOutputs.T

    for i in range(len(allOutputs)):
        max_index = util.chooseMax(allOutputs[i])
        if allLabels[i][max_index] == 1:
            cnt += 1
    print("{0}/{1} = {2}".format(cnt, len(allOutputs), float(cnt)/len(allOutputs)))
    out.write("{2}({1})".format(cnt, len(allOutputs), float(cnt)/len(allOutputs)))

if __name__ == "__main__":
    main()
