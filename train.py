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
EPOCH_MAX = 1000
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
    allDatas = []
    allLabels = []
    for batch in trainData:
        allDatas.extend(batch) 
    for batch in trainLabel:
        allLabels.extend(batch)
    lenBatch = len(trainData)
    print("training start")
    for z in range(1, EPOCH_MAX+1):
        for i in range(lenBatch):
            dnn.train(np.transpose(np.array(trainData[i], dtype="float32")), np.transpose(np.array(trainLabel[i], dtype="float32")))
        print("{0}/{1}".format(z, EPOCH_MAX))

        if z % 10 == 0:
            trans = []
            util.loadMapList(trans)
            cnt = 0

            allOutputs= dnn.forward(np.transpose(np.array(allDatas, dtype="float32")))
            allOutputs = allOutputs.T

            for i in range(len(allOutputs)):
                max_index = util.chooseMax(allOutputs[i])
                if allLabels[i][max_index] == 1:
                    cnt += 1
            dnn.saveModel("tmpModel/" + "{2}x{1}".format(cnt, len(allOutputs), float(cnt)/len(allOutputs)))
            print("{0}/{1} = {2}".format(cnt, len(allOutputs), float(cnt)/len(allOutputs)))

if __name__ == "__main__":
    main()
