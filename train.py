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
from DNN.iofile import *
import util

#Predefined const
sizes = [39,128,48]
EPOCH_MAX = 50
BATCH_SIZE = 10

def main():
    dnn = Network()
    if sys.argv[1] == "random":
        dnn.initialize(sizes = sizes)
    else:
        dnn.initialize(parsPath = "model/" + sys.argv[1])

    dataset = infile("data/mfcc/trainToy.ark", "data/label/trainToy.lab")
    batchs = miniBatch(BATCH_SIZE, dataset)

    #training stage
    for i in range(EPOCH_MAX):
        print("{0}/{1}".format(i+1, EPOCH_MAX))
        #print(len(batchs))
        for batch in batchs:
            labels=[batch[i][1] for i in range(BATCH_SIZE)]
            dnn.setLabel(labels)
            dnn.train(batch)
        #dnn.reportErrorrate()

    if len(sys.argv) > 2:
        modelName = "model/" + str(sys.argv[2])
    else:
        import time
        modelName = "model/" + str(time.time())
    dnn.saveModel(modelName)
    
    out = open(modelName, "a")
    trans = []
    util.loadMap(trans)
    cnt = 0
    for row in dataset:
        max_index = util.chooseMax(dnn.forward(row[0]))
        print(trans[max_index][1], row[1])
        if trans[max_index][1] == row[1]:
            cnt += 1
    print("{0}/{1} = {2}".format(cnt, len(dataset), float(cnt)/len(dataset)))
    out.write("{0}/{1} = {2}".format(cnt, len(dataset), float(cnt)/len(dataset)))

if __name__ == "__main__":
    main()
