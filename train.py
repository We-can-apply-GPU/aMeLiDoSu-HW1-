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

#Predefined const
sizes = [39,128,48]
EPOCH_MAX = 100
BATCH_SIZE = 2 

def main():
    dnn = Network()
    if sys.argv[1] == "random":
        dnn.initialize(sizes = sizes)
    else:
        dnn.initialize(parsPath = "model/" + sys.argv[1])

    #training stage
    for i in range(EPOCH_MAX):
        print("{0}/{1}".format(i, EPOCH_MAX))
        dataset = infile("data/mfcc/trainToy.ark", "data/label/trainToy.lab")
        batchs = miniBatch(BATCH_SIZE, dataset)

        #print(len(batchs))
        for batch in batchs:
            labels=[batch[i][1] for i in range(BATCH_SIZE)]
            dnn.setLabel(labels)
            dnn.train(batch)
        #dnn.reportErrorrate()

        if len(sys.argv) > 2:
            dnn.saveModel("model/" + str(sys.argv[2]))
        else:
            import time
            dnn.saveModel("model/" + str(time.time()))

if __name__ == "__main__":
    main()
