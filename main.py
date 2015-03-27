#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: main.py
Description: 
"""

#import section
from DNN.network import *
from DNN.iofile import *

#Predefined const
sizes = [39,48]
EPOCH_MAX = 100
BATCH_SIZE = 50

def main():
    dnn = Network(sizes)
    dnn.initialize() #default is random

    #training stage
    for i in range(EPOCH_MAX):
        dataset = infile("data/mfcc/trainToy.ark", "data/label/trainToy.lab")
        batchs = miniBatch(BATCH_SIZE,dataset)

        for batch in batchs:
            labels=[batch[i][1] for i in range(BATCH_SIZE)]
            dnn.setLabel(labels)
            dnn.train(batch)
        #dnn.reportErrorrate()
    #dnn.saveModel("")


    #predicting
    #dnn.predict("")

if __name__ == "__main__":
    main()
