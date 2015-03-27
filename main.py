#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: main.py
Description: 
"""

#import section
from dnn.network import *
from iofile.infile import *

#Predefined const
sizes = [39,128,48]
EPOCH_MAX = 100
BATCH_SIZE = 50

def main():
    dnn = network(sizes)
    dnn.initialize()#default is random

    #training stage
    for i in range(EPOCH_MAX):
        dataset = infile()
        batchs = miniBatch(BATCH_SIZE,dataset)

        for batch in batchs:
            labels=[batch[i][1] for i in BATCH_SIZE]
            dnn.setLabel(labels)
            dnn.train(batch)
        #dnn.reportErrorrate()
    #dnn.saveModel("")


    #predicting
    #dnn.predict("")



main()
