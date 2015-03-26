#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: main.py
Description: 
"""

#import section
from dnn import *
from util import *
from iofile import *

#Predefined const
sizes = [39,128,48]
EPOCH_MAX = 100
BATCH_SIZE

def main():
    dnn = network(sizes)
    dnn.initialize()#default is random

    #training stage
    for i in range(EPOCH_MAX):
        dataset = infile()
        batchs = miniBatch(BATCH_SIZE,dataset)

        for batch in batchs:
            labels=[batch[i][1] for i in range(len(batch))]
            dnn.setLabel(labels)
            dnn.train(batch)

    #predicting

if __name__ == "main":
    main()
