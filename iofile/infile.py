#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: infile.py
Description: helper function : loadData and create minibatch
"""
import numpy as np
import random
from math import ceil

def infile(ark = "../mfcc/trainToy.ark", lab = "../label/trainToy.lab"):
    dic = {}
    dataset=[]

    data = open(ark)
    labl = open(lab)
    for line in data:
        s = line.rstrip().split(" ")
        dic[s[0]] = np.array([float(i) for i in s[1:]])
    for line in labl:
        s = line.rstrip().split(",")
        if s[0] in dic:  #just to handle error input
            dataset.append(tuple((dic[s[0]],s[1])))
    #test section
    #for i in dic.keys():
        #print("{} , ".format(i))
    #for i in range(len(dataset)):
        #print("{}:{}".format(i,dataset[i]))
    return dataset

def miniBatch(size,dataset):
    #size define the size per batch
    
    random.seed()
    random.shuffle(dataset)
    batchs = []

    numBatchs = ceil(len(dataset) / size)
    compBatch = size - (len(dataset)%size) #for last batch

    for cnt in range(numBatchs):
        batch = dataset[size*cnt:size*(cnt+1)]
        batchs.append(batch)

    if(compBatch != size): #last batch isn't full
        for i in range(compBatch):
            batchs[-1].extend(dataset[0:compBatch])
    
    #test section
    for i in range(len(batchs)):
        #print(len(batchs))
        print("batch{} : {}".format(i,batchs[i]))
    return batchs


#test section
dataset = infile()
miniBatch(2,dataset)
