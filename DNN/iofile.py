#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: infile.py
Description: helper function : loadData and create minibatch
"""
import numpy as np
import random
from math import ceil

def arkIn(ark):
    ans = []
    data = open(ark)
    for line in data:
        s = line.rstrip().split(" ")
        for i in range(1, len(s)):
            s[i] = float(s[i])
        ans.append(s)
    data.close()
    return ans 

def infile(ark, lab):
    dataset=[]
    dic = {}
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
    #for i in range(len(batchs)):
        ##print(len(batchs))
    for batch in batchs:
        labels=[batch[i][1] for i in range(len(batch))]
        #for label in labels:
            #print("{}".format(label))
    return batchs

if __name__ == "__main__":
    dataset = infile()
    miniBatch(2, dataset)
