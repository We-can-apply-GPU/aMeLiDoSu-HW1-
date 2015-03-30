<<<<<<< HEAD
import random
from math import ceil

def loadMap(trans):
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

def miniBatch(size, dataset):
    #size define the size per batch
    
    random.seed()
    random.shuffle(dataset)
    batchs = []

    numBatchs = ceil(len(dataset) / size)

    for cnt in range(numBatchs - 1):
        batch = dataset[size*cnt:size*(cnt+1)]
        batchs.append(batch)

    batchs.append(dataset[size*(numBatchs-1):])
    
    return batchs
