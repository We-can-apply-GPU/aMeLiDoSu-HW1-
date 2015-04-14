#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File: train.py
Description: 
"""

#import section
import sys
from DNN.network import *
import util
import random
import theano
import theano.tensor as T

sizes = [39, 64, 64, 64, 48]
EPOCH_MAX = 1000000
BATCH_SIZE = 100
lr = 0.01
momentum = 0.9
decay = 1

def main():
    w_init = []
    b_init = []
    if sys.argv[1] == "random":
        for n_input, n_output in zip(sizes[:-1], sizes[1:]):
            w_init.append(np.random.randn(n_output, n_input))
            b_init.append(np.zeros(n_output))
    else:
        [w_init, b_init] = loadModel("model/" + sys.argv[1])
    dnn = Network(w_init, b_init)
    print("Compiling theano")
    inputData = T.matrix("inputData")
    targetData = T.matrix("outputData")
    learning_rate = theano.shared(value=np.float32(lr))
    cost = dnn.error(inputData, targetData)
    train = theano.function([inputData, targetData], inputData, updates=dnn.update(cost, learning_rate, decay, momentum))
    forward = theano.function([inputData], dnn.forward(inputData))
    
    print("Reading data")
    dataset = util.infile("data/mfcc/train.ark", "data/label/train.lab")
    random.seed()
    random.shuffle(dataset)
    print("Processing data")
    [X, Y] = util.genBatchs(BATCH_SIZE, dataset)
    print("Processing all data")
    [allX, allY] = util.genBatchs(10000000, dataset)
    allX = allX[0]
    allY = allY[0]
    print("training")

    for z in range(1, EPOCH_MAX+1):
        tmp = 0
        for x, y in zip(X, Y):
            train(x.T, y.T)
        if z % 10 == 0:
            print(cost.eval({inputData: allX.T, targetData: allY.T}))
            allOutput = forward(allX.T).T
            cnt = 0
            for eachOutput, eachTarget in zip(allOutput, allY):
                test = util.chooseMax(eachOutput)
                if eachTarget[test] == 1:
                    cnt += 1
            print("{0}/{1}".format(cnt, len(allX)))
            w = []
            b = []
            for layer in dnn.layers:
                w += [layer.w.get_value().tolist()]
                b += [layer.b.get_value().tolist()]
            saveModel("tmpModel/"+"{0}x{1}".format(cnt, len(allX)), w, b)

        print("Epoch: {0}".format(z))
        

if __name__ == "__main__":
    main()
