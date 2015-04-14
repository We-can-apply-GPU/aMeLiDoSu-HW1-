#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: network.py
Description: define the whole dnn,and learning alg 
"""
import numpy as np
from .calculation import *
from .errorFunc import *
import theano
import theano.tensor as T

class Layer:
    def __init__(self, w_init, b_init):
        self.w = theano.shared(value=w_init, borrow=True)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX), borrow=True, broadcastable=(False, True))
        self.params = [self.w, self.b]

    def output(self, x):
        return 1.0 / ( 1 + T.exp(-(T.dot(self.w, x) + self.b)))

class Network:
    def __init__(self, w_init, b_init):
        self.layers = []
        for w, b in zip(w_init, b_init):
            self.layers.append(Layer(w, b))
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def forward(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def error(self, x, y):
        x = self.forward(x)
        return -T.sum(y * T.log(x) + (1 - y) * T.log(1 - x)) / x.shape[1]

    def update(self, cost, learning_rate, decay, momentum):
        updates = []
        for param in self.params:
            param_update = theano.shared(param.get_value()*0., broadcastable = param.broadcastable)
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update + (1 - momentum) * T.grad(cost, param)))
        updates.append((learning_rate, learning_rate * decay))
        return updates
    
def loadModel(parsPath):
    f = open(parsPath, "r")
    import json
    w_init = []
    b_init = []
    for tmp in json.loads(f.readline()):
        w_init += [np.asarray(tmp, dtype=theano.config.floatX)]
    for tmp in json.loads(f.readline()):
        b_init += [np.asarray(tmp, dtype=theano.config.floatX)]
    f.close()
    return [w_init, b_init]

def saveModel(savePath, w, b):
    f = open(savePath, "w")
    import json
    f.write(json.dumps(w))
    f.write("\n")
    f.write(json.dumps(b))
    f.write("\n")
