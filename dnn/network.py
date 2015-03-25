#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: network.py
Description: define the whole dnn,and learning alg 
"""
import numpy as np
import layer
from util import *


class Network:
    def __init__(self,sizes):
        self._numLayers = len(sizes)
        self._sizes = sizes
            
        self._gradW= [np.zeros(j,i) \
                    for (i,j) in zip(sizes[:-1],[sizes[1:]])]
        self._gradB = [np.zeros(b,1)\
                    for b in sizes[1:]]
        self._layers = []

        for size in sizes:
            layer = Layer(size)
            self._layers.append(layer)
######################
    def initialize(self,parsPath=""):
        if(parsPath ==""):
            self._weights = [np.random.randn(j,i)\
                    for(i,j) in zip(sizes[:-1],sizes[1:])]
            self._biases =[np.random.randn(b,1)\
                    for b in sizes[1:]]
        else:
            #self.loadModel(parsPath)
            pass
######################
    def train(self,batch):
        for data in batch:
            dnn.forward(data)
            dnn.errorFunc()
            dnn.backpro()
            dnn.update()
###################
    def forward(self,inData):
        #z is input  vector of neuron layer
        #a is output vector of neuron layer,a=activate(z)

        #inData must be a np.array
        self._layers[0]._z = inData
        for b,w,z,a in zip(self._biases,self._weights,
                         self._layers[1:],self._layers[:-1]):
            z = activate(np.dot(w,a) + b)
            #dot can used on matrix product
            #the last layer input z is the output of dnn
        return (z[-1]/np.linalg.norm(z[-1]))#normalize it!
#############################

    def backpro(self):
        deltaW[-1] = 2 * \
            np.dot(sigmoidprime(self._layers[-1]._z))

        for l in xrange(2,self._numLayers-1):
            self._gradW[-l] = np.dot\
                    (sigmoidprime(self._layers[-l]._z),\
                    np.dot(np.transpose(self.weights[-l+1]),\
                           deltaW[-l+1]))              
            self._gradB[-l] = np.dot\
                    (sigmoidprime(self._layers[-l]._z),\
                    np.dot(np.transpose(self.weights[-l+1]),\
                        deltaB[-l+1]))              
######################

    def update(self,eta):
        for w,b,i in zip(self._weights,self._biases,
                xrange(len(self._numLayers)-1)):
            w = np.subtract(w,gradW[i]* eta)
            b = np.subtract(b,gradB[i]* eta)


    def predict(self,inData):
        conseq = self.forward(inData)

        #Mapping to 39 phome
        pass


    def loadModel(self):
        pass

    def saveModel(self):
        pass
