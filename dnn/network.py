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
                    for (i,j)in zip(_sizes[:-1],[_sizes[1:]])]
        self._gradB = [np.zeros(b,1)\
                    for b in _sizes[1:]]
        self._layers = []
        
        for size in _sizes:
            layer = Layer(size)
            self._layers.append(layer)

        self._labels = [] # a phone list
        for
######################
    def initialize(self,parsPath=""):
        if(parsPath ==""):
            self._weights = [np.random.randn(j,i)\
                    for(i,j) in zip(_sizes[:-1],_sizes[1:])]
            self._biases =[np)random.randn(b,1)\
                    for b in _sizes[1:]]
        else:
            #self.loadModel(parsPath)
            pass
    def setLabel(labels):
        self._labels.extend(labels)
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
        
        self._layers[-1]._a = activate(self._layers[-1]._z)
        #dot can used on matrix product
        return self._layers[-1]._a
#############################

    def backpro(self):
        #This function will use backpropograte
        #to calculate partial C^r over partial layer input
        #then , multiplicate layer output with it ->gradient
        #and store gradient in _gradW and _gradB
       
        delta = activatePrime(self._layers[-1]._z)* \
                errFuncPrime(self._layers[-1]._a) #delta^{L}
        
        #delta is N_L   dim
        #   _a is N_{L-1} dim
        #the weight matrix is N_L x N_{L-1}
        #outer(a,b) = a b^T
############I just think this maybe right@@################
        
        _gradW[-1] = np.outer(delta,self._layers[-2]._a)
        _gradB[-1] = delta  #delta * 1<-- partial z over b
        
        for l in range(2,self._numLayers):
            delta = activatePrime(self._layers[-l]._z)*  \
            np.dot(np.transpose(self._weights[-l+1],delta)
############As above ,I just think this maybe right@@#####
            _gradW[-l]=np.outer(delta,self._layers[-l-1]._a)
            _gradB[-l] = delta 

######################
    def update(self,eta):
        for w,b,i in zip(self._weights,self._biases,
                range(len(self._numLayers)-1)):
            w = np.subtract(w,_gradW[i]* eta)
            b = np.subtract(b,_gradB[i]* eta)


    def predict(self,inData):
        conseq = self.forward(inData)

        #Mapping to 39 phome
        pass


    def loadModel(self,parsPath=""):
        pass

    def saveModel(self,savePath=""):
        pass
