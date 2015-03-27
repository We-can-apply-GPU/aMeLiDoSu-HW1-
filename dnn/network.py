#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: network.py
Description: define the whole dnn,and learning alg 
"""
import numpy as np
from layer import *
from util import *


class Network:
    def __init__(self,sizes):
        self._numLayers = len(sizes)
        self._sizes = sizes
            
        self._gradW = [np.zeros((j,i)) for (i, j) in zip(self._sizes[:-1],self._sizes[1:])]
        self._gradB = [np.zeros((b,1))\
                    for b in self._sizes[1:]]
        self._layers = []
        
        for size in self._sizes:
            layer = Layer(size)
            self._layers.append(layer)

        self._labels = [] # a phone list
######################
    def initialize(self,parsPath=""):
        if(parsPath ==""):
            self._weights = [np.random.randn(j,i)\
                    for(i,j) in zip(self._sizes[:-1], self._sizes[1:])]
            self._biases =[np.random.randn(b,1)\
                    for b in self._sizes[1:]]
        else:
            #self.loadModel(parsPath)
            pass
    def setLabel(self,labels):
        self._labels.extend(labels)
######################
    def train(self,batch):
        for data in batch:
            dnn.forward(data)
            #dnn.errorFunc()
            dnn.backpro() #update gradW and gradB
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
            errFuncPrime(self._layers[-1]._a,self._labels)

            #delta^{L}
        
        #delta is N_L   dim
        #   _a is N_{L-1} dim
        #the weight matrix is N_L x N_{L-1}
        #outer(a,b) = a b^T
############I just think this maybe right@@################
        
        self._gradW[-1] += np.outer(delta,self._layers[-2]._a)
        self._gradB[-1] += delta  #delta * 1<-- partial z over b
        
        for l in range(2,self._numLayers):
            delta = activatePrime(self._layers[-l]._z)*  \
            np.dot(np.transpose(self._weights[-l+1],delta))
############As above ,I just think this maybe right@@#####
            self._gradW[-l]+=np.outer(delta,self._layers[-l-1]._a)
            self._gradB[-l]+= delta 

######################
    def update(self,eta):
        for w,b,i in zip(self._weights,self._biases,
                range(len(self._numLayers)-1)):
            w = np.subtract(w,_gradW[i]* eta / len(batch))
            b = np.subtract(b,_gradB[i]* eta / len(batch))


    def predict(self,inData):
        conseq = self.forward(inData)

        #Mapping to 39 phome
        pass


    def loadModel(self,parsPath):
        pass


    def saveModel(self, savePath):
        f = open(savePath, "w")
        import json

        f.write(json.dumps(self._sizes))

        for (i, j) in zip(self._sizes[:-1] ,self._sizes[1:]):
            print(self._gradW[j][i])


    def activate(self,x):
       sigmoidVec(x)
    
    def activatePrime(self,x):
        activatePrimeVec(x)

    def errorFunc(self,x):
        errorFuncVec(x)
    
    def errorFuncPrime(self,x):
        errorFuncPrimeVec(x)

