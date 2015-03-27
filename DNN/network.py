#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: network.py
Description: define the whole dnn,and learning alg 
"""
import numpy as np
from .calculation import *
from .errorFunc import *
#from util.calculation import *

class Layer:
    def __init__(self,numNeurons):
        self._z = np.zeros((numNeurons,1))
        self._a = np.zeros((numNeurons,1))

class Network:
    def __init__(self,sizes):
        self._numLayers = len(sizes)
        self._sizes = sizes
            
        self._gradW = [np.zeros((j,i)) for (i, j) in zip(self._sizes[:-1],self._sizes[1:])]
        self._gradB = [np.zeros((b,1)) for b in self._sizes[1:]]
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
            self._biases =[np.zeros((b,1))\
                    for b in self._sizes[1:]]
        else:
            #self.loadModel(parsPath)
            pass
    def setLabel(self,labels):
        self._labels.extend(labels)
######################
    def train(self,batch):
        for data in batch:
            self.forward(data[0])
            #dnn.errorFunc()
            self.backpro() #update gradW and gradB
        self.update()
###################
    def forward(self,inData):
        #z is input  vector of neuron layer
        #a is output vector of neuron layer,a=activate(z)

        #inData must be a np.array
        self._layers[0]._z = inData

        for b,w,i in zip(self._biases,self._weights, \
                         range(len(self._biases))):
            self._layers[i]._a = self.activate(self._layers[i]._z)
            self._layers[i+1]._z = np.dot(w,self._layers[i]._a)+b
        self._layers[-1]._a = activate(self._layers[-1]._z)
        #dot can used on matrix product
        return self._layers[-1]._a
#############################

    def backpro(self):
        #This function will use backpropograte
        #to calculate partial C^r over partial layer input
        #then , multiplicate layer output with it ->gradient
        #and store gradient in _gradW and _gradB
       
        delta = self.activatePrime(self._layers[-1]._z)* \
            errFuncPrime(self._layers[-1]._a,self._labels)
        print("delta is {}".format(delta))
        #delta^{L}
        
        #delta is N_L   dim
        #   _a is N_{L-1} dim
        #the weight matrix is N_L x N_{L-1}
        #outer(a,b) = a b^T
        
        self._gradW[-1] += np.outer(delta,self._layers[-2]._a)
        self._gradB[-1] += delta  #delta * 1<-- partial z over b
        
        for l in range(2,self._numLayers):
            delta = activatePrime(self._layers[-l]._z)*  \
            np.dot(np.transpose(self._weights[-l+1],delta))
            self._gradW[-l]+=np.outer(delta,self._layers[-l-1]._a)
            self._gradB[-l]+= delta 

######################
    def update(self,eta):
        for w,b,i in zip(self._weights,self._biases,
                range(len(self._numLayers)-1)):
            w = np.subtract(w,_gradW[i]* eta / len(batch))
            b = np.subtract(b,_gradB[i]* eta / len(batch))
            _gradW[i] = np.zeros(_gradW.shape)
            _gradB[i] = np.zeros(_gradB.shape)


    def predict(self,inData):
        conseq = self.forward(inData)

        #Mapping to 39 phome
        pass


    def loadModel(self,parsPath):
        f = open(parsPath, "r")
        import json
        self._sizes = json.loads(f.readline())
        self._gradW = []
        self._gradB = []
        for tmp in json.loads(f.readline()):
            self._gradW.append(np.asarray(tmp))
        for tmp in json.loads(f.readline()):
            self._gradB.append(np.asarray(tmp))


    def saveModel(self, savePath):
        f = open(savePath, "w")

        import json
        f.write(json.dumps(self._sizes))
        f.write("\n")
        f.write(json.dumps([tmp.tolist() for tmp in self._gradW]))
        f.write("\n")
        f.write(json.dumps([tmp.tolist() for tmp in self._gradB]))
        f.write("\n")
        f.close()

#different activate functions

    def activate(self,x):  #x is a vector
        return sigmoidVec(x)

    def activatePrime(self,x):
        return sigmoidPrimeVec(x)
    
