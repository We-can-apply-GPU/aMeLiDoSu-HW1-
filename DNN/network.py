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
import theano.tensor as T
class Layer:
    def __init__(self,numNeurons):
        self._z = np.zeros((numNeurons,1))
        self._a = np.zeros((numNeurons,1))

class Network:

    def initialize(self, parsPath = "", sizes = []):
        self._sizes = sizes
        self._numLayers = len(sizes)

        if parsPath == "" :
            self._weights = [np.random.randn(j,i)\
                    for(i,j) in zip(self._sizes[:-1], self._sizes[1:])]
            self._biases =[np.zeros(b) for b in self._sizes[1:]]
        else:
            self.loadModel(parsPath)

        self._layers = []

        for size in self._sizes:
            layer = Layer(size)
            self._layers.append(layer)

    def setLabel(self,labels):
        self._labels = labels
######################
    def train(self,batch):
        #print(len(batch))
        self._gradW = [np.zeros((j, i)) for i, j in zip(self._sizes[:-1], self._sizes[1:])]
        self._gradB = [np.zeros(b) for b in self._sizes[1:]]
        for data , dataId in zip (batch,(range(len(batch)))):
            # print(data, dataId)
            self.forward(data[0])
            #dnn.errorFunc()
            self.backpro(dataId) #update gradW and gradB
        self.update(0.01,batch.__len__())
###################
    def forward(self,inData):
        #z is input  vector of neuron layer
        #a is output vector of neuron layer,a=activate(z)

        #inData must be a np.array
        #self._layers[0]._z = inData #not necessary
        self._layers[0]._a = inData

        for i in range(len(self._biases)):
            self._layers[i+1]._z = TMVdot(self._weights[i], self._layers[i]._a) + self._biases[i]
            self._layers[i+1]._a = self.activate(self._layers[i+1]._z)
        #dot can used on matrix product
        return self._layers[-1]._a
#############################

    def backpro(self,dataId):
        #This function will use backpropograte
        #to calculate partial C^r over partial layer input
        #then , multiplicate layer output with it ->gradient
        #and store gradient in _gradW and _gradB
       
        aPl = self.activatePrime(self._layers[-1]._z)
        CrP = errFuncPrime(self._layers[-1]._a,self._labels[dataId])
        delta = aPl* CrP 
        #delta^{L}
        
        #delta is N_L   dim
        #   _a is N_{L-1} dim
        #the weight matrix is N_L x N_{L-1}
        #outer(a,b) = a b^T
        
        self._gradW[-1] += Touter(delta,self._layers[-2]._a)
        self._gradB[-1] += delta  #delta * 1<-- partial z over b
        
        for l in range(2,self._numLayers):
            delta = self.activatePrime(self._layers[-l]._z)*  \
            TMVdot(np.transpose(self._weights[-l+1]),delta)
            self._gradW[-l]+=Touter(delta,self._layers[-l-1]._a)
            self._gradB[-l]+= delta 

######################
    def update(self,eta,batch_len):
        for i in range(len(self._layers)-1):
            # print("w(before) = {0}\nb(before) = {1}\n".format(self._gradW, self._gradB))
            self._weights[i] = np.subtract(self._weights[i], self._gradW[i] * eta / batch_len)
            self._biases[i] = np.subtract(self._biases[i], self._gradB[i] * eta / batch_len)
            # print("w(after) = {0}\nb(after) = {1}\n".format(self._gradW, self._gradB))

    def predict(self,inData):
        conseq = self.forward(inData)

        #Mapping to 39 phome
        pass


    def loadModel(self,parsPath):
        f = open(parsPath, "r")
        import json
        self._sizes = json.loads(f.readline())
        self._numLayers = len(self._sizes)
        self._weights = []
        self._biases = []
        for tmp in json.loads(f.readline()):
            self._weights.append(np.asarray(tmp))
        for tmp in json.loads(f.readline()):
            self._biases.append(np.asarray(tmp))


    def saveModel(self, savePath):
        f = open(savePath, "w")

        import json
        f.write(json.dumps(self._sizes))
        f.write("\n")
        f.write(json.dumps([tmp.tolist() for tmp in self._weights]))
        f.write("\n")
        f.write(json.dumps([tmp.tolist() for tmp in self._biases]))
        f.write("\n")
        f.close()

#different activate functions

    def activate(self,x):  #x is a vector
        return sigmoidVec(x)

    def activatePrime(self,x):
        return sigmoidPrimeVec(x)
    
