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
    def __init__(self,numNeurons,batchSize):
        self._z = np.zeros((numNeurons,batchSize)) 
        self._a = np.zeros((numNeurons,batchSize))

class Network:

    def initialize(self, batchSize,parsPath = "", sizes = []):

        self._sizes = sizes
        self._numLayers = len(sizes)
        
        if parsPath == "" :
            self._weights = [np.random.randn(j,i) for(i,j) in zip(self._sizes[:-1], self._sizes[1:])]
            self._biases =[np.zeros(b) for b in self._sizes[1:]]
        else:
            self.loadModel(parsPath)

        self._layers = []
        for size in self._sizes:
            layer = Layer(size,batchSize)
            self._layers.append(layer)

        self._softMaxLayer = Layer(size,batchSize)

        self._prevGradW = [np.zeros(w.shape) for w in self._weights]
        self._prevGradB = [np.zeros(b) for b in self._sizes[1:]]
        self.loss = 0.0
######################
    def train(self,datas,labels):
        #print(len(batch))
        #for momentum usage
        self._gradW = [np.zeros(w.shape) for w in self._weights]
        self._gradB = [np.zeros(b.shape) for b in self._biases]
        self._gradBs = [np.zeros((b,datas.shape[1])) for b in self._sizes[1:]]
        datas = self.preprocess(datas)

        self.forward(datas)
        self.backpro(labels)
        self.update(0.001, 0.9, datas.shape[1])
###################
    def preprocess(self,datas)
        datas = np.transpose(datas)
        for i in range(datas.__len__()):
            datas[i] = (datas[i]-np.mean(datas[i]))/np.std(datas[i])
        return np.transpose(datas)
###################
    def forward(self,inData):
        #z is input  matrix of neuron layer
        #a is output matrix of neuron layer,a=activate(z)
        
        #inData must be a np.array(39 * BATCH_SIZE)
        self._layers[0]._a = inData
        
        #trace layer to layer
        for i in range(len(self._layers) - 1):
            self._layers[i+1]._z = Tdot(self._weights[i], self._layers[i]._a) 
            for column in self._layers[i+1]._z.T:
                column += self._biases[i]
            self._layers[i+1]._a = self.activate(self._layers[i+1]._z)

        self._softMaxLayer._z = self._layers[-1]._a
        self._softMaxLayer._a = softMax(self._softMaxLayer._z)

        return self._softMaxLayer._a
#############################

    def backpro(self,labels):
        #This function will use backpropograte
        #to calculate partial C^r over partial layer input
        #then , multiplicate layer output with it ->gradient
        #and store gradient in _gradW and _gradB
        #print(dataId)
        aPl = self.activatePrime(self._layers[-1]._z)
        #t1 = time.clock()
        #CrP = errFuncPrime(self._layers[-1]._a,labels)
        self.loss = np.sum(errFunc(self._softMaxLayer._a,labels))
        CrP = errFuncPrime(self._softMaxLayer._a,labels)
        #print(time.clock() - t1)
        delta = aPl* CrP 
        #delta^{L}
        
        #delta is N_L*BATCHSIZE 
        #   _a is N_{L-1}*BATCHSIZE 
        #the weight matrix is N_L x N_{L-1}
        #outer(a,b) = a b^T
        #t1 = time.clock()  
        self._gradW[-1] = Tdot(delta,np.transpose(self._layers[-2]._a)) 
        #haven't been tested,the consequence is the sum of each deltaW
        self._gradBs[-1] = delta  #delta * 1<-- partial z over b
        
        for l in range(2,self._numLayers):
            delta = self.activatePrime(self._layers[-l]._z)*  \
            Tdot(np.transpose(self._weights[-l+1]),delta)
            self._gradW[-l]=Tdot(delta,np.transpose(self._layers[-l-1]._a))
            self._gradBs[-l]= delta 
######################
    def update(self,eta,momentum,batch_len):
        
        for i in range(len(self._layers) - 1):
            self._gradW[i] = np.add(self._gradW[i],momentum*self._prevGradW[i])

            for column in self._gradBs[i].T :
                self._gradB[i] += column
            
            self._gradB[i] = np.add(self._gradB[i],momentum*self._prevGradB[i])

        for i in range(len(self._layers)-1):
            self._weights[i] = np.subtract(self._weights[i], self._gradW[i] * eta / batch_len)
            self._biases[i] = np.subtract(self._biases[i],self._gradB[i] * eta / batch_len)

            self._prevGradW = self._gradW
            self._prevGradB = self._gradB


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

    def activate(self,x):  
        return sigmoidVec(x)

    def activatePrime(self,x):
        return sigmoidPrimeVec(x)
    
