#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: network.py
Description: define the whole dnn,and learning alg 
"""
import numpy as np


class Network:
    def __init__(self,sizes):
        self._numLayers = len(sizes)
        self._sizes = sizes
        self._weights = [np.random.randn(j,i)
                for(i,j)in zip(sizes[:-1],sizes[1:])]
        self._biases = [np.random.randn(b,1) for b in sizes]
        self._layers = []

        for i in range(_numLayers)
            layer = Layer(sizes[i])
            self._layers.append(layer)

        #load the test data to the first layer 'z'




    def forward(self,inData) #z is input vector of neuron layer
        #inData must be a np.array
        self._layers[0]._z = inData
        for b,w,z,a in zip(self._biases,self._weights,
                         self._layers[1:],self._layers[:-1]):
            z = activate(np.dot(w,a) + b)#dot is matrix product
            #the last layer input z is the output of dnn
        return z[-1]

    def backpro(self):

    def update(self,zeta):

    def predict(self,inData):
        conseq = forward(inData)

        #Mapping to 39 phome




    def loadModel(self):

    def saveModel(self):
