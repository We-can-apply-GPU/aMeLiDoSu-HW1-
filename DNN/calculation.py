#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: calculation.py
Description: use GPU acceleration method to do some calculation
"""

import numpy as np

def activate(x):  #x is a vector
    return sigmoidVec(x)

#different activate functions
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-1.0*(x))))
#def sigmoidVec(xVec):
    #outVec = []
    #for x in xVec:
        #outVec.append(sigmoid(x))
    #return np.array(outVec)


sigmoidVec = np.vectorize(sigmoid)

def sigmoidPrime(x):
    return (np.exp(-1.0*float((x)) / (1.0 + np.exp(-1.0*(x)))** 2))

#def sigmoidPrimeVec(xVec):
    #outVec = []
    #print("TT xVec is {}".format(xVec))
    #for x in xVec:
        #print("TT x is {}".format(x))
        #outVec.append(sigmoidPrime(x))
    #return np.array(outVec)

sigmoidPrimeVec = np.vectorize(sigmoidPrime)
