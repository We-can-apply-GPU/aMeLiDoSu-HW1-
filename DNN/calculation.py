#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: calculation.py
Description: use GPU acceleration method to do some calculation
"""

import numpy as np
import theano
import theano.tensor as T


#def activate(x):  #x is a vector
    #return sigmoidVec(x)

#different activate functions
def sigmoid(x):
    return (1.0 / (1.0 + T.exp(-1.0*x)))

#sigmoidVec = np.vectorize(sigmoid)

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#sigmoidPrimeVec = np.vectorize(sigmoidPrime)



##Theano section
vec1 = T.vector(name='vec1')
vec2 = T.vector(name='vec2')
matrix = T.matrix(name="matrix")

Tdot = theano.function([vec1,vec2], T.dot(vec1, vec2), name='Tdot')
TMVdot = theano.function([matrix,vec2], T.dot(matrix, vec2), name='TMVdot')
Touter = theano.function([vec1,vec2], T.outer(vec1, vec2), name='Tdot')
sigmoidVec = theano.function([vec1],sigmoid(vec1),name="sigmoidVec")
sigmoidPrimeVec = theano.function([vec1],sigmoidPrime(vec1),name="sigmoidPrimeVec")

if __name__ == '__main__':
    print(sigmoidVec([0,10,-100]))
    print(sigmoidPrimeVec([0,10,-100]))
