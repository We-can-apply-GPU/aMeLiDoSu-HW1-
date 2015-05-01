#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: calculation.py
Description: use GPU acceleration method to do some calculation
"""

import numpy as np
#import theano
#import theano.tensor as T


#def activate(x):  #x is a vector
    #return sigmoidVec(x)

#different activate functions
#def sigmoid(x):
z = [0]
a = [(0,0),(1,0)]

def sigmoidMaxout(x):
    for i in range(x.__len__()):
        for j in range(x[i].__len__()):
            k = 0
            while z[k] < x[i][j]:
                k+=1
                if k == z.__len__():
                    break
            x[i][j] = x[i][j]*a[k][0]+a[k][1]
    return x
#sigmoidVec = np.vectorize(sigmoid)
def sigmoidPrimeMaxout(x):
    for i in range(x.__len__()):
        for j in range(x[i].__len__()):
            k = 0
            while z[k] < x[i][j]:
                k+=1
                if k == z.__len__():
                    break
            x[i][j] = a[k][0]
    return x

def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#sigmoidtPrimeVec = np.vectorize(sigmoidPrime)

def softMax(Mart):
   #out = Mart
   out = np.exp(Mart)
   for column in out.T:
       total = float(np.sum(column))
       column /= total
   return out

##Theano section
#vec1 = T.vector(name='vec1')
#vec2 = T.vector(name='vec2')
#matrix1 = T.matrix(name="matrix1")
#matrix2 = T.matrix(name="matrix2")

#Tdot = theano.function([matrix1,matrix2], T.dot(matrix1, matrix2), name='Tdot', allow_input_downcast = True)
#Touter = theano.function([matrix1,matrix2], T.outer(matrix1, matrix2), name='Touter', allow_input_downcast = True)
#sigmoidVec = theano.function([matrix1],sigmoid(matrix1),name="sigmoidVec", allow_input_downcast = True)
#sigmoidPrimeVec = theano.function([matrix1],sigmoidPrime(matrix1),name="sigmoidPrimeVec", allow_input_downcast = True)

if __name__ == '__main__':
    #print(sigmoidVec([0,10,-100]))
    #print(sigmoidPrimeVec([0,10,-100]))
    print("SOFTMAX TEST")
    a = (np.arange(1,10,1,dtype = np.float32).reshape((3,3)))
    print(softMax(a))
    #x=[[0,-1],[2,5]]
    #print(sigmoidMaxout(x))
    #print(sigmoidPrimeMaxout(x))
#>>>>>>> origin/maxout
