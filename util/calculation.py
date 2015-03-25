#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: calculation.py
Description: use GPU acceleration method to do some calculation
"""

import numpy as np
    def activate(x):  #x is a vector
        sigmoidVec(x)



#different activate functions
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))

sigmoidVec = np.vectorize(sigmoid)

def sigmoidprime(x):
<<<<<<< HEAD
    return (np.exp(-x) / (1.0 + np.exp(-x))**2)

sigmoidprimeVec = np.vectorize(sigmoid)
=======
    return (np.exp(-x) / (1.0 + np.exp(-x))** 2)

sigmoidprimeVec = np.vectorize(sigmoidprime)
>>>>>>> f5f1844540528752abf51a067ec9ea37c9424ca2

