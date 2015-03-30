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

sigmoidVec = np.vectorize(sigmoid)

def sigmoidPrime(x):
    return (np.exp(-1.0*float((x)) / (1.0 + np.exp(-1.0*(x)))** 2))

sigmoidPrimeVec = np.vectorize(sigmoidPrime)
