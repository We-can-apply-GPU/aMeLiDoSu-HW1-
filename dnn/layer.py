#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: layer.py
Description:define the structure of each layer,its input and output 
"""
import numpy as np
class Layer:
    def __init__(self,numNeurons):
        self._z = np.zeros(numNeurons)
        self._a = np.zeros(numNeurons)

