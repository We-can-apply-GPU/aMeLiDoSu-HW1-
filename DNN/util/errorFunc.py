#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: errorFunc.py
Description: evaluate the cost function
"""
import numpy as np
# write in the Network class??
def errFunc(ls):
    return errSquare(ls)
def errSquare(ls):
    r = 0.0
    mapdic = {}
    newls = []
    m = open("48_39.map")
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]]=tmp[1]
    for i in ls:
        newls.append((mapdic[i[0]],i[1]))
    for i in newls:
        r += np.sum((np.asarray(i[0])-np.asarray(i[1]))**2)
    return r/ls.__len__()

#The errorFunc will receive one argument, a list,
#and compare it with the valuse set in labels(need mapping)
#errorFuncPrime will,too
