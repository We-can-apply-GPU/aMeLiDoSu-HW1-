#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: errorFunc.py
Description: evaluate the cost function
"""
import numpy as np
# write in the Network class??
def errFunc(ls,lb):
    return errSquare(ls,lb)
def errSquare(ls,lb):
    r = []
    mapdic = {}
    newls = []
    idx = 0
    m = open("48_39.map")
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]] = idx
        idx += 1
    for i,j in zip(ls,lb):
        zv = [0]*48
        zv[mapdic[j]] = 1
        r.append(2*(np.asarray(i)-np.asarray(zv))/ls.__len__())
    return r
#x = [0]*48
#x[2]=1
print(errFunc([x],["aa"]))
#The errorFunc will receive one argument, a list,
#and compare it with the valuse set in labels(need mapping)
#errorFuncPrime will,too
