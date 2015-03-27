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
def errFuncGradient(ls):
    r = []
    newls = []
    mart = f48t39()
    for i in ls:
        newvec = mart*i[0]
        newls.append((newvec,i[1]))
    for i in newls:
        r.append(2*np.sum(i[0]-i[1]))
    return r/ls.__len__()
def errSquare(ls):
    r = 0.0
    newls = []
    mart = f48t39()
    for i in ls:
        newvec = mart*i[0]
        newls.append((newvec,i[1]))
    for i in newls:
        r += np.sum((i[0]-i[1])**2)
    return r/ls.__len__()
def f48t39():
    m = open("../../phones/48_39.map")
    mapdic = {}
    iter39 = {}
    cnt = 0
    mart = []
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]]=tmp[1]
        if tmp[1] not in iter39:
            iter39[tmp[1]] = cnt
            cnt += 1
            vec = [0]*39
            vec[iter39[tmp[1]]] = 1
            mart.append(vec)
    return np.asarray(mart).transpose()

#The errorFunc will receive one argument, a list,
#and compare it with the valuse set in labels(need mapping)
#errorFuncPrime will,too
