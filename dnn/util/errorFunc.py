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
    for i in ls:
        r.append(2*np.sum(i[0]-i[1]))
    return np.asarray(r/ls.__len__())
def errSquare(ls):
    r = 0.0
    for i in ls:
        r += np.sum((i[0]-i[1])**2)
    return r/ls.__len__()

#not completed
def f48t39_2(vec):
    m = open("../../phones/48_39.map")
    mapdic = {}
    iter39 = {}
    cnt = 0
    ls39 = []
    martls = []
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]]=tmp[1]
        if tmp[1] not in iter39:
            ls39.append(tmp[1]) 
            iter39[tmp[1]] = cnt
            cnt += 1
            vec = [0]*39
            vec[iter39[tmp[1]]] = 1
            martls.append(vec)
    mart = np.asarray(martls).transpose()
    vec = mart*np.transpose(np.asarray(vec))
    print(vec.shape)
    return -1#ls39[np.argmax(vec)]

def f48t39_1(vec):
    m = open("../../phones/48_39.map")
    mapdic = {}
    ls48 = []
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]]=tmp[1]
        ls48.append(tmp[0]) 
    return mapdic[ls48[np.argmax(vec)]]
arr = [0]*48
arr[45] = 100
arr = np.asarray(arr)
print(f48t39_2(arr))
#The errorFunc will receive one argument, a list,
#and compare it with the valuse set in labels(need mapping)
#errorFuncPrime will,too
