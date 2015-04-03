#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: errorFunc.py
Description: evaluate the cost function
"""
import numpy as np
import os
def errFunc(ls):
    return errSquare(ls)
def errFuncPrime(Mart,lbs):
    one = np.ones(Mart.shape)
    rs = (Mart-lbs)/Mart
    return rs/(one-Mart)# maybe need to be divided by length
def errFuncPrimeSingle(ls,lb):
    r = [0]*48
    r = np.asarray(r)
    mapdic = {}
    idx = 0
    m = open("data/phones/48_39.map")
    for s in m:
        tmp = s.rstrip().split("\t")
        mapdic[tmp[0]] = idx
        idx += 1
    zv = [0]*48
    zv[mapdic[lb]] = 1
    ls = np.asarray(ls)
    lsc = np.asarray([1]*48)- ls
    r += (ls-np.asarray(zv))/ls
    return r/lsc # maybe need to be divided by length
def errSquare(ls):
    r = 0.0
    for i in ls:
        r += np.sum((i[0]-i[1])**2)
    return r 

#not completed
def f48t39_2(vec):
   ### 
    m = open("data/phones/48_39.map")
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
if __name__== "__main__":
    c = [0.1]*48
    b = [0.1]*48
    a = [0.1]*48
    a[5]=0.5
    b[1]=0.5
    c[0]=0.5
    print(errFuncPrime([a,b],["aa","aa"]))
   
#The errorFunc will receive one argument, a list,
#and compare it with the valuse set in labels(need mapping)
#errorFuncPrime will,too
