#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: infile.py
Description: helper function : loadData and create minibatch
"""
import numpy as np

def arkIn(ark):
    ans = []
    data = open(ark)
    for line in data:
        s = line.rstrip().split(" ")
        for i in range(1, len(s)):
            s[i] = float(s[i])
        ans.append(s)
    data.close()
    return ans 

def infile(ark, lab):
    dataset=[]
    dic = {}
    data = open(ark)
    labl = open(lab)
    for line in data:
        s = line.rstrip().split(" ")
        dic[s[0]] = np.array([float(i) for i in s[1:]])
    for line in labl:
        s = line.rstrip().split(",")
        if s[0] in dic:  #just to handle error input
            dataset.append(tuple((dic[s[0]],s[1])))
    #test section
    #for i in dic.keys():
        #print("{} , ".format(i))
    #for i in range(len(dataset)):
        #print("{}:{}".format(i,dataset[i]))
    return dataset

if __name__ == "__main__":
    dataset = infile()
    miniBatch(2, dataset)
