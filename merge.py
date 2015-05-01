#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File: merge.py
Description: merge many results to one file
"""

import sys

ID = []
result = []
candidate = {}

def infile():
    lab = open('model/' + sys.argv[1],'r')
    next(lab)
    for line in lab:
        l = line.rstrip().split(',')
        ID.append(l[0])            
    for i in range(1,len(sys.argv)):
        Predict = []
        lab = open('model/' + sys.argv[i],'r')
        next(lab)
        for line in lab:
            l = line.rstrip().split(',')
            Predict.append(l[1])            
        candidate[i-1]=Predict

def conbine():
    for i in range (0, len(candidate[0])):
        most = 0
        index = 0
        for j in range (0, len(candidate)):
            num = 0
            for k in range (0, len(candidate)):
                if candidate[j][i]==candidate[k][i]:
                    num=num+1
            if num>most:
                most = num
                index = j
        result.append(candidate[index][i])        
           
def outfile():
    f=open('merge.CSV','w')
    f.write('ID,Prediction\n')
    f=open('merge.CSV','a')
    for i in range (0, len(ID)):
        f.write(ID[i]+','+result[i]+'\n')



if __name__ == "__main__":
    infile()
    conbine()
    outfile()
