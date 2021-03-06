#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File: trim.py
Description: trim
"""
import sys

ID = []
Prediction = []
        
def infile():
    lab = open(sys.argv[1],'r')
    next(lab)
    for line in lab:
        l = line.rstrip().split(',')
        ID.append(l[0])
        Prediction.append(l[1])

def trim(startnum,end):
    string = ''
    name = ID[startnum]
    for i in range (startnum, end):
        if Prediction[i] != 'sil':
            startnum = i
            break
    for i in range (end-1, startnum, -1):
        if Prediction[i] != 'sil':
            end = i
            break
    for i in range (startnum, end):
        if Prediction[i] != Prediction[i+1]:
            string = string + chartoalpha(Prediction[i])
        if i == end-1:
            string = string + chartoalpha(Prediction[i])
    outfile(name,string)

def chartoalpha(c):
    chrmap = open('data/48_idx_chr.map_b','r')
    for line in chrmap:
        s = line.split()
        for line in s:
            if s[0] == c:
                string = s[2]
                break
    return string

def outfile(name,string):
    name = name[0:-2]
    if name=='fadg0_si1279':
        f=open('trim.CSV','w')
        f.write('id,phone_sequence\n')
    f=open('trim.CSV','a')
    f.write(name + ',' + string)    
    f.write('\n')

def segment():
    startnum = 0
    nowID = ID[startnum]
    nowID = nowID[0:-1]
    for i in range (0, len(ID)):
        if nowID not in ID[i]:
            trim(startnum, i)
            startnum = i
            nowID = ID[i]
            nowID = nowID[0:-1]
        if i == len(ID)-1:
            trim(startnum,len(ID))

if __name__=="__main__":
    infile()
    segment()
