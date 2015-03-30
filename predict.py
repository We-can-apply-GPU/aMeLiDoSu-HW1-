#!/usr/bin/python
"""
File: predict.py
Desctiption:
"""

import sys
from DNN.network import *
from DNN.iofile import *

def loadMap(trans):
    data = open("DNN/48_39.map")
    for line in data:
        print(line)
        s = line.rstrip().split("\t")
        trans.append((s[0], s[1]))

def main():
    dnn = Network()
    dnn.initialize(parsPath = "model/" + sys.argv[1])
    data = arkIn("data/mfcc/test.ark")
    out = open("output/" + sys.argv[1], "w")
    trans = []
    loadMap(trans)
    cnt = 1
    for row in data:
        output = dnn.forward(row[1:])
        max_index = 0
        for i in range(len(output)):
            if output[i] > output[max_index]:
                max_index = i
        out.write("{0},{1}\n".format(row[0], trans[max_index][1]))
        print("{0}/{1}".format(cnt, len(data)))
        cnt += 1

if __name__ == '__main__':
    main()
