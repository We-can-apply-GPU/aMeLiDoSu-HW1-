#!/usr/bin/python3
"""
File: predict.py
Desctiption:
"""

import sys
from DNN.network import *
from DNN.iofile import *
import util


def main():
    dnn = Network()
    dnn.initialize(parsPath = "model/" + sys.argv[1])
    data = arkIn("data/mfcc/test.ark")
    out = open("output/" + sys.argv[1] + ".csv", "w")
    trans = []
    util.loadMap(trans)
    cnt = 1
    for row in data:
        max_index = util.chooseMax(dnn.forward(row[1:]))
        out.write("{0},{1}\n".format(row[0], trans[max_index][1]))
        print("{0}/{1}".format(cnt, len(data)))
        cnt += 1

if __name__ == '__main__':
    main()
