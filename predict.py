#!/usr/bin/python3
"""
File: predict.py
Desctiption:
"""

import sys
from DNN.network import *
from iofile import *
import util


def main():
    dnn = Network()
    dnn.initialize(1,parsPath = "model/" + sys.argv[1]) # batchsize is 1
    data = arkIn("data/mfcc/test.ark")
    out39 = open("output48/" + sys.argv[1] + ".csv", "w")
    out48 = open("output39/" + sys.argv[1] + ".csv", "w")

    trans = []
    util.loadMapList(trans)
    cnt = 1
    out39.write("Id,Prediction\n")
    out48.write("Id,Prediction\n")
    inputData = []
    for row in data:
        inputData.append(row[1:])
    output = dnn.forward(np.transpose(np.array(inputData,dtype="float32")))
    output = output.T
    
    for i in range(len(output)):
        maxIndex = util.chooseMax(output[i])
        out39.write("{0},{1}\n".format(data[i][0],trans[maxIndex][1]))
        out48.write("{0},{1}\n".format(data[i][0],trans[maxIndex][0]))

    #for row in data:
        #max_index = util.chooseMax(dnn.forward(row[1:]))
        #out.write("{0},{1}\n".format(row[0], trans[max_index][1]))
        #print("{0}/{1}".format(cnt, len(data)))
        #cnt += 1

if __name__ == '__main__':
    main()
