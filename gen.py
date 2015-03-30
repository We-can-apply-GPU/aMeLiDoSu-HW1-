#!/usr/bin/env python

import sys

def main():

    data = open("data/mfcc/train.ark")
    labl = open("data/label/train.lab")
    outdata = open("data/mfcc/trainToy.ark", "w")
    outlabl = open("data/label/trainToy.lab", "w")

    dic = {}

    for line in data:
        s = line.rstrip().split(" ")
        dic[s[0]] = line

    cnt = int(sys.argv[1])

    for line in labl:
        if cnt == 0:
            break
        cnt -= 1 
        s = line.rstrip().split(",")
        if s[0] in dic:
            outdata.write(dic[s[0]])
            outlabl.write(line)

if __name__ == '__main__':
    main()
