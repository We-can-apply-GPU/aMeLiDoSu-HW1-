def infile(ark = "./mfcc/train.ark", lab = "./label/train.lab"):
    dic = {}
    data = open(ark)
    labl = open(lab)
    for line in data:
        s = line.rstrip().split(" ")
        dic[s[0]] = [float(i) for i in s[1:]]
    for line in labl:
        s = line.rstrip().split(",")
        if s[0] in dic:  #just to handle error input
            dic[s[0]].append(s[1])
    return dic
infile()
