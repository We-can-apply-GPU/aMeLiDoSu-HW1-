def loadMap(trans):
    data = open("DNN/48_39.map")
    for line in data:
        s = line.rstrip().split("\t")
        trans.append((s[0], s[1]))

def chooseMax(xxx):
    max_index = 0
    for i in range(len(xxx)):
        if xxx[i] > xxx[max_index]:
            max_index = i
    return max_index
