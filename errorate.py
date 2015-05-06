import numpy as np
def error(x,y):
    lx = x.__len__()
    ly = y.__len__()
    d = np.zeros((lx+1,ly+1))
    for i in range(lx) : 
        d[i][0] = i
    for j in range(ly) : 
        d[0][j] = j
    for j in range(1,ly+1):
        for i in range(1,lx+1):
            if x[i-1]== y[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min([d[i-1][j],d[i][j-1],d[i-1][j-1]])+1
    return int(d[lx][ly])
def read(infile="./train.csv",ans="./ans.csv"):
    x = open(infile)
    y = open(ans)
    err = 0
    reflen = 0 
    dic = {}
    for line in y:
        line = line.rstrip().split(",")
        dic[line[0]] = line[1]
    for line in x:
        line = line.rstrip().split(",")
        err += error(dic[line[0]],line[1])
        reflen+=dic[line[0]].__len__()
    return err/reflen

print(read())
