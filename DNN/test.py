from Network.network import Network 
import numpy as np
dnn = Network([2, 3, 4])
dnn.initialize()
dnn._gradW[0][0][0] = 100
dnn.saveModel("../model/123.txt")
dnn._gradW[0][0][0] = 0.0 
dnn.loadModel("../model/123.txt")
print(dnn._gradW[0][0][0])

