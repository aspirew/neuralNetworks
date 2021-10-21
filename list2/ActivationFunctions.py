import math

import numpy
    
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def tanh(x):
    return 2 / (1 + numpy.exp(-2*x) - 1)

def ReLU(x):
    return 0 if x < 0 else x