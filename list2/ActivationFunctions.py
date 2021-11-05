import numpy
    
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoidD(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoidF():
    return sigmoid, sigmoidD

def tanh(x):
    return (numpy.exp(x) - numpy.exp(-x))/(numpy.exp(x) + numpy.exp(-x))

def tanhD(x):
    return (1 - numpy.power(x, 2))

def tanhF():
    return tanh, tanhD

def softplus(x):
    return numpy.log(1 + numpy.exp(x))

def softplusD(x):
    return 1 / (1 + numpy.exp(-x))

def softplusF():
    return softplus, softplusD

def ReLU(x):
    return numpy.max(0, x)