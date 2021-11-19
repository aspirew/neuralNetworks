import numpy
    
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoidD(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoidF():
    return sigmoid, sigmoidD

def tanh(x):
    v = (numpy.exp(x) - numpy.exp(-x))/(numpy.exp(x) + numpy.exp(-x))
    if(0 in v):
        print(v)
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
    return numpy.array(x).clip(0)

def ReLUF():
    return ReLU, softplusD