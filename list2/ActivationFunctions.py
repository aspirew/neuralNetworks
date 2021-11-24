import numpy
    
def sigmoid(x):
    x = numpy.clip(x, -700, 700)
    return 1 / (1 + numpy.exp(-x))

def sigmoidD(x):
    x = numpy.clip(x, -700, 700)
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoidF():
    return sigmoid, sigmoidD

def tanh(x):
    x = numpy.clip(x, -700, 700)
    return (numpy.exp(x) - numpy.exp(-x))/(numpy.exp(x) + numpy.exp(-x))

def tanhD(x):
    x = numpy.clip(x, -700, 700)
    return (1 - numpy.power(x, 2))

def tanhF():
    return tanh, tanhD

def softplus(x):
    x = numpy.clip(x, -700, 700)
    return numpy.log(1 + numpy.exp(x))

def softplusD(x):
    x = numpy.clip(x, -700, 700)
    return 1 / (1 + numpy.exp(-x))

def softplusF():
    return softplus, softplusD

def ReLU(x):
    return numpy.array(x).clip(0)

def ReLUF():
    return ReLU, softplusD