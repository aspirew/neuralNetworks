from typing import List

import numpy
from NeuralNetworkLayer import NeuralNetworkLayer

def softMax(inputs):
    return numpy.exp(inputs) / numpy.sum(numpy.exp(inputs))

def executeNeuralNetwork(inputs, neuralNetwork: List[NeuralNetworkLayer], activationFun):
    for layer in neuralNetwork:
        inputs = layer.calculateActivationVector(inputs, activationFun)

    return softMax(inputs)