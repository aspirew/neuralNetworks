import numpy
from ActivationFunctions import sigmoid
from NeuralNetworkExecutor import executeNeuralNetwork, softMax
from NeuralNetworkLayer import NeuralNetworkLayer
from WeightsGenerator import normalDistribution

def generateNetwork(inputSize, neuronsCounts, randomFun, randomFunParams):
    layersArray = []
    for nc in neuronsCounts:
        layersArray.append(NeuralNetworkLayer(inputSize, nc, randomFun, randomFunParams))
        inputSize = nc
    return layersArray

if __name__ == '__main__':

    inputs = numpy.array([[1], [1]])

    neuralNetwork = [
        NeuralNetworkLayer(2, 3, normalDistribution, [0.1, 1]),
        NeuralNetworkLayer(3, 1, normalDistribution, [0.1, 1]),
        NeuralNetworkLayer(1, 4, normalDistribution, [0.1, 1]),
        NeuralNetworkLayer(4, 2, normalDistribution, [0.1, 1]),
    ]

    generatedNeuralNetwork = generateNetwork(len(inputs), [3, 1, 4, 2], normalDistribution, [0.1, 5])

    print(executeNeuralNetwork(inputs, generatedNeuralNetwork, sigmoid))
