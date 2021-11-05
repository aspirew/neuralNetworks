import numpy
from ActivationFunctions import sigmoidF, softplusF, tanhF
from NeuralNetworkExecutor import executeNeuralNetwork, softMaxF
from NeuralNetworkLayer import NeuralNetworkLayer
from WeightsGenerator import normalDistribution

def generateNetwork(neuralNetwork: NeuralNetworkLayer, neuronsCounts, learnRate, activationFun, randomFun, randomFunParams):
    inputSize = neuralNetwork.numOfNeurons
    for nc in neuronsCounts:
        neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, nc, learnRate, activationFun, randomFun, randomFunParams)
        inputSize = nc

if __name__ == '__main__':

    inputs = numpy.array([[[0.2]], [[0.5]], [[1]]])

    expectedOutputs = numpy.array([[[0], [1]], [[1], [1]], [[0], [1]]])
    
    neuralNetwork = NeuralNetworkLayer(len(inputs[0]), 2, 0.1, sigmoidF, normalDistribution, [0.5, 1])
    generateNetwork(neuralNetwork, [4, 3, 2], 0.1, sigmoidF, normalDistribution, [0.1, 5])

    print(executeNeuralNetwork(inputs, expectedOutputs, neuralNetwork))
