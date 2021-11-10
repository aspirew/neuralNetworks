from ActivationFunctions import sigmoidF
from Loader import loadData
from NeuralNetworkExecutor import softMaxF, trainNeuralNetwork
from NeuralNetworkLayer import NeuralNetworkLayer
from WeightsGenerator import normalDistribution

    
def generateNetwork(neuralNetwork, neuronsCounts, outputLayerSize, learnRate, activationFun, randomFun, randomFunParams):
    inputSize = neuralNetwork.numOfNeurons
    for nc in neuronsCounts:
        neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, nc, learnRate, activationFun, randomFun, randomFunParams)
        inputSize = nc
    neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, outputLayerSize, learnRate, softMaxF, randomFun, randomFunParams)

if __name__ == '__main__':

    trainingData, inputData = loadData()
    inputVectorSize = len(trainingData[0].inputs)
    
    # create first layer
    neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.01, sigmoidF, normalDistribution, [0, 0.5])

    # add hidden layers of size 3 and 2 and output layer of size 10
    generateNetwork(neuralNetwork, [3, 2], 10, 0.01, sigmoidF, normalDistribution, [0, 0.5])

    trainNeuralNetwork(trainingData, neuralNetwork)
