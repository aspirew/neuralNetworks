from os import name
import random
import sys

from numpy import tan

from ActivationFunctions import ReLUF, sigmoid, sigmoidF, softplusF, tanhF
from Loader import loadData
from NeuralNetworkExecutor import softMaxF, trainNeuralNetwork
from NeuralNetworkLayer import NeuralNetworkLayer
from WeightsGenerator import normalDistribution
import matplotlib.pyplot as plt
import json
    
def generateNetwork(neuralNetwork, neuronsCounts, outputLayerSize, learnRate, activationFun, randomFun, randomFunParams):
    inputSize = neuralNetwork.numOfNeurons
    for nc in neuronsCounts:
        neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, nc, learnRate, activationFun, randomFun, randomFunParams)
        inputSize = nc
    neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, outputLayerSize, learnRate, softMaxF, randomFun, randomFunParams)


def hiddenLayersTest(trainingData, testData):

    testCases = [[5], [10], [20], [30]]

    testCases.append([5, 5])
    testCases.append([5, 10, 20])

    for case in testCases:

        print("testing hidden layers", case)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, case[0], 0.3, tanhF, normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, case[1:], 10, 0.3, tanhF, normalDistribution, [0, 0.5])
        errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, 50)

        plt.title(f"Wartość błędu dla sieci {case}")
        plt.xlabel("Epoka")
        plt.ylabel("Liczba zdanych testów")
        plt.plot(errors)
        plt.savefig(f"report/images/hiddenLayer/error_learn_rate_{case}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla sieci {case}")
        plt.xlabel("Epoka")
        plt.ylabel("Liczba zdanych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"report/images/hiddenLayer/passedTest_learn_rate_{case}.png")
        plt.clf()
        print(avgError)
        print(avgPass)


def learnRateTest(trainingData, testData):

    testCases = [0.1, 0.2, 0.3, 0.5, 0.75, 1]
    
    for learnRate in testCases:

        print("Testing learn rate", learnRate)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, learnRate, tanhF, normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [], 10, learnRate, tanhF, normalDistribution, [0, 0.5])
        errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, 50)

        plt.title(f"Wartość błędu dla współczynnika {learnRate}")
        plt.xlabel("Epoka")
        plt.ylabel("Liczba zdanych testów")
        plt.plot(errors)
        plt.savefig(f"report/images/learnRate/error_learn_rate_{learnRate}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla współczynnika {learnRate}")
        plt.xlabel("Epoka")
        plt.ylabel("Liczba zdanych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"report/images/learnRate/test_learn_rate_{learnRate}.png")
        plt.clf()

        print(avgError)
        print(avgPass)
    

def batchSizeTest(trainingData, testData):
    
    testCases = [10, 50, 100, 500, 1000, 10000, len(trainingData)]

    for batchSize in testCases:

        print("Testing batchSize ", batchSize)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.3, tanhF, normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [], 10, 0.3, tanhF, normalDistribution, [0, 0.5])
        errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, batchSize)

        plt.title(f"Wielkość błędu dla wielkości paczki {batchSize}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"report/images/batchsize/error_batch_size_{batchSize}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla wielkości paczki {batchSize}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"report/images/batchsize/test__batch_size_{batchSize}.png")
        plt.clf()

        print(avgError)
        print(avgPass)

def weightsInitTest(trainingData, testData):
   
    testCases = [0.6, 0.7, 0.3]

    for sigma in testCases:

        print("Testing normal distribution for sigma ", sigma)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.3, tanhF, normalDistribution, [0, sigma])
        generateNetwork(neuralNetwork, [], 10, 0.3, tanhF, normalDistribution, [0, sigma])
        errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork)

        plt.title(f"Wielkość błędu dla rozkładu normalnego dla sigma {sigma}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"report/images/weight/error_weight_{sigma}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla rozkładu normalnego dla {sigma}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"report/images/weight/test_weight_{sigma}.png")
        plt.clf()

        print(avgError)
        print(avgPass)


def activationFunTest(trainingData, testData):
   
    activationFuncs = [softplusF]
    names = ["softplus"]

    for i in range(len(activationFuncs)):

        print("Testing activation function ", names[i])

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.3, activationFuncs[i], normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [], 10, 0.05, activationFuncs[i], normalDistribution, [0, 0.5])
        errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork)

        plt.title(f"Wielkość błędu dla {names[i]}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"report/images/activation/error_activation_{names[i]}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla {names[i]}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"report/images/activation/test_activation_{names[i]}.png")
        plt.clf()

        print(avgError)
        print(avgPass)

if __name__ == '__main__':

    seed = random.randrange(sys.maxsize)
    random.seed(seed)

    # load mnist training data
    trainingData, testData = loadData()
    inputVectorSize = len(trainingData[0].inputs)
    batchSizeTest(trainingData, testData)
    
    # create first layer with 4 neurons
    # neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.1, sigmoidF, normalDistribution, [0, 0.5])

    # # add hidden layers of size 3 and 2 and output layer of size 10
    # generateNetwork(neuralNetwork, [3, 2], 10, 0.1, sigmoidF, normalDistribution, [0, 0.5])

    # # generated network is of sizes: * -> 4 -> 3 -> 2 -> 10

    # # train neural network
    # trainNeuralNetwork(trainingData, testData, neuralNetwork)