from os import name
import random
import sys

from numpy import tan, tanh
import numpy

from ActivationFunctions import ReLUF, sigmoid, sigmoidF, softplusF, tanhF
from LearningRateOptimizer import adadelta, adagrad, adam, momentum, nestrovMomentum
from Loader import loadData
from NeuralNetworkExecutor import softMaxF, trainNeuralNetwork
from NeuralNetworkLayer import NeuralNetworkLayer
from WeightsGenerator import he, normalDistribution, xavier
import matplotlib.pyplot as plt
import json
    
def generateNetwork(neuralNetwork, neuronsCounts, outputLayerSize, learnRate, activationFun, randomFun, randomFunParams):
    inputSize = neuralNetwork.numOfNeurons
    for i, nc in enumerate(neuronsCounts):
        if(randomFun == xavier):
            randomFunParams = [outputLayerSize if i > len(neuronsCounts)-1 else neuronsCounts[i + 1]]
        neuralNetwork.getLastLayer().nextLayer = NeuralNetworkLayer(inputSize, nc, learnRate, activationFun, randomFun, randomFunParams)
        inputSize = nc
    if(randomFun == xavier):
        randomFunParams = [0]
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

def optimizerTests(trainingData, testData):

    optimizers = [adagrad]
    activationFuncs = [sigmoidF]

    for activation in activationFuncs:
        for optimizer in optimizers:
            totalErrors = []
            totalPassedTests = []
            avgAvgError = 0
            avgAvgPass = 0
            for _ in range(10):
                neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.01, activation, normalDistribution, [0, 0.1])
                generateNetwork(neuralNetwork, [], 10, 0.01, activation, normalDistribution, [0, 0.1])
                errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, optimizer)

                if(len(totalErrors) < 1):
                    totalErrors = errors
                    totalPassedTests = passedTestsList
                else:
                    for i in range(len(totalErrors)):
                        totalErrors[i] += errors[i]
                        totalPassedTests[i] += passedTestsList[i]
                avgAvgError += avgError
                avgAvgPass += avgPass

            for i in range(len(totalErrors)):
                totalErrors[i] = totalErrors[i] / 10
                totalPassedTests[i] = totalPassedTests[i] / 10

            name1 = optimizer.__name__ if(optimizer is not None) else "none"
            name2 = activation.__name__

            plt.title(f"Wielkość błędu dla {optimizer.__name__ if(optimizer is not None) else 'braku optymalizatora'} oraz {name2}")
            plt.xlabel("Epoka")
            plt.ylabel("Błąd")
            plt.plot(totalErrors)
            plt.savefig(f"report/images/error_{name1}_{name2}.png")
            plt.clf()

            plt.title(f"Ilość pomyślnych testów dla {optimizer.__name__ if(optimizer is not None) else 'braku optymalizatora'} oraz {name2}")
            plt.xlabel("Epoka")
            plt.ylabel("Ilość pomyślnych testów")
            plt.plot(totalPassedTests)
            plt.savefig(f"report/images/test_{name1}_{name2}.png")
            plt.clf()

            print(avgAvgError / 10)
            print(avgAvgPass / 10)


def weightsInitTest(trainingData, testData):

    weightInits = [xavier, he]
    activationFuncs = [sigmoidF]

    for activation in activationFuncs:
        for weightInit in weightInits:
            totalErrors = []
            totalPassedTests = []
            avgAvgError = 0
            avgAvgPass = 0
            for _ in range(10):
                if(weightInit == xavier):
                    s = [10]
                else:
                    s = []
                neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.01, activation, weightInit, s)
                generateNetwork(neuralNetwork, [], 10, 0.01, activation, weightInit, [])
                errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, None)

                if(len(totalErrors) < 1):
                    totalErrors = errors
                    totalPassedTests = passedTestsList
                else:
                    for i in range(len(totalErrors)):
                        totalErrors[i] += errors[i]
                        totalPassedTests[i] += passedTestsList[i]
                avgAvgError += avgError
                avgAvgPass += avgPass

            for i in range(len(totalErrors)):
                totalErrors[i] = totalErrors[i] / 10
                totalPassedTests[i] = totalPassedTests[i] / 10

            name1 = weightInit.__name__
            name2 = activation.__name__

            plt.title(f"Wielkość błędu dla {name1} oraz {name2}")
            plt.xlabel("Epoka")
            plt.ylabel("Błąd")
            plt.plot(totalErrors)
            plt.savefig(f"report/images/error_{name1}_{name2}.png")
            plt.clf()

            plt.title(f"Ilość pomyślnych testów dla {name1} oraz {name2}")
            plt.xlabel("Epoka")
            plt.ylabel("Ilość pomyślnych testów")
            plt.plot(totalPassedTests)
            plt.savefig(f"report/images/test_{name1}_{name2}.png")
            plt.clf()

            print(avgAvgError / 10)
            print(avgAvgPass / 10)

if __name__ == '__main__':

    seed = random.randrange(sys.maxsize)
    random.seed(seed)

    # # load mnist training data
    trainingData, testData = loadData()
    inputVectorSize = len(trainingData[0].inputs)
    # optimizerTests(trainingData, testData)
    weightsInitTest(trainingData, testData)
    
    # # create first layer with 4 neurons
    # neuralNetwork = NeuralNetworkLayer(inputVectorSize, 20, 0.01, sigmoidF, normalDistribution, [0, 0.1])

    # # # add hidden layers of size 3 and 2 and output layer of size 10
    # generateNetwork(neuralNetwork, [], 10, 0.01, sigmoidF, normalDistribution, [0, 0.1])

    # # generated network is of sizes: * -> 4 -> 3 -> 2 -> 10

    # # train neural network
    # errors, avgError, passedTestsList, avgPass = trainNeuralNetwork(trainingData, testData, neuralNetwork, None, 500, 10)

    # plt.xlabel("Epoka")
    # plt.ylabel("Błąd")
    # plt.plot(errors)
    # plt.show()
    # plt.clf()

    # plt.xlabel("Epoka")
    # plt.ylabel("Ilość pomyślnych testów")
    # plt.plot(passedTestsList)
    # plt.show()
    # plt.clf()