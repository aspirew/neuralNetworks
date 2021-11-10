from os import name
import random
import sys
from ActivationFunctions import ReLUF, sigmoidF, softplusF, tanhF
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


def speedAndEfficiencyTest(trainingData, testData):

    errs = []
    passed = []

    neuralNetwork1 = NeuralNetworkLayer(inputVectorSize, 2, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork1, [], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork1)
    errs.append(len(errors))
    passed.append(max(passedTestsList))

    print(errs)
    print(passed)

    neuralNetwork2 = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork2, [], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork2)
    errs.append(len(errors))
    passed.append(max(passedTestsList))

    print(errs)
    print(passed)


    neuralNetwork3 = NeuralNetworkLayer(inputVectorSize, 6, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork3, [], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork3)
    errs.append(len(errors))
    passed.append(max(passedTestsList))

    neuralNetwork4 = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork4, [3], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork4)
    errs.append(len(errors))
    passed.append(max(passedTestsList))

    neuralNetwork5 = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork5, [3, 2], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork5)
    errs.append(len(errors))
    passed.append(max(passedTestsList))

    neuralNetwork6 = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    generateNetwork(neuralNetwork6, [3, 2, 5], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
    errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork6)
    errs.append(len(errors))
    passed.append(max(passedTestsList))


    with open('results/speed.txt', 'w') as filehandle:
        json.dump(errs, filehandle)

    with open('results/efficiency.txt', 'w') as filehandle:
        json.dump(passed, filehandle)

def learnRateTest(trainingData, testData):
    
    for i in range(10):

        learnRate = 0.005 * (i * 3 + 1)
        print("Testing learn rate", learnRate)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, learnRate, sigmoidF, normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [3, 2], 10, learnRate, sigmoidF, normalDistribution, [0, 0.5])
        errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork)

        plt.title(f"Wielkość błędu dla współczynnika {learnRate}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"results/images/learnRate/error_learn_rate_{learnRate}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla współczynnika {learnRate}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"results/images/learnRate/test_learn_rate_{learnRate}.png")
        plt.clf()

        with open('results/speedlr.txt', 'w') as filehandle:
            json.dump(errors, filehandle)

        with open('results/efficiencylr.txt', 'w') as filehandle:
            json.dump(passedTestsList, filehandle)


def batchSizeTest(trainingData, testData):
    
    testCases = [1, 10, 50, 100, 500, 1000, 2000, len(trainingData)]

    for batchSize in testCases:

        print("Testing batchSize", batchSize)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [3, 2], 10, 0.05, sigmoidF, normalDistribution, [0, 0.5])
        errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork, batchSize)

        plt.title(f"Wielkość błędu dla wielkości paczki {batchSize}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"results/images/batchsize/error_batch_size_{batchSize}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla wielkości paczki {batchSize}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"results/images/batchsize/test__batch_size_{batchSize}.png")
        plt.clf()

    with open('results/speedbs.txt', 'w') as filehandle:
        json.dump(errors, filehandle)

    with open('results/efficiencybs.txt', 'w') as filehandle:
        json.dump(passedTestsList, filehandle)

def weightsInitTest(trainingData, testData):
   
    for i in range(10):

        sigma = 0.1 * (i+1)

        print("Testing normal distribution for sigma ", sigma)

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.05, sigmoidF, normalDistribution, [0, sigma])
        generateNetwork(neuralNetwork, [3, 2], 10, 0.05, sigmoidF, normalDistribution, [0, sigma])
        errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork)

        plt.title(f"Wielkość błędu dla rozkładu normalnego dla sigma {sigma}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"results/images/weight/error_weight_{sigma}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla rozkładu normalnego dla {sigma}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"results/images/weight/test_weight_{sigma}.png")
        plt.clf()

        with open('results/speedwi.txt', 'w') as filehandle:
            json.dump(errors, filehandle)

        with open('results/efficiencywi.txt', 'w') as filehandle:
            json.dump(passedTestsList, filehandle)


def activationFunTest(trainingData, testData):
   
    activationFuncs = [sigmoidF, tanhF, ReLUF]
    names = ["sigmoid", "tanh", "ReLU"]

    for i in range(len(activationFuncs)):

        print("Testing activation function ", names[i])

        neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.05, activationFuncs[i], normalDistribution, [0, 0.5])
        generateNetwork(neuralNetwork, [3, 2], 10, 0.05, activationFuncs[i], normalDistribution, [0, 0.5])
        errors, passedTestsList = trainNeuralNetwork(trainingData, testData, neuralNetwork)

        plt.title(f"Wielkość błędu dla {names[i]}")
        plt.xlabel("Epoka")
        plt.ylabel("Błąd")
        plt.plot(errors)
        plt.savefig(f"results/images/activation/error_activation_{names[i]}.png")
        plt.clf()

        plt.title(f"Ilość pomyślnych testów dla {names[i]}")
        plt.xlabel("Epoka")
        plt.ylabel("Ilość pomyślnych testów")
        plt.plot(passedTestsList)
        plt.savefig(f"results/images/activation/test_activation_{names[i]}.png")
        plt.clf()


if __name__ == '__main__':

    seed = random.randrange(sys.maxsize)
    random.seed(seed)

    # load mnist training data
    trainingData, testData = loadData()
    inputVectorSize = len(trainingData[0].inputs)
    activationFunTest(trainingData, testData)

    
    # # create first layer with 4 neurons
    # neuralNetwork = NeuralNetworkLayer(inputVectorSize, 4, 0.1, sigmoidF, normalDistribution, [0, 0.5])

    # # add hidden layers of size 3 and 2 and output layer of size 10
    # generateNetwork(neuralNetwork, [3, 2], 10, 0.1, sigmoidF, normalDistribution, [0, 0.5])

    # # generated network is of sizes: * -> 4 -> 3 -> 2 -> 10

    # # train neural network
    # trainNeuralNetwork(trainingData, testData, neuralNetwork)