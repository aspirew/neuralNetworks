import numpy
import random
import matplotlib.pyplot as plt

errors = []

def softMax(inputs):
    return numpy.exp(inputs) / numpy.sum(numpy.exp(inputs))

def softMaxD(softMaxOutputs, expectedOutputs):
    return -(expectedOutputs - softMaxOutputs)

def softMaxF():
    return softMax, softMaxD

def negativeLogLikelihood(outputs, expected):
    return numpy.sum(-numpy.log(outputs) * expected)

def lossFunctionGradient(softMaxOutputs, expectedOutputs, inputs):
    return numpy.matmul(softMaxD(softMaxOutputs, expectedOutputs), inputs.T)

def executeNeuralNetwork(inputs, expectedOutputs, neuralNetwork, batchSize = 0, acceptedError = 0.7, numOfEpochs = 1000):

    epoch = 1
    foundOptimalValues = False
    batchSize = len(inputs) if batchSize < 1 or batchSize > len(inputs) else batchSize

    while numOfEpochs > epoch and not foundOptimalValues:
        batch = range(len(inputs)) if batchSize < 1 or batchSize >= len(inputs) else random.sample(range(len(inputs)), batchSize)

        for input in batch:
            neuralNetworkOutputs = neuralNetwork.propagateForward(inputs[input])
            neuralNetwork.propagateBackward(expectedOutputs[input], batchSize)
        
        epoch = epoch + 1
        errors.append(negativeLogLikelihood(neuralNetworkOutputs, expectedOutputs))
        foundOptimalValues = errors[-1] < acceptedError
    
    print(epoch)
    plt.plot(errors)
    plt.show()

    return neuralNetwork.propagateForward(inputs[0])

