import sys
from typing import List
import numpy
import random
from numpy.lib.function_base import copy

from Loader import InputOutputSet, chunks

errors = []

def softMax(inputs):
    inputs = numpy.clip(inputs, -700, 700)
    exp = numpy.exp(inputs)
    return exp / numpy.sum(exp)

def softMaxF():
    return softMax, None

def neuralNetworkError(expectedOutputs, predictedOutputs):
    return predictedOutputs - expectedOutputs

def negativeLogLikelihood(errors, expectedOutputs):
    return -numpy.log(numpy.sum(errors * expectedOutputs))
    

def trainNeuralNetwork(dataset: List[InputOutputSet], testDataset, neuralNetwork, optimizer = None, batchSize = 500, numOfEpochs = 50):

    epoch = 0
    inputSize = len(dataset[0].inputs)
    batchSize = min(max(1, batchSize), inputSize)
    errors = []
    passedTestsList = []

    # execute while numOfEpochs is not exceeded
    while (numOfEpochs > epoch):

        # shuffle dataset
        random.shuffle(dataset)

        # make batchErrorList
        batchErrorList = []

        # yield a chunk of batchSize data from dataset
        for batch in chunks(dataset, batchSize):

            # process batch of data
            for data in batch:
                # propagate data forward and get softMax output
                output = neuralNetwork.propagateForward(data.inputs)

                # calculate error with negative log likelihood - using argmax as expected output
                neuralNetworkError = negativeLogLikelihood(output, data.getOutputAsArgmax())

                # propagate error backward
                neuralNetwork.propagateBackward(data.getOutputAsArgmax())

            # update weights after batch processing
            neuralNetwork.updateWeights(batchSize, optimizer)

            # add error to batch error list
            batchErrorList.append(neuralNetworkError)

        # when all batches are processed calculate average error and check how well network can predict values
        errors.append(numpy.mean(batchErrorList))
        passedTestsList.append(testNeuralNetwork(testDataset, neuralNetwork))
        neuralNetwork.setBatchEpochNum(1)

        # increase epoch
        epoch = epoch + 1

        print("█", end='')
        sys.stdout.flush()

    return errors, numpy.mean(errors), passedTestsList, numpy.mean(passedTestsList)


def testNeuralNetwork(dataset: List[InputOutputSet], neuralNetwork):

    passedTests = 0
    
    for data in dataset:
        output = neuralNetwork.propagateForward(data.inputs)
        argmax = numpy.argmax(output)

        if(argmax == data.output):
            passedTests += 1

    return copy(passedTests)
