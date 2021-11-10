from typing import List
import numpy
import random
import matplotlib.pyplot as plt
import sys

from Loader import InputOutputSet, chunks

errors = []

def softMax(inputs):
    return numpy.exp(inputs) / numpy.sum(numpy.exp(inputs))

def softMaxF():
    return softMax, None

def neuralNetworkError(expectedOutputs, predictedOutputs):
    return predictedOutputs - expectedOutputs

def negativeLogLikelihood(errors, expectedOutputs):
    return -numpy.log(numpy.sum(errors * expectedOutputs))

def trainNeuralNetwork(dataset: List[InputOutputSet], neuralNetwork, batchSize = 50, acceptedError = 0.7, numOfEpochs = 100):

    epoch = 1
    foundOptimalValues = False
    inputSize = len(dataset[0].inputs)
    batchSize = min(max(1, batchSize), inputSize)
    errors = []

    # execute while numOfEpochs is not exceeded and optimal value was not found yet
    while (numOfEpochs > epoch and not foundOptimalValues): #or lastValidationError > validationErrorAcceptedValue:

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
            neuralNetwork.updateWeights(batchSize)

            # add error to batch error list
            batchErrorList.append(neuralNetworkError)

        # when all batches are processed calculate average error
        errors.append(numpy.average(batchErrorList))

        # increase epoch
        epoch = epoch + 1

        # set optimal value flag
        foundOptimalValues = errors[-1] < acceptedError
        print("█", end='')
        sys.stdout.flush()


    print(epoch)
    plt.plot(errors)
    plt.show()

