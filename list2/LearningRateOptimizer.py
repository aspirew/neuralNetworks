from copy import copy, deepcopy
import numpy
from NeuralNetworkLayer import NeuralNetworkLayer


def momentum(neuralNetwork: NeuralNetworkLayer, batchSize):
    momentumRate = 0.8

    # take last change and multiply with momentum rate
    weightsIncrease = momentumRate * neuralNetwork.lastWeightsArrayChange
    biasIncrease = momentumRate * neuralNetwork.lastBiasArrayChange

    # from such momentum substract current value change
    neuralNetwork.lastWeightsArrayChange = weightsIncrease + (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
    neuralNetwork.lastBiasArrayChange = biasIncrease + (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue

    # update weights according to calculated change
    neuralNetwork.weightsArray = neuralNetwork.weightsArray - neuralNetwork.lastWeightsArrayChange
    neuralNetwork.biasArray = neuralNetwork.biasArray - neuralNetwork.lastBiasArrayChange

def nestrovMomentum(neuralNetwork: NeuralNetworkLayer, batchSize):
    momentumRate = 0.8

    weightsIncrease = momentumRate * neuralNetwork.lastWeightsArrayChange
    biasIncrease = momentumRate * neuralNetwork.lastBiasArrayChange

    # calculate change according to prediction
    neuralNetwork.lastWeightsArrayChange = weightsIncrease + (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValuePrediction
    neuralNetwork.lastBiasArrayChange = biasIncrease + (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValuePrediction

    # update weights according to calculated change
    neuralNetwork.weightsArray = neuralNetwork.weightsArray - neuralNetwork.lastWeightsArrayChange
    neuralNetwork.biasArray = neuralNetwork.biasArray - neuralNetwork.lastBiasArrayChange

def adagrad(neuralNetwork: NeuralNetworkLayer, batchSize):
    eps = 0.00000001

    if(len(neuralNetwork.weightsGradientSum) > 0):
        neuralNetwork.weightsGradientSum += neuralNetwork.weightsUpdateValue
        neuralNetwork.biasGradientSum += neuralNetwork.biasGradientSum
    else:
        neuralNetwork.weightsGradientSum = copy(neuralNetwork.weightsUpdateValue)
        neuralNetwork.biasGradientSum = copy(neuralNetwork.biasUpdateValue)

    neuralNetwork.weightsArray = neuralNetwork.weightsArray - ((neuralNetwork.learnRate / batchSize) / numpy.sqrt(neuralNetwork.weightsGradientSum ** 2 + eps)) * neuralNetwork.weightsUpdateValue
    neuralNetwork.biasArray = neuralNetwork.biasArray - ((neuralNetwork.learnRate / batchSize) / numpy.sqrt(neuralNetwork.biasGradientSum ** 2 + eps)) * neuralNetwork.biasUpdateValue

def adadelta(neuralNetwork: NeuralNetworkLayer, batchSize):
    eps = 0.00000001
    gamma = 0.9

    if(len(neuralNetwork.lastLastWeightsArrayChangePrediction) > 0):
        neuralNetwork.lastWeightsArrayChangePrediction = gamma * neuralNetwork.lastLastWeightsArrayChangePrediction + (1 - gamma) * neuralNetwork.lastWeightsArrayChange
        neuralNetwork.lastBiasArrayChangePrediction = gamma * neuralNetwork.lastLastBiasArrayChangePrediction + (1 - gamma) * neuralNetwork.lastBiasArrayChange
    
    if(len(neuralNetwork.lastWeightsArrayChange) > 0):
        neuralNetwork.lastLastWeightsArrayChangePrediction = copy(neuralNetwork.lastWeightsArrayChangePrediction)
        neuralNetwork.lastLastBiasArrayChangePrediction = copy(neuralNetwork.lastBiasArrayChangePrediction)

    else:
        neuralNetwork.lastWeightsArrayChange = - (numpy.sqrt(neuralNetwork.lastLastWeightsArrayChangePrediction + eps) / numpy.sqrt((neuralNetwork.weightsUpdateValue ** 2) + eps)) * neuralNetwork.weightsUpdateValue

        neuralNetwork.lastWeightsArrayChange = - (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
        neuralNetwork.lastBiasArrayChange = - (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue

        neuralNetwork.weightsArray = neuralNetwork.weightsArray + neuralNetwork.lastWeightsArrayChange
        neuralNetwork.biasArray = neuralNetwork.biasArray + neuralNetwork.lastBiasArrayChange