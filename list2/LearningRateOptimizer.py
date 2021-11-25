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

    neuralNetwork.optimizerWeights += (neuralNetwork.weightsUpdateValue / batchSize) ** 2
    neuralNetwork.optimizerBias += (neuralNetwork.biasUpdateValue / batchSize) ** 2

    neuralNetwork.lastWeightsArrayChange = neuralNetwork.learnRate / batchSize * neuralNetwork.weightsUpdateValue
    neuralNetwork.lastBiasArrayChange = neuralNetwork.learnRate / batchSize * neuralNetwork.biasUpdateValue

    neuralNetwork.weightsArray = neuralNetwork.weightsArray - (neuralNetwork.lastWeightsArrayChange / (numpy.sqrt(neuralNetwork.optimizerWeights) + eps))
    neuralNetwork.biasArray = neuralNetwork.biasArray - (neuralNetwork.lastBiasArrayChange / (numpy.sqrt(neuralNetwork.optimizerBias) + eps))


def adadelta(neuralNetwork: NeuralNetworkLayer, batchSize):
    eps = 0.00000001
    gamma = 0.9

    neuralNetwork.adadelta_w_s = gamma * neuralNetwork.adadelta_w_s + (1 - gamma) * (neuralNetwork.weightsUpdateValue ** 2)
    neuralNetwork.adadelta_b_s = gamma * neuralNetwork.adadelta_b_s + (1 - gamma) * (neuralNetwork.biasUpdateValue ** 2)

    neuralNetwork.weightsUpdateValue *= numpy.sqrt(neuralNetwork.adadelta_w_d + eps) / numpy.sqrt(neuralNetwork.adadelta_w_s + eps)
    neuralNetwork.biasUpdateValue *= numpy.sqrt(neuralNetwork.adadelta_b_d + eps) / numpy.sqrt(neuralNetwork.adadelta_b_s + eps)

    neuralNetwork.adadelta_w_d = gamma * neuralNetwork.adadelta_w_d + (1 - gamma) * (neuralNetwork.weightsUpdateValue ** 2)
    neuralNetwork.adadelta_b_d = gamma * neuralNetwork.adadelta_b_d + (1 - gamma) * (neuralNetwork.biasUpdateValue ** 2)


def adam(neuralNetwork: NeuralNetworkLayer, batchSize):
    eps = 0.00000001
    beta1 = 0.9
    beta2 = 0.999

    update = neuralNetwork.weightsUpdateValue / batchSize
    updateBias = neuralNetwork.biasUpdateValue / batchSize

    neuralNetwork.optimizerWeights_m = beta1 * neuralNetwork.optimizerWeights_m + (1-beta1) * update
    neuralNetwork.optimizerWeights = beta2 * neuralNetwork.optimizerWeights + (1-beta2) * (update ** 2)

    neuralNetwork.optimizerBias_m = beta1 * neuralNetwork.optimizerBias_m + (1-beta1) * updateBias
    neuralNetwork.optimizerBias = beta2 * neuralNetwork.optimizerBias + (1-beta2) * (updateBias ** 2)

    weightsCorrection_m = neuralNetwork.optimizerWeights_m / (1 - numpy.power(beta1, neuralNetwork.batchEpochNum))
    weightsCorrection_v = neuralNetwork.optimizerWeights / (1 - numpy.power(beta2, neuralNetwork.batchEpochNum))

    biasCorrection_m = neuralNetwork.optimizerBias_m / (1 - numpy.power(beta1, neuralNetwork.batchEpochNum))
    biasCorrection_v = neuralNetwork.optimizerBias / (1 - numpy.power(beta2, neuralNetwork.batchEpochNum))

    neuralNetwork.batchEpochNum += 1

    neuralNetwork.weightsArray = neuralNetwork.weightsArray - (neuralNetwork.learnRate * weightsCorrection_m) / (numpy.sqrt(weightsCorrection_v) + eps)
    neuralNetwork.biasArray = neuralNetwork.biasArray - (neuralNetwork.learnRate * biasCorrection_m) / (numpy.sqrt(biasCorrection_v) + eps)