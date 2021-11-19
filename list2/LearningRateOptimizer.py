import numpy
from NeuralNetworkLayer import NeuralNetworkLayer


def momentum(neuralNetwork: NeuralNetworkLayer, batchSize):
    momentumRate = 0.8

    # if it is first change, one has to be made so it can be used later
    if(len(neuralNetwork.lastWeightsArrayChange) < 1):
        neuralNetwork.lastWeightsArrayChange = (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
        neuralNetwork.lastBiasArrayChange = (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue

    # if there is a change momentum can be calculated
    else:
        # take last change and multiply with momentum rate
        weightsIncrease = momentumRate * neuralNetwork.lastWeightsArrayChange
        biasIncrease = momentumRate * neuralNetwork.lastBiasArrayChange

        # from such momentum substract current value change
        neuralNetwork.lastWeightsArrayChange = weightsIncrease - (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
        neuralNetwork.lastBiasArrayChange = biasIncrease -+ (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue
    
    # update weights according to calculated change
    neuralNetwork.weightsArray = neuralNetwork.weightsArray + neuralNetwork.lastWeightsArrayChange
    neuralNetwork.biasArray = neuralNetwork.biasArray + neuralNetwork.lastBiasArrayChange


def nestrovMomentum(neuralNetwork: NeuralNetworkLayer, batchSize):
    momentumRate = 0.8

    # if it is first change, one has to be made so it can be used later
    if(len(neuralNetwork.lastWeightsArrayChange) < 1):
        neuralNetwork.lastWeightsArrayChange = (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
        neuralNetwork.lastBiasArrayChange = (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue

    # if there is a change, momentum can be calculated
    else:
        # take last change and multiply with momentum rate
        weightsIncrease = momentumRate * neuralNetwork.lastWeightsArrayChange
        biasIncrease = momentumRate * neuralNetwork.lastBiasArrayChange

        # predict value according to last change increases
        predictedWeightChange, predictedBiasChange = neuralNetwork.predictChange(weightsIncrease, biasIncrease, batchSize)

        # calculate change according to prediction
        neuralNetwork.lastWeightsArrayChange = weightsIncrease + (neuralNetwork.learnRate / batchSize) * predictedWeightChange
        neuralNetwork.lastBiasArrayChange = biasIncrease + (neuralNetwork.learnRate / batchSize) * predictedBiasChange
    
    # update weights according to calculated change
    neuralNetwork.weightsArray = neuralNetwork.weightsArray - neuralNetwork.lastWeightsArrayChange
    neuralNetwork.biasArray = neuralNetwork.biasArray - neuralNetwork.lastBiasArrayChange

def adagrad(neuralNetwork: NeuralNetworkLayer, batchSize):
        if(len(neuralNetwork.weightsGradientSum) < 1):
            neuralNetwork.weightsGradientSum = (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
            neuralNetwork.biasGradientSum = (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue
        else:
            neuralNetwork.weightsGradientSum = neuralNetwork.weightsGradientSum + (neuralNetwork.learnRate / batchSize) * neuralNetwork.weightsUpdateValue
            neuralNetwork.biasGradientSum = neuralNetwork.biasGradientSum + (neuralNetwork.learnRate / batchSize) * neuralNetwork.biasUpdateValue

        neuralNetwork.learnRate = neuralNetwork.learnRate / numpy.sqrt(neuralNetwork.weightsGradientSum ** 2)
        # after that operation weights update value must be cleaned so next epoch can gather new changes
        neuralNetwork.weightsUpdateValue = []
        neuralNetwork.biasUpdateValue = []