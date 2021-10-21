import numpy

class NeuralNetworkLayer():

    def __init__(self, inputSize, numOfNeurons, weightsGenerationFun, weightsGenerationParams):
        self.numOfNeurons = numOfNeurons
        self.weightsArray = weightsGenerationFun(inputSize, numOfNeurons, *weightsGenerationParams)
        self.biasArray = weightsGenerationFun(1, numOfNeurons, *weightsGenerationParams)
        self.weightsBackup = numpy.copy(self.weightsArray)

    def calculateActivationVector(self, inputs, activationFun):
        multiplied = numpy.matmul(self.weightsArray, inputs)
        summed = numpy.add(multiplied, self.biasArray)
        return activationFun(summed)

    def createWeightsBackup(self):
        self.weightsBackup = numpy.copy(self.weightsArray)

    def returnToPreviousWeights(self):
        self.weightsArray = numpy.copy(self.weightsBackup)

    