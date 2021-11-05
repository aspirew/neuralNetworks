import numpy

from NeuralNetworkExecutor import lossFunctionGradient, softMax

class NeuralNetworkLayer():

    def __init__(self, inputSize, numOfNeurons, learnRate, activationFun, weightsGenerationFun, weightsGenerationParams):
        self.numOfNeurons = numOfNeurons
        self.weightsArray = weightsGenerationFun(inputSize, numOfNeurons, *weightsGenerationParams)
        self.biasArray = weightsGenerationFun(1, numOfNeurons, *weightsGenerationParams)
        self.weightsBackup = numpy.copy(self.weightsArray)
        self.activationFun = activationFun()[0]
        self.activationFunDerivative = activationFun()[1]
        self.inputs = []
        self.outputs = []
        self.nextLayer: NeuralNetworkLayer = None
        self.derivative = None
        self.learnRate = learnRate
        self.error = None

    def calculateActivationVector(self, inputs):
        self.inputs = inputs
        multiplied = numpy.matmul(self.weightsArray, inputs)
        summed = numpy.add(multiplied, self.biasArray)
        if(self.nextLayer != None):
            self.outputs = self.activationFun(summed)
        else:
            self.outputs = summed #softmax
        return self.outputs

    def createWeightsBackup(self):
        self.weightsBackup = numpy.copy(self.weightsArray)

    def returnToPreviousWeights(self):
        self.weightsArray = numpy.copy(self.weightsBackup)

    def propagateForward(self, inputs):
        activationVector = self.calculateActivationVector(inputs)
        if(self.nextLayer != None):
            return self.nextLayer.propagateForward(activationVector)
        return self.outputs

    def propagateBackward(self, expectedOutputs, batchSize):
        if(self.nextLayer != None):
            self.nextLayer.propagateBackward(expectedOutputs, batchSize)
        if(self.nextLayer == None):
            self.derivative = lossFunctionGradient(self.outputs, expectedOutputs, self.inputs)
        else:
            self.derivative = self.activationFunDerivative(self.inputs)
        self.updateWeights(batchSize)

    def getRecursiveError(self):
        if(self.nextLayer != None):
            print(self.weightsArray.T)
            print(self.derivative)
            step = numpy.matmul(self.weightsArray.T, self.derivative)
            return step * self.nextLayer.getRecursiveError()
        return self.derivative
        
    def updateWeights(self, batchSize):
        print("der", self.derivative)
        print("input", self.inputs.T)
        print("steps", self.getRecursiveError())
        layerError = numpy.matmul(self.getRecursiveError(), self.inputs.T)
        self.weightsArray = self.weightsArray - ((self.learnRate / batchSize) * numpy.sum(layerError))

    def getLastLayer(self):
        if(self.nextLayer != None):
            return self.nextLayer.getLastLayer()
        return self


    