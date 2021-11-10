import numpy

from NeuralNetworkExecutor import neuralNetworkError

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
        self.weightsUpdateValue = []
        self.biasUpdateValue = []

    def calculateActivationVector(self, inputs):

        # set inputs of current layer
        self.inputs = inputs

        # multiply weights array with inputs array
        multiplied = numpy.matmul(self.weightsArray, inputs)

        # sum result with bias array
        summed = numpy.add(multiplied, self.biasArray)

        # set outputs of current layer
        self.outputs = self.activationFun(summed)
        return self.outputs

    def createWeightsBackup(self):
        self.weightsBackup = numpy.copy(self.weightsArray)

    def returnToPreviousWeights(self):
        self.weightsArray = numpy.copy(self.weightsBackup)

    def propagateForward(self, inputs):
        # calculate activation vector of current layer
        activationVector = self.calculateActivationVector(inputs)

        # calculate activation vector of next layer (if such exists) using current layer activation vector as input
        if(self.nextLayer != None):
            return self.nextLayer.propagateForward(activationVector)
        
        # return neural network outputs
        return activationVector

    def propagateBackward(self, expectedOutputs):
        # if next layer exists
        if(self.nextLayer != None):
            # get error of next layer
            err = self.nextLayer.propagateBackward(expectedOutputs)

            # get derivative of current layer
            self.derivative = self.activationFunDerivative(self.outputs)

            # each label iteration increases weight change (they are summed). Total change will be calculated later
            self.saveWeightChange()
            return err
        # calculate error of last layer <predictedOutputs - expectedOutputs>
        self.error = neuralNetworkError(expectedOutputs, self.outputs)
        # save weight change for last layer
        self.saveWeightChange()
        return self.error

    def getRecursiveError(self):
        # get error of current layer
        if(self.nextLayer != None):
            # transpose weights of next layer and multiply with error of next layer
            # take hadamard product of such calculation with derivative of current layer
            # this is step of chain rule calculation which gives current layer error
            self.error = numpy.matmul(self.nextLayer.weightsArray.T, self.nextLayer.error) * self.derivative
            return self.error
        return self.error

    def saveWeightChange(self):
        # get error of current layer: getRecursiveError uses the chain rule to calculate error of current layer
        layerError = self.getRecursiveError()

        # calculate update value: current layer multiplied with transposed current inputs
        updateVal = numpy.matmul(layerError, self.inputs.T)
        # later weight update value is increased, or created
        if(len(self.weightsUpdateValue) > 0):
            self.weightsUpdateValue += updateVal
            self.biasUpdateValue += layerError
        else:
            self.weightsUpdateValue = updateVal
            self.biasUpdateValue = layerError

    def updateWeights(self, batchSize):

        # backpropagation with each label increased weights update value.
        # this operation was repeated batch size times
        # change of weights array and bias array is divided by batch size to get average change
        # and is multiplied with learn rate
        # this value is then substracted from arrays

        if(self.nextLayer != None):
            self.nextLayer.updateWeights(batchSize)

        self.weightsArray = self.weightsArray - ((self.learnRate / batchSize) * self.weightsUpdateValue)
        self.biasArray = self.biasArray - ((self.learnRate / batchSize) * self.biasUpdateValue)
        
        # after that operation weights must be cleaned so next epoch can gather new changes
        self.weightsUpdateValue = []
        self.biasUpdateValue = []

    def getLastLayer(self):
        if(self.nextLayer != None):
            return self.nextLayer.getLastLayer()
        return self


    