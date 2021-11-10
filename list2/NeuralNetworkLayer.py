import numpy

from NeuralNetworkExecutor import neuralNetworkError, softMax

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
        self.inputs = inputs
        multiplied = numpy.matmul(self.weightsArray, inputs)
        summed = numpy.add(multiplied, self.biasArray)
        self.outputs = self.activationFun(summed)
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

    def propagateBackward(self, expectedOutputs):
        if(self.nextLayer != None):
            err = self.nextLayer.propagateBackward(expectedOutputs)
            self.derivative = self.activationFunDerivative(self.outputs)
            self.saveWeightChange(expectedOutputs)
            return err
        self.saveWeightChange(expectedOutputs)
        return self.error

    def getRecursiveError(self, expectedOutputs):
        # print("WEIGHTS: ", self.weightsArray.T)
        # print("DER: ", self.derivative)
        if(self.nextLayer != None):
            # print("MULTIPLYING WEIGHTS OF ANOTHER LAYER")
            # print(self.nextLayer.weightsArray.T)
            # print("WITH ERROR OF ANOTHER LAYER")
            # print(self.nextLayer.error)
            # print("AND HADAMADA WITH MY DERIVATIVE")
            # print(self.derivative)
            self.error = numpy.matmul(self.nextLayer.weightsArray.T, self.nextLayer.error) * self.derivative
            return self.error
        self.error = neuralNetworkError(expectedOutputs, self.outputs)
        return self.error

    def saveWeightChange(self, expectedOutputs):
        layerError = self.getRecursiveError(expectedOutputs)
        updateVal = numpy.matmul(layerError, self.inputs.T)
        if(len(self.weightsUpdateValue) > 0):
            self.weightsUpdateValue += updateVal
            self.biasUpdateValue += layerError
        else:
            self.weightsUpdateValue = updateVal
            self.biasUpdateValue = layerError

    def updateWeights(self, batchSize):
        # print("MY INPUTS: ", self.inputs)
        # print("MY OUTPUTS: ", self.outputs)
        # print("MY WEIGHTS: ", self.weightsArray)
        # print("EXPECTED OUTPUTS: ", expectedOutputs)
        # self.createWeightsBackup()
        # # print("der", self.derivative)
        # # print("input", self.inputs.T)
        # layerError = self.getRecursiveError(expectedOutputs)
        # # print("error", self.getRecursiveError())
        # print("ERROR OF MY LAYER IS: ")
        # print(layerError)
        # print("I WANT TO MULTIPLY IT WITH MY TRANSPOSED INPUTS")
        # print(self.inputs.T)
        # updateValue = numpy.matmul(layerError, self.inputs.T)
        # print("AND I GET UPDATE VALUE")
        # print(updateValue)
        # print("SUM!")
        # print(numpy.sum(updateValue))
        # print("WEIGHTS BEFORE UPDATE")
        # print(self.weightsArray)
        self.weightsArray = self.weightsArray - ((self.learnRate / batchSize) * self.weightsUpdateValue)
        self.biasArray = self.biasArray - ((self.learnRate / batchSize) * self.biasUpdateValue)
        self.weightsUpdateValue = []
        self.biasUpdateValue = []

    def getLastLayer(self):
        if(self.nextLayer != None):
            return self.nextLayer.getLastLayer()
        return self


    