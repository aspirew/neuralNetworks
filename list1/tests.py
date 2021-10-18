from Neuron import Neuron
from Data import Data
from Perceptron import Perceptron
from Adaline import Adaline
import copy

def manualBiasTest(neuron: Neuron):
    epochs = 1000
    bias = -1
    data = Data()
    signals = data.predefinedSignals

    for i in range(11):
        weights = copy.copy(data.predefinedWeights)
        weights[0] = bias + 0.2 * i
        epoch = 1
        found = False
        while epoch < epochs and not found:
            epoch = epoch + 1
            found = neuron.learningIteration(signals, weights)
        
        print(epoch)

def dynamicWeightsTest(neuron: Neuron):
    epochs = 1000
    data = Data()
    weightRange = 0.1
    signals = data.predefinedSignals
    

    for i in range(10):
        reqEpochs = []
        for _ in range(10):
            weights = data.weights(-weightRange - i * 0.1 , weightRange + i * 0.1)
            epoch = 1
            found = False
            while epoch < epochs and not found:
                epoch = epoch + 1
                found = neuron.learningIteration(signals, weights)
            reqEpochs.append(epoch)
            
        print(sum(reqEpochs) / len(reqEpochs))


def learnRateTest(neuron: Neuron):
    epochs = 1000
    data = Data()
    signals = data.predefinedSignals
    
    for i in range(10):

        neuron.learnRate = 0.005 * (i * 3 + 1)
        weights = copy.copy(data.predefinedWeights)
        epoch = 1
        found = False
        while epoch < epochs and not found:
            epoch = epoch + 1
            found = neuron.learningIteration(signals, weights)
        
        print(0.005 * (i * 3 + 1), epoch)


def acceptedErrorCheck(adaline: Adaline):
    epochs = 1000
    data = Data()
    signals = data.predefinedSignals

    weights = data.weights(-0.1, 0.1)
    epoch = 1
    found = False
    while epoch < epochs and not found:
        epoch = epoch + 1
        found = adaline.learningIteration(signals, weights)

    adaline.plotErrors()
    

if __name__ == "__main__":
    # manualBiasTest(Perceptron(0.01, "bipolar"))
    # manualBiasTest(Perceptron(0.01, "unipolar"))
    # manualBiasTest(Adaline(0.01, 0.3))
    # dynamicWeightsTest(Perceptron(0.01, "bipolar"))
    # dynamicWeightsTest(Perceptron(0.01, "unipolar"))
    # dynamicWeightsTest(Adaline(0.01, 0.3))
    # learnRateTest(Perceptron(0.01, "bipolar"))
    # learnRateTest(Perceptron(0.01, "unipolar"))
    learnRateTest(Adaline(0.01, 0.3))
    # acceptedErrorCheck(Adaline(0.01, 0))
    # acceptedErrorCheck(Adaline(0.01, 0.3))







