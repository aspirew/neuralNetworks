from Neuron import Neuron
from Data import Data
from Perceptron import Perceptron
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
        print(weights)
        while epoch < epochs and not found:
            epoch = epoch + 1
            found = neuron.learningIteration(signals, weights)
        
        print(bias + 0.2 * i, epoch)




if __name__ == "__main__":
    manualBiasTest(Perceptron(0.1, "bipolar"))



