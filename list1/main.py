from Adaline import Adaline
from Data import Data
from Perceptron import Perceptron
import random
import copy

if __name__ == "__main__":
    
    epochs = 1000
    epoch = 1
    quantityOfTestData = 30
    weightRange = 0.1
    learnRate = 0.01
    acceptedError = 0.3
    bias = None
    found = False
    functionUse = "bipolar" # unipolar (default) / bipolar

    perceptron = Perceptron(learnRate, functionUse)
    adaline = Adaline(learnRate, acceptedError)
    data = Data()

    print(f"GENERATING {quantityOfTestData} TEST DATA INSTANCES")
    print(f"USING {functionUse}")

    if functionUse == "bipolar":
        data.signals = data.signalsBipolar

    signals = data.generatedSignals(weightRange, -weightRange, quantityOfTestData)
    random.shuffle(signals)
    weights = data.weights(-weightRange, weightRange, bias)

    while epoch < epochs and not found:
        epoch = epoch + 1
        found = adaline.learningIteration(signals, weights)

    print("ADALINE LEARNT IN " + str(epoch) + " EPOCH")
    print('w1 -> ' + str(weights[1]))
    print('w2 -> ' + str(weights[2]))
    print('w0 -> ' + str(weights[0]))

    testSignals = [[0.99, -0.01], [0.1, 0.1], [0, 0.99], [1.001, 1.2]]

    print("\n")

    for s in testSignals:
        print(str(s) + " -> " + str(perceptron.defineClass(s, weights[-2:], weights[0])))
    