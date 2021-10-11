from Data import Data
from Perceptron import Perceptron

if __name__ == "__main__":
    perceptron = Perceptron()

    epochs = 50
    epoch = 1
    containsErrors = True

    signals = Data.generatedSignals(Data.signals, 4)
    classes = Data.generatedClasses(5)
    weights = Data.weights

    while epoch < epochs and containsErrors:
        containsErrors = not perceptron.learningIteration(signals, classes, weights)
        epoch = epoch + 1
    
    perceptron.plotErrors()
    print("LEARNT IN " + str(epoch) + " EPOCH")
    print('w1 -> ' + str(weights[1]))
    print('w2 -> ' + str(weights[2]))
    print('w0 -> ' + str(weights[0]))

    testSignals = [[0.99, -0.01]]

    for s in testSignals:
        print(str(s) + " -> " + str(perceptron.defineClass(s, weights[-2:], weights[0])))

    