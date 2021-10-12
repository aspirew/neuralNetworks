from Data import Data
from Perceptron import Perceptron

if __name__ == "__main__":
    
    epochs = 50
    epoch = 1
    quantityOfTestData = 10
    learnRate = 0.1
    bias = 0.5
    containsErrors = True
    functionUse = "bipolar" # unipolar (default) / bipolar

    perceptron = Perceptron(learnRate, functionUse)
    data = Data(0.1, -0.1, quantityOfTestData)

    print(f"GENERATING {quantityOfTestData} TEST DATA INSTANCES")
    print(f"USING {functionUse}")

    if functionUse == "bipolar":
        data.signals = data.signalsBipolar

    weights = data.weights(bias)

    while epoch < epochs and containsErrors:
        containsErrors = perceptron.learningIteration(data.signals, weights)
        epoch = epoch + 1
    
    print("LEARNT IN " + str(epoch) + " EPOCH")
    print('w1 -> ' + str(weights[1]))
    print('w2 -> ' + str(weights[2]))
    print('w0 -> ' + str(weights[0]))

    testSignals = [[0.99, -0.01], [0.1, 0.1], [0, 0.99], [1.2, 1.01]]

    for s in testSignals:
        print(str(s) + " -> " + str(perceptron.defineClass(s, weights[-2:], weights[0])))

    