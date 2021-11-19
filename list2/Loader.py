import numpy
from mnist import MNIST
from dataclasses import dataclass
from copy import copy

@dataclass
class InputOutputSet:
    inputs: numpy
    output: int

    def getOutputAsArgmax(self):
        xs = [[0]] * 10
        xs[self.output] = [1]
        return xs

learnData = None
# learnDataCopy = None
testData = None

def loadData():
    global learnData
    global learnDataCopy
    global testData

    print("Loading MNIST data...")
    data = MNIST('./mnist_data')
    data.gz = True
    images, labels = data.load_training()
    testImages, testLabels = data.load_testing()
    print("MNIST data loaded.")
    print("Generating sets...")
    learnData = [InputOutputSet(numpy.array([inputs]).T / 255, label) for inputs, label in zip(images, labels)]
    # learnDataCopy = copy(learnData)
    testData = [InputOutputSet(numpy.array([inputs]).T / 255, label) for inputs, label in zip(testImages, testLabels)]

    print("Sets generated.")
    return learnData, testData

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
