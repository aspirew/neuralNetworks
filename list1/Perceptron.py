from Data import Data
import matplotlib.pyplot as plt

class Perceptron:
    errors = []
    learnRate = 0.1

    def scalarSum(self, signals, weights):
        return sum([s * w for s, w in zip(signals, weights)])

    def unipolarFunction(self, sum, bias):
        if (sum + bias) > 0:
            return 1
        return 0

    def bipolarFunction(self, sum, bias):
        if (sum + bias) > 0:
            return 1
        return -1

    def defineClass(self, signals, weights, bias):
        return self.unipolarFunction(self.scalarSum(signals, weights), bias)

    def learningIteration(self, signals, classes, weights):
        for i in range(len(signals)):
            Z = self.defineClass(signals[i], weights[-2:], weights[0])
            self.errors.append(classes[i] - Z)
            weights[1] += self.learnRate * self.errors[-1] * signals[i][0]
            weights[2] += self.learnRate * self.errors[-1] * signals[i][1]
            weights[0] += self.learnRate * self.errors[-1]
        return all([e == 0 for e in self.errors[-4:]])

    def plotErrors(self):
        plt.plot(self.errors)
        plt.show()


    
