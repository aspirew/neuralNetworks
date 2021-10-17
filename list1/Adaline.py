import matplotlib.pyplot as plt

from Neuron import Neuron

class Adaline(Neuron):

    def __init__(self, learnRate, acceptedError):
        self.learnRate = learnRate        
        self.acceptedError = acceptedError
        self.errors = []
        self.squaredErrors = []

    def meanSquaredError(self):
        result = sum(self.errors) / len(self.errors)
        self.squaredErrors.append(result)
        return result

    def learningIteration(self, signals, weights):

        for s in signals:
            error = s[1] - self.scalarSum([1] + s[0], weights)
            self.errors.append(error ** 2)
            errorAcceptable = self.meanSquaredError() < self.acceptedError
            
            if not errorAcceptable:
                weights[1] += 2 * self.learnRate * error * s[0][0]
                weights[2] += 2 * self.learnRate * error * s[0][1]
                weights[0] += 2 * self.learnRate * error

        return errorAcceptable

    def plotErrors(self):
        plt.plot(self.squaredErrors)
        plt.show()



    

             
                