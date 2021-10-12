from Data import Data
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, learnRate, fun):
        if(fun == "bipolar"):
            self.usedFun = self.bipolarFunction
        else:
            self.usedFun = self.unipolarFunction
        self.learnRate = learnRate        

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
        return self.usedFun(self.scalarSum(signals, weights), bias)

    def learningIteration(self, signals, weights):
        errorAppeared = False

        for s in signals:
            Z = self.defineClass(s[0], weights[-2:], weights[0])
            error = s[1] - Z
            if error != 0:
                errorAppeared = True 
                weights[1] += self.learnRate * error * s[0][0]
                weights[2] += self.learnRate * error * s[0][1]
                weights[0] += self.learnRate * error
                
        return errorAppeared

    def plotErrors(self):
        plt.plot(self.errors)
        plt.show()


    
