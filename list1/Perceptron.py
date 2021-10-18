from copy import copy
from Neuron import Neuron


class Perceptron(Neuron):

    def __init__(self, learnRate, fun):
        if(fun == "bipolar"):
            self.usedFun = self.bipolarFunction
        else:
            self.usedFun = self.unipolarFunction
        self.learnRate = learnRate        

    def unipolarFunction(self, sum, bias):
        if (sum + bias) > 0:
            return 1
        return 0

    def defineClass(self, signals, weights, bias):
        return self.usedFun(self.scalarSum(signals, weights), bias)

    def learningIteration(self, signals, weights):
        found = True
        sigs = copy(signals)
        
        if(self.usedFun == self.unipolarFunction):
            sigs = list(map(lambda s: (s[0], 0) if s[1] == -1 else s, signals))

        for s in sigs:
            Z = self.defineClass(s[0], weights[-2:], weights[0])
            error = s[1] - Z
            if error != 0:
                found = False 
                weights[1] += self.learnRate * error * s[0][0]
                weights[2] += self.learnRate * error * s[0][1]
                weights[0] += self.learnRate * error
                
        return found


    
