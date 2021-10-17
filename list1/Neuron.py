class Neuron:
    
    def scalarSum(self, signals, weights):
        return sum([s * w for s, w in zip(signals, weights)])

    def bipolarFunction(self, sum, bias):
        if (sum + bias) > 0:
            return 1
        return -1

    def learningIteration(self, signals, weights):
        pass