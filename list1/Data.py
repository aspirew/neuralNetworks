import random

class Data:

    def __init__(self, max, min, quantity):
        self.randomWeightMax = max
        self.randomWeightMin = min
        self.quantity = quantity
        
    signals = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]

    signalsBipolar = list(map(lambda s: (s[0], -1) if s[1] == 0 else s, signals))

    def generatedSignals(self):
        sig = []
        sig = sig + self.signals
        for _ in range(self.quantity):
            randSignal = self.signals[random.randint(0, 3)]
            sig.append((list(map(lambda s: s + random.uniform(-0.1, 0.1), randSignal[0])), randSignal[1]))
        return sig

    def weights(self, threshold):
        if(threshold):
            return [threshold] + [random.uniform(-1, self.randomWeightMax) for _ in range(2)]
        return [random.uniform(-1, self.randomWeightMax) for _ in range(3)]
