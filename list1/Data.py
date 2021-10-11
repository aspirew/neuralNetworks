import random

class Data:

    signals = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]

    def generatedSignals(signals, num):
        sig = []
        sig = sig + signals
        for _ in range(num):
            randSignal = signals[random.randint(0, 3)]
            sig.append((list(map(lambda s: s + random.uniform(-0.1, 0.1), randSignal[0])), randSignal[1]))
        return sig

    weights = [random.uniform(-1, 1) for _ in range(3)]
    
