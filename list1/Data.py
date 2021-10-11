import random

class Data:

    signals = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    classes = [
        0, 0, 0, 1
    ]

    def generatedSignals(signals, num):
        sig = []
        sig = sig + signals
        for _ in range(num):
            for s in signals:
                rand = random.uniform(-0.1, 0.1)
                sig.append([s[0] + rand, s[1] + rand])
        return sig

    def generatedClasses(num):
        return [0, 0, 0, 1] * num

    weights = [random.uniform(-1, 1) for _ in range(3)]
    
