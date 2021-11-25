import numpy

def generateWeights(function, parameters):
    return function(*parameters)

def normalDistribution(columns, rows, mu, sigma):
    array = numpy.empty((0, columns), int)
    for _ in range(rows) : array = numpy.append(array, numpy.array([numpy.random.normal(mu, sigma, columns)]), axis=0)
    return array

def xavier(columns, rows, nextLayerNeurons):
    variance = 2 / (columns + nextLayerNeurons)
    return normalDistribution(columns, rows, 0, numpy.sqrt(variance))

def he(columns, rows):
    variance = 2 / (columns)
    return normalDistribution(columns, rows, 0, numpy.sqrt(variance))
