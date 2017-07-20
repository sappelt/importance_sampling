import math

def rbf(x, y, sigma):
    distances = 0
    sigmaSquared = math.pow(sigma, 2)

    for i in range(0, len(x)):
        distances += math.pow(x[i] - y[i], 2)

    return math.exp(-(distances / (2*sigmaSquared)))