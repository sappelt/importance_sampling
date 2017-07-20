import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.semi_supervised import label_propagation
import random
from importance_sampling import ImportanceSampling
from sklearn.datasets import make_circles

class Sampler:
    def __init__(self):
        self.labels = []
        self.points = []
        self.sampledPoints = []

    def bernoulli(self, points, countLabels):
        self.labels = np.zeros(len(points))
        self.sampledPoints = []

        index = 0
        labeled = 0
        while labeled < countLabels and index < len(points):
            if random.random() < .5:
                self.labels[index] = 1
                labeled += 1
                self.sampledPoints.append(points[index])
            else:
                self.labels[index] = 0
            index += 1

        self.points = points
        return self.sampledPoints

    def importance(self, points, countLabels):
        importanceSampling = ImportanceSampling(stream=np.array([points[0]]), initial_points_m=2, initial_clusters=2)

        self.labels = np.zeros(len(points))
        self.sampledPoints = []

        index = 1
        labeled = 0
        while labeled < countLabels and index < len(points):
            if importanceSampling.addPoint(points[index]):
                self.labels[index] = 1
                labeled += 1
                self.sampledPoints.append(points[index])
            else:
                self.labels[index] = 0
            index += 1

        self.points = points
        return self.sampledPoints

    def plot(self):
        plt.scatter(self.points[:, 0], self.points[:, 1], color='darkorange',
                    marker='.', label='unlabeled')

        plt.scatter(self.points[self.labels == 1, 0], self.points[self.labels == 1, 1], color='blue',
                    label='sampled')
        plt.show()

#helper to generate circular shaped data
def generate_circles(countPoints, countCircles):
    iterations =  countCircles // 2
    samplesPer2Circles = countPoints // iterations
    allCircles = []
    for i in range(1, iterations+1):
        tempCircles, tempLabels = make_circles(n_samples=samplesPer2Circles, shuffle=False)
        if i == 1:
            allCircles = tempCircles*1.3
        else:
            allCircles = np.concatenate((allCircles, (tempCircles)*(i*1)))

    np.random.shuffle(allCircles)
    return allCircles

#generat a dataset with 10 radial clusters consisting of total 1000 points
circles = generate_circles(1000, 10)

#First test: Bernoulli sampling with p=0.5 and 50 points
bernoulliSampler = Sampler()
sampledPoints = np.array(bernoulliSampler.bernoulli(circles, 50))
print("Number of sampled points: ", len(sampledPoints))
bernoulliSampler.plot()

#Second test: Bernoulli sampling with p=0.5 and 100 points
bernoulliSampler = Sampler()
sampledPoints = np.array(bernoulliSampler.bernoulli(circles, 100))
print("Number of sampled points: ", len(sampledPoints))
bernoulliSampler.plot()

#Use importance sampler
importanceSampler = Sampler()
sampledPoints = np.array(importanceSampler.importance(circles, 100))
importanceSampler.plot()
print("Number of sampled points: ", len(sampledPoints))
