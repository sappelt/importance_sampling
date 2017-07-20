import random
import numpy as np
import rbf
import math
import kernel
from cluster import Cluster

class ImportanceSampling:
    def __init__(self, stream, initial_clusters=1, kernelFunction=rbf.rbf, initial_points_m=2):
        self.stream = stream
        self.initial_clusters = initial_clusters
        self.kernelFunction = kernelFunction
        self.initial_points_m = initial_points_m
        self.kernel = kernel.Kernel(np.array([[1]], dtype=float), np.array([1], dtype=float), np.array([[1]], dtype=float))
        self.clusters = []
        for i in range(0, initial_clusters):
            self.clusters.append(Cluster())

    def addPoint(self, point):

        #Initialize the stream
        if len(self.stream) < self.initial_points_m:
            self.kernel = self._reCalculateKernel(point)
            self.stream = np.vstack((self.stream, point))

            #Point sampled
            return True

            #Usually you would cluster here
            #This is missing yet
        else:
            #Calculcate probability pt of sampling point and check if point will be sampled
            if self._probability(self._reCalculateKernel(point)) >= random.random():
                #Recalculate the kernel matrix
                self.kernel = self._reCalculateKernel(point)

                #Add point to stream
                self.stream = np.vstack((self.stream, point))

                #Point sampled
                return True

            #Point not sampled
            return False

    def _probability(self, kernel):
        # Calculate probability of adding a point
        # Take last row of eigenvectors and calculate l2 norm
        l2_norm = self._row_l2_norm(kernel.eigenvectors[kernel.eigenvectors.shape[0] - 1, 0:len(self.clusters)])
        print("l2_norm:", l2_norm)
        probability = (1/len(self.clusters))*l2_norm
        print("probability: ", probability);
        return probability

    def _row_l2_norm(self, row):
        return math.pow(np.linalg.norm(row), 2)

    def _reCalculateKernel(self, point):
        matrix_old = np.copy(self.kernel.kernel_matrix)
        kernel_xt = self.kernelFunction(point, point, 1.5)

        # Calculate gamma
        # = Distances of point_xt to every point in stream
        gamma = np.array([[]], dtype=float)
        for i in range(0, len(self.stream)):
            gamma = np.hstack((gamma, [[self.kernelFunction(point, self.stream[i], 1.5)]]))

        # Calculate gamma transposed
        gamma_transposed = np.transpose(gamma)

        # Generate new matrix K
        old_length = matrix_old.shape[0]
        matrix_size = old_length + 1
        matrix_new = np.zeros((matrix_size, matrix_size))

        # Copy old content
        for i in range(0, old_length):
            for j in range(0, old_length):
                matrix_new[i, j] = matrix_old[i, j]

        matrix_new[0:matrix_size - 1, old_length:matrix_size] = gamma_transposed
        matrix_new[old_length:matrix_size, 0:matrix_size - 1] = gamma
        matrix_new[matrix_size - 1, matrix_size - 1] = kernel_xt

        print(self.stream)
        print(matrix_new)
        print()

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix_new)
        print(eigenvalues)
        print(eigenvectors)

        # Save only the first C=cluster size eigenvectors and eigenvalues
        return kernel.Kernel(matrix_new, eigenvalues, eigenvectors)