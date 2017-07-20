class Kernel:
    def __init__(self, kernel_matrix, eigenvalues, eigenvectors):
        self.kernel_matrix=kernel_matrix
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors