"""Multiple Kernel Learning (MKL) with convex kernel combination."""

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel


class MultipleKernelLearning:
    """Implements Multiple Kernel Learning with a convex combination of kernels."""

    def __init__(self, kernel_types=("rbf", "polynomial", "linear"), kernel_weights=None):
        self.kernel_types = list(kernel_types)
        self.n_kernels = len(self.kernel_types)

        if kernel_weights is None:
            self.kernel_weights = np.ones(self.n_kernels) / self.n_kernels
        else:
            self.kernel_weights = kernel_weights

    def compute_kernels(self, X, Y=None):
        """Compute individual kernel matrices."""
        if Y is None:
            Y = X

        kernels = []
        for kernel_type in self.kernel_types:
            if kernel_type == "rbf":
                K = rbf_kernel(X, Y, gamma=0.1)
            elif kernel_type == "polynomial":
                K = polynomial_kernel(X, Y, degree=3, gamma=0.1)
            elif kernel_type == "linear":
                K = linear_kernel(X, Y)
            else:
                raise ValueError(f"Unknown kernel: {kernel_type}")
            kernels.append(K)

        return kernels

    def compute_combined_kernel(self, X, Y=None):
        """Compute the convex combination of kernels."""
        kernels = self.compute_kernels(X, Y)
        combined_kernel = np.zeros_like(kernels[0])

        for kernel, weight in zip(kernels, self.kernel_weights):
            combined_kernel += weight * kernel

        return combined_kernel

    def optimize_weights(self, X, y, max_iter=100):
        """Optimize kernel weights using a simple kernel-alignment gradient step."""
        for _ in range(max_iter):
            yy = np.outer(y, y)

            gradients = []
            kernels = self.compute_kernels(X)
            for kernel in kernels:
                grad = np.trace(kernel @ yy)
                gradients.append(grad)

            gradients = np.array(gradients)
            self.kernel_weights += 0.01 * gradients
            self.kernel_weights = np.maximum(self.kernel_weights, 0)
            self.kernel_weights /= self.kernel_weights.sum()

        return self.kernel_weights
