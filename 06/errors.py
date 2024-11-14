import numpy as np

class Errors:
    """ A class to handle different types of errors for neural networks."""
    def cross_entropy_error(self, y, t):
        """Calculates the cross-entropy error between the predicted and true labels."""
        delta = 1e-7
        batch_size = 1 if y.ndim == 1 else y.shape[0]

        return -np.sum(t*np.log(y + delta)) / batch_size