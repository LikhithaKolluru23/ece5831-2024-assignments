import numpy as np

class Activations:
    """ A class to handle different types of activations for neural networks."""
    def sigmoid(self, x):
        """Sigmoid activation function implementation"""
        return 1/(1 + np.exp(-x))
    
    # for multi-dimensional x
    def softmax(self, x):
        """Softmax activation function implementation"""
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x)  
        return np.exp(x) / np.sum(np.exp(x))