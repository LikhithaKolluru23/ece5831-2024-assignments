from activations import Activations
import numpy as np
from layers import Affine, Relu, SoftmaxWithLoss
from errors import Errors
from collections import OrderedDict

class TwoLayerNetWithBackProp:
    """
    A two-layer neural network with backpropagation.
    The network has:
        - One hidden layer with ReLU activation.
        - One output layer with softmax and cross-entropy loss.
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """ Initializes the two-layer neural network with specified sizes for the input,
        hidden, and output layers. It also initializes the weights and biases. """
        self.params = {}

        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.activations = Activations()
        self.errors = Errors()

        # add layers
        self.layers = OrderedDict()
        self.update_layers()

        self.last_layer = SoftmaxWithLoss()

    def update_layers(self):
        """
        Updates the layers in the network. It includes:
            - An affine layer (linear transformation) followed by ReLU for the hidden layer.
            - An affine layer for the output layer.
        """
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])   

        self.layers['Rele1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
    
    

    def predict(self, x):
        """ Makes predictions by passing the input through the network."""
        ## new implementation for backprop
        for layer in self.layers.values():
            x = layer.forward(x)

        y = x
        return y
    
    def loss(self, x, y):
        """  Calculates the loss for a given input and true label."""
        y_hat = self.predict(x)

        # return self.errors.cross_entropy_error(y_hat, y)
        return self.last_layer.forward(y_hat, y)

    def accuracy(self, x, y):
        """ Calculates the accuracy of the model for a given input and true label."""
        y_hat = self.predict(x)
        p = np.argmax(y_hat, axis=1)
        y_p = np.argmax(y, axis=1)

        return np.sum(p == y_p)/float(x.shape[0])
    

    def gradient(self, x, y):
        """ Computes the gradient of the loss function with respect to the weights and biases."""
        self.loss(x, y)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
      
        return grads