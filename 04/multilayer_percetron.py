import numpy as np

class MultiLayerPerceptron:
    def __init__(self):
        # Initialize an empty dictionary to hold the network parameters (weights and biases)
        self.net = {}
        pass

    def init_network(self):
        # Initialize the neural network's weights and biases
        net = {}
        
        # Define weights and biases for layer 1 (input layer to hidden layer 1)
        net['w1'] = np.array([[0.7, 0.9, 0.3], [0.5, 0.4, 0.1]])  # Weights for connections from input to hidden layer
        net['b1'] = np.array([1, 1, 1])  # Biases for hidden layer 1
        
        # Define weights and biases for layer 2 (hidden layer 1 to hidden layer 2)
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]])  # Weights for connections from hidden layer 1 to hidden layer 2
        net['b2'] = np.array([0.5, 0.5])  # Biases for hidden layer 2
        
        # Define weights and biases for layer 3 (hidden layer 2 to output layer)
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]])  # Weights for connections from hidden layer 2 to output layer
        net['b3'] = np.array([0.1, 0.2])  # Biases for output layer

        # Store the initialized network parameters
        self.net = net

    def forward(self, x):
        # Perform a forward pass through the network
        
        # Retrieve weights and biases from the network parameters
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        # Calculate activations for the first layer
        a1 = np.dot(x, w1) + b1  # Weighted sum of inputs plus biases
        z1 = self.sigmoid(a1)  # Apply activation function (sigmoid) to the output of the first layer

        # Calculate activations for the second layer
        a2 = np.dot(z1, w2) + b2  # Weighted sum of the first layer's outputs plus biases
        z2 = self.sigmoid(a2)  # Apply activation function to the second layer's output

        # Calculate activations for the output layer
        a3 = np.dot(z2, w3) + b3  # Weighted sum of the second layer's outputs plus biases
        y = self.identity(a3)  # Apply identity function to the output (linear activation)

        return y  # Return the final output of the network
    
    def identity(self, x):
        # Identity activation function (no transformation)
        return x
    
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))  # Apply the sigmoid function to the input
