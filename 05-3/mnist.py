import mnist_data
import numpy as np
import pickle
import cv2 

class Mnist():
    """ Mnist class to load MNIST data and predict handwritten digits """
    def __init__(self):
        """ Initialize Mnist class and load MNIST data """
        self.data = mnist_data.MnistData()
        self.params = {}


    def sigmoid(self, x):
        """ Sigmoid function as activation function """
        return 1/(1 + np.exp(-x))


    def softmax(self, a):
        """ Softmax function to get the probability """
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a/np.sum(exp_a)
    

    def load(self):
        """ Load MNIST data """
        (x_train, y_train), (x_test, y_test) = self.data.load()
        return (x_train, y_train), (x_test, y_test)
    
    
    def init_network(self):
        """ Initialize the network weights """
        with open('model/sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)

    def load_handwritten_image(self, filename):
        """ Load an image and resize it to 28x28  """
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0
        img = img.flatten()  # Flatten the image to match MNIST data shape
        return img
    
    def predict_image(self, filename):
        """ Predict the digit from the image """
        x = self.load_handwritten_image(filename)
        y = self.predict(x)
        predicted_digit = np.argmax(y)  # Get the digit with the highest probability
        return predicted_digit

    def predict(self, x):
        """ Predict the digit from the input data """
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)

        return y    