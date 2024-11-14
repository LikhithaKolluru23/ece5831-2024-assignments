import numpy as np
import pickle
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
from mnist_data import MnistData

# Hyperparameters
iterations = 10000
batch_size = 16
learning_rate = 0.01

"""
    Main function to train a two-layer neural network on the MNIST dataset.
    This function initializes the dataset, the network, and runs the training loop.
    After training, it saves the trained model parameters to a pickle file. """
# Initialize dataset and model
mnist_data = MnistData()
(x_train, y_train), (x_test, y_test) = mnist_data.load()

network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

# Training
train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)

for i in range(iterations): # training loop
    batch_mask = np.random.choice(train_size, batch_size) # random batch selection
    x_batch = x_train[batch_mask] # batch input
    y_batch = y_train[batch_mask]   # batch output
    
    grads = network.gradient(x_batch, y_batch) # compute gradients
    
    # Update parameters
    for key in ('w1', 'b1', 'w2', 'b2'): # update weights and biases
        network.params[key] -= learning_rate * grads[key]
    
    if i % iter_per_epoch == 0: # print accuracy every epoch
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        print(f"Iteration {i} | Train Accuracy: {train_acc} | Test Accuracy: {test_acc}")

# Save trained model
with open("kolluru_mnist_model.pkl", 'wb') as f: # save model to pickle file
    pickle.dump(network.params, f)
