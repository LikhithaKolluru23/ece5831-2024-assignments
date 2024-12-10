import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing

class BostonHousing:
    """
    A class to build, train, and evaluate a regression model for the Boston Housing dataset.
    """

    def __init__(self):
        """
        Initializes the BostonHousing class with placeholders for dataset, model, and training history.
        """
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.model = None
        self.history = None

    def prepare_data(self):
        """
        Loads the Boston Housing dataset and normalizes the data to have zero mean and unit variance.
        """
        # Load the dataset into training and testing sets
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = boston_housing.load_data()

        # Compute the mean and standard deviation of the training data
        mean = self.train_data.mean(axis=0)
        std = self.train_data.std(axis=0)

        # Normalize the training and testing data
        self.train_data -= mean
        self.train_data /= std
        self.test_data -= mean
        self.test_data /= std

    def build_model(self):
        """
        Builds and compiles a regression model using the Keras Sequential API.
        """
        # Define the model architecture
        self.model = models.Sequential([
            layers.Dense(64, activation="relu"),  # Hidden layer with 64 units and ReLU activation
            layers.Dense(64, activation="relu"),  # Another hidden layer
            layers.Dense(1)                       # Output layer with a single neuron for regression
        ])

        # Compile the model with loss function and optimizer
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def train(self, num_epochs=100, batch_size=16):
        """
        Trains the model on the training data with a validation split.
        """
        # Train the model and store the training history
        self.history = self.model.fit(
            self.train_data,                # Training input data
            self.train_targets,             # Training target values
            validation_split=0.2,           # Reserve 20% of data for validation
            epochs=num_epochs,              # Number of epochs
            batch_size=batch_size,          # Batch size
            verbose=0                       # Suppress verbose output
        )

    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.
        """
        # Plot the loss for both training and validation
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.show()

    def plot_mae(self):
        """
        Plots the training and validation Mean Absolute Error (MAE) over epochs.
        """
        # Plot the MAE for both training and validation
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.legend()
        plt.title("Training and Validation MAE")
        plt.show()

    def evaluate(self):
        """
        Evaluates the model on the test data and prints the results.
        """
        # Evaluate the model on test data
        test_mse_score, test_mae_score = self.model.evaluate(self.test_data, self.test_targets)
        print(f"Test MSE (Mean Square Error): {test_mse_score}")
        print(f"Test MAE (Mean Absolute Error): {test_mae_score}")
        return test_mse_score, test_mae_score

