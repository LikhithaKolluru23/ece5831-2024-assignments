import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np

class Reuters:
    """
    A class to build, train, and evaluate a text classification model on the Reuters dataset.
    """

    def __init__(self, num_words=10000):
        """
        Initializes the Reuters class with placeholders for dataset, model, and training history.

        Args:
            num_words : The maximum number of words to consider in the dataset.
        """
        self.num_words = num_words
        self.model = None
        self.history = None

    def prepare_data(self):
        """
        Loads and preprocesses the Reuters dataset by vectorizing sequences and one-hot encoding labels.
        """
        # Load the dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = reuters.load_data(num_words=self.num_words)

        def vectorize_sequences(sequences, dimension=10000):
            """
            Converts a list of sequences into a 2D NumPy array.

            Args:
                sequences : List of sequences to vectorize.
                dimension : Dimension of the one-hot vector.
            """
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                for j in sequence:
                    results[i, j] = 1.0
            return results

        # Vectorize input sequences
        self.x_train = vectorize_sequences(self.x_train)
        self.x_test = vectorize_sequences(self.x_test)

        # One-hot encode the labels
        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes=46)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes=46)

    def build_model(self):
        """
        Builds and compiles a multi-class classification model using the Keras Sequential API.
        """
        self.model = models.Sequential([
            layers.Dense(64, activation="relu"),  # First hidden layer
            layers.Dense(64, activation="relu"),  # Second hidden layer
            layers.Dense(46, activation="softmax")  # Output layer for multi-class classification
        ])
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, initial_epochs=20, retrain_batch_size=512): # tried retraining for better accuarcy as in the textbook
        """
        Trains the model with validation, and retrains it with the best epoch.

        Args:
            initial_epochs : Number of epochs for initial training.
            retrain_batch_size : Batch size for retraining.
        """
        # Create validation and partial training datasets
        x_val = self.x_train[:1000]
        partial_x_train = self.x_train[1000:]
        y_val = self.y_train[:1000]
        partial_y_train = self.y_train[1000:]

        # Train the model with validation data
        self.history = self.model.fit(
            partial_x_train, partial_y_train,
            epochs=initial_epochs,
            batch_size=retrain_batch_size,
            validation_data=(x_val, y_val)
        )

        # Find the best epoch based on validation accuracy
        best_epoch = np.argmax(self.history.history["val_accuracy"]) + 1
        print(f"Best epoch: {best_epoch}")

        # Retrain the model on the full training set
        self.build_model()  # Rebuild the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=best_epoch,
            batch_size=retrain_batch_size,
            validation_data=(x_val, y_val)
        )

    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.
        """
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss Over Epochs')
        plt.show()

    def plot_accuracy(self):
        """
        Plots the training and validation accuracy over epochs.
        """
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy Over Epochs')
        plt.show()

    def evaluate(self):
        """
        Evaluates the model on the test set and prints results.
        """
        results = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        return results

