import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

class Imdb:
    """
    A class to build, train, and evaluate a sentiment analysis model on the IMDb dataset.
    """

    def __init__(self, num_words=10000):
        """
        Initializes the Imdb class with placeholders for dataset, model, and training history.

        Args:
            num_words: The maximum number of words to consider in the dataset.
        """
        self.num_words = num_words
        self.model = None
        self.history = None 

    def prepare_data(self):
        """
        Loads and preprocesses the IMDb dataset by padding sequences to a fixed length.
        """
        # Load the dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=self.num_words)

        # Pad sequences to ensure consistent input length
        self.x_train = tf.keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=256) 
        self.x_test = tf.keras.preprocessing.sequence.pad_sequences(self.x_test, maxlen=256)

    def build_model(self):
        """
        Builds and compiles a binary classification model using the Keras Sequential API.
        """
        # Define the model architecture
        self.model = models.Sequential([
            layers.Embedding(self.num_words, 16),      # Embedding layer
            layers.GlobalAveragePooling1D(),          # Global pooling layer
            layers.Dense(16, activation='relu'),      # Hidden layer 
            layers.Dense(1, activation='sigmoid')     # Output layer for binary classification
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=512):
        """
        Trains the model with a validation split.

        Args:
            epochs: Number of epochs to train the model.
            batch_size: Batch size for training.
        """
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.
        """
        # Plot the loss
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss Over Epochs')
        plt.show()

    def plot_accuracy(self):
        """
        Plots the training and validation accuracy over epochs.
        """
        # Plot the accuracy
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy Over Epochs')
        plt.show()

    def evaluate(self):
        """
        Evaluates the model on the test data.
        """
        results = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        return results

