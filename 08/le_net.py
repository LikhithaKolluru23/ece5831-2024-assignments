from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class LeNet:
    """
    Implementation of the LeNet architecture for digit classification on the MNIST dataset.
    """
    def __init__(self, batch_size=32, epochs=20):
        """
        Initialize the LeNet instance with default parameters.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()  # Build the LeNet model
        self._compile()       # Compile the model

    def _create_lenet(self):
        """Create the LeNet model architecture."""
        self.model = Sequential([
            # First convolutional layer with 6 filters, kernel size of 5x5, and sigmoid activation.
            Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid',
                   input_shape=(28, 28, 1), padding='same'),
            # First pooling layer with a 2x2 pool size and stride of 2.
            AveragePooling2D(pool_size=(2, 2), strides=2),

            # Second convolutional layer with 16 filters, kernel size of 5x5, and sigmoid activation.
            Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            # Second pooling layer with a 2x2 pool size and stride of 2.
            AveragePooling2D(pool_size=(2, 2), strides=2),

            # Flatten the output from the convolutional layers.
            Flatten(),

            # Fully connected layer with 120 units and sigmoid activation.
            Dense(120, activation='sigmoid'),
            # Fully connected layer with 84 units and sigmoid activation.
            Dense(84, activation='sigmoid'),
            # Output layer with 10 units (one for each digit) and softmax activation.
            Dense(10, activation='softmax')
        ])

    def _compile(self):
        """
        Compile the model with the Adam optimizer, categorical cross-entropy loss,
        and accuracy as a metric.
        """
        if self.model is None:
            raise ValueError('Error: Create a model first.')

        self.model.compile(optimizer='Adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _preprocess(self):
        """
        Load and preprocess the MNIST dataset. 
        - Normalize pixel values to the range [0, 1].
        - Reshape input images to include the channel dimension.
        - Convert labels to one-hot encoded format.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the images
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape to include the channel dimension
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # One-hot encode the labels
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        """
        Train the model on the preprocessed MNIST training data.
        """
        self._preprocess()
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.1)

    def evaluate(self):
        """
        Evaluate the model's performance on the test dataset.
        Prints Test loss and accuracy.
        """
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    def save(self, model_path_name):
        """
        Save the trained model to a file.

        """
        self.model.save(f"{model_path_name}.keras")
        print(f"Model saved as {model_path_name}.keras")

    def load(self, model_path_name):
        """
        Load a previously saved model from a file.

        """
        self.model = load_model(f"{model_path_name}.keras")
        print(f"Model loaded from {model_path_name}.keras")

    def predict(self, images):
        """
        Predict the class labels for the given images.

        Takes Input images to classify. Should have shape  (n_samples, 28, 28) or (28, 28) for a single image.

        Returns Predicted class labels for each input image.
        """
        if len(images.shape) == 3:  # Single image case
            images = np.expand_dims(images, axis=0)

        if images.shape[-1] != 1:  # Ensure channel dimension exists
            images = images.reshape(images.shape[0], 28, 28, 1)

        predictions = self.model.predict(images)
        return np.argmax(predictions, axis=1)
