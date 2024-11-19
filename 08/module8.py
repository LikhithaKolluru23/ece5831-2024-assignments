import sys
import pickle
import argparse
import cv2
import numpy as np
from le_net import LeNet  # Import the LeNet class for digit recognition
import matplotlib.pyplot as plt

def main():
    """
    Main function to perform handwritten digit recognition on an input image.
    - Loads a pre-trained LeNet model.
    - Preprocesses an input image provided as a command-line argument.
    - Predicts the digit in the image using the model.
    - Compares the predicted digit with the correct digit provided as input.
    - Displays the image and prints the prediction result.
    """
    # Initialize argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Handwritten digit recognizer with user-provided handwritten images")
    parser.add_argument("filename", type=str, help="Path to the image file (grayscale)")
    parser.add_argument("digit", type=int, help="Correct digit (0-9) for comparison")
    args = parser.parse_args()

    # Load the pre-trained LeNet model
    lenet = LeNet(batch_size=64, epochs=10)  # Initialize LeNet with appropriate batch size and epochs
    lenet.load("kolluru")  # Load the trained model from file

    # Loading and preprocessing the input image
    # Reading the image in grayscale mode (single channel)
    image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image file '{args.filename}'.")
        sys.exit(1) 

    # Resizing the image to 28x28 (LeNet input size) and normalize pixel values
    img = cv2.resize(image, (28, 28))  # Resize the image to match the model's input dimensions
    img = img.astype(np.float32) / 255.0  # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=(0, -1))  # Adding batch and channel dimensions to avoid error

    #prediction using the pre-trained model
    predictions = lenet.model.predict(img)  # Predicting the digit from the image
    predicted_digit = np.argmax(predictions)  # Get the class with the highest probability

    # Displays the input image
    plt.imshow(image, cmap='gray')  # Shows sthe original grayscale image
    plt.axis('off')  # Hides the axes for better visualization
    plt.title(f"Predicted: {predicted_digit}")  # Displays the predicted digit as title
    plt.show()

    # Comparing the prediction with the correct digit
    if predicted_digit == args.digit:
        print(f"Success: Image '{args.filename}' is for digit {args.digit} and recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image '{args.filename}' is for digit {args.digit} but the inference result is {predicted_digit}.")


if __name__ == "__main__":
    main()  # Calls the main function
