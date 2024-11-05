import sys
import argparse
from mnist import Mnist
import cv2
import matplotlib.pyplot as plt

def main():
    """ Main function to test handwritten digit recognizer with own handwritten images
     use: python module5-3.py <filename> <digit>
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Handwritten digit recognizer with own handwritten images")
    parser.add_argument("filename", type=str, help="Image filename")
    parser.add_argument("digit", type=int, help="Correct digit")
    args = parser.parse_args()

    # Initialize Mnist class and load model
    mnist = Mnist()
    mnist.init_network()

    # Predict the digit
    predicted_digit = mnist.predict_image(args.filename)

    if predicted_digit == args.digit:
        print(f"Success: Image {args.filename} is for digit {args.digit} and recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {args.filename} is for digit {args.digit} but the inference result is {predicted_digit}.")

    # Display the image
    img = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

   

if __name__ == "__main__":
    main()