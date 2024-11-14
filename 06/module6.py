import sys
import pickle
import argparse
import cv2
import numpy as np
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import matplotlib.pyplot as plt

def main():
    """
    Main function to perform handwritten digit recognition on an input image.
    It loads a pre-trained model, processes the input image, makes a prediction,
    and compares the predicted digit with the correct digit provided.
    """
    # Load trained model
    network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)
    with open("kolluru_mnist_model.pkl", 'rb') as f:
        network.params = pickle.load(f)

    
    network.update_layers()
    
    # Load image and preprocess
   
    parser = argparse.ArgumentParser(description="Handwritten digit recognizer with own handwritten images")
    parser.add_argument("filename", type=str, help="Image filename")
    parser.add_argument("digit", type=int, help="Correct digit")
    args = parser.parse_args()
  
    image = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(image, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = img.flatten().reshape(1, 784)  # Flatten the image to match MNIST data shape
      

    y = network.predict(img) # Perform prediction
    predicted_digit = np.argmax(y)
    
    plt.imshow( image, cmap='gray') # Display the image
    plt.axis('off')
    plt.show()

   
    
    if predicted_digit == args.digit:
        print(f"Success: Image {args.filename} is for digit {args.digit} and recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {args.filename} is for digit {args.digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()
