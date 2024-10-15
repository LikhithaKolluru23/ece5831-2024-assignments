"""
rock-paper-scissors.py

This script uses a pre-trained Keras model to predict the class of an input image like rock, paper, scissors from a given image file. It prepares the image 
by resizing, normalizing, and feeding it to the model, then outputs the predicted 
class and confidence score. It also displays the image with the prediction and 
confidence using matplotlib.

Functions:
-----------
prepare_image(image_path):
    Loads, resizes, and normalizes an image to the required shape for model prediction.

main(image_path):
    Prepares the image and uses the pre-trained Keras model to predict the class. 
    It outputs the predicted class and confidence score and displays the image 
    with the prediction.

Usage:
------
Run the script from the command line by providing the image path:
    python rock-paper-scissors.py <image_path>

Parameters:
-----------
image_path (str): The file path to the image you want to classify.

"""

import sys
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/keras_Model.h5", compile=False)

# Load the labels
with open("model/labels.txt", "r") as f:
    class_names = f.read().splitlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def prepare_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Resize the image to be 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    return normalized_image_array

def main(image_path):
    # Prepare the image
    data[0] = prepare_image(image_path)

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score in the desired format
    print(f"Class: {class_name[2:]}")
    print(f"Confidence Score: {confidence_score:.4f}")
    # Show the image using matplotlib
    plt.imshow(Image.open(image_path))
    plt.axis('off')  # Hide axes
    plt.title(f"Predicted: {class_name[2:]}\nConfidence: {confidence_score:.4f}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rock-paper-scissors.py <image_path>")
    else:
        main(sys.argv[1])
