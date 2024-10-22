import argparse
import numpy as np
import matplotlib.pyplot as plt
from mnist_data import MnistData

def show_image_and_label(dataset_type, index):
    """
    Displays an image and its corresponding label from the MNIST dataset.

    Parameters:
    ----------
    dataset_type : str
        The type of dataset to load. Must be either 'train' or 'test'.
    index : int
        The index of the image and label to display.
     usage : python module5-2.py train 5

    """
    mnist = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist.load()

    if dataset_type == 'train':
        images, labels = train_images, train_labels
    elif dataset_type == 'test':
        images, labels = test_images, test_labels
    else:
        raise ValueError("First argument must be 'train' or 'test'.")

    if index < 0 or index >= len(images):
        raise IndexError(f"Index {index} is out of range for {dataset_type} dataset.")

  # Print the label
    label = np.argmax(labels[index])  # Get the index of the one-hot encoded label
    print(f"Label: {label}")

    # Show the image
    image = images[index].reshape(28, 28)  # Reshape flattened image back to 28x28
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()

  
if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Showing an image from the MNIST dataset with its label.')
    parser.add_argument('dataset_type', choices=['train', 'test'], help='Specify "train" or "test" dataset.')
    parser.add_argument('index', type=int, help='Specify the index number of the image to display.')

    args = parser.parse_args()

    # Show the image and label based on input arguments
    show_image_and_label(args.dataset_type, args.index)

    ## usage : python module5-2.py train 5
