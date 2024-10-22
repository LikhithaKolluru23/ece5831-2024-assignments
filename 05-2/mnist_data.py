import os
import pickle
import urllib.request
import gzip
import numpy as np

class MnistData:
    """
    A class to handle the MNIST dataset, including downloading, loading, and processing the data.
    
    Attributes:
    -----------
    image_dim : tuple
        Dimensions of each image in the MNIST dataset (28x28 pixels).
    image_size : int
        Total number of pixels per image (28 * 28).
    dataset_dir : str
        Directory where the dataset files will be stored.
    dataset_pkl : str
        Filename for the serialized dataset in pickle format.
    url_base : str
        Base URL for downloading the MNIST dataset.
    key_file : dict
        Dictionary mapping dataset types (train/test images and labels) to their respective filenames.
    dataset : dict
        Stores the MNIST data, including images and labels for training and testing.
    dataset_pkl_path : str
        The full path to the pickle file where the dataset is stored.
    """
    image_dim = (28, 28)
    image_size = image_dim[0] * image_dim[1]
    dataset_dir = 'dataset'
    dataset_pkl = 'mnist.pkl'
    url_base = 'http://jrkwon.com/data/ece5831/mnist/'

    key_file = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        """
        Initializes the MnistData object. Sets up the dataset directory, initializes the dataset,
        and ensures that the necessary files are downloaded and loaded into memory.
        """
        self.dataset = {}
        self.dataset_pkl_path = os.path.join(self.dataset_dir, self.dataset_pkl)
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        self._init_dataset()

    def _download(self, file_name):
        """
        Downloads a MNIST dataset file if it does not already exist in the dataset directory.
        
        Parameters:
        -----------
        file_name : str
            The name of the file to download.
        """
        file_path = os.path.join(self.dataset_dir, file_name)
        if os.path.exists(file_path):
            print(f"File: {file_name} already exists.")
            return
        print(f"Downloading {file_name}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('Accept', '')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print("Done")

    def _download_all(self):
        """
        Downloads all required MNIST dataset files (train/test images and labels).
        """
        for file_name in self.key_file.values():
            self._download(file_name)

    def _load_images(self, file_name):
        """
        Loads and processes MNIST images from a gzipped file.
        
        Parameters:
        -----------
        file_name : str
            The name of the file containing the image data.
            
        Returns:
        --------
        np.ndarray
            A NumPy array containing the reshaped image data.
        """
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        return images.reshape(-1, self.image_size)

    def _load_labels(self, file_name):
        """
        Loads MNIST labels 
        Parameters:
        -----------
        file_name : str
            The name of the file containing the label data.
            
        Returns:
        --------
        np.ndarray
            A NumPy array containing the label data.
        """
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _create_dataset(self):
        """
        Loads the MNIST dataset (both images and labels for training and testing) and saves it as a pickle file.
        """
        self.dataset['train_images'] = self._load_images(os.path.join(self.dataset_dir, self.key_file['train_images']))
        self.dataset['train_labels'] = self._load_labels(os.path.join(self.dataset_dir, self.key_file['train_labels']))
        self.dataset['test_images'] = self._load_images(os.path.join(self.dataset_dir, self.key_file['test_images']))
        self.dataset['test_labels'] = self._load_labels(os.path.join(self.dataset_dir, self.key_file['test_labels']))
        with open(self.dataset_pkl_path, 'wb') as f:
            print(f'Creating Pickle: {self.dataset_pkl_path}')
            pickle.dump(self.dataset, f)
            print('Done')

    def _init_dataset(self):
        """
        Initializes the dataset by downloading the required files, creating the dataset if needed, and loading
        it from a pickle file if it exists.
        """
        self._download_all()
        if os.path.exists(self.dataset_pkl_path):
            with open(self.dataset_pkl_path, 'rb') as f:
                print(f'Pickle: {self.dataset_pkl_path} already exists.')
                print('Loading...')
                self.dataset = pickle.load(f)
                print('Done.')
        else:
            self._create_dataset()

    def load(self):
        """
        Normalizes the image data (converting pixel values to the range [0, 1]) and converts the labels to one-hot
        encoding.
        
        Returns:
        --------
        tuple:
            Two tuples containing training data (images, labels) and test data (images, labels).
        """
        for key in ('train_images', 'test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32) / 255.0
        for key in ('train_labels', 'test_labels'):
            self.dataset[key] = self._change_one_hot_label(self.dataset[key], 10)
        return (self.dataset['train_images'], self.dataset['train_labels']), \
               (self.dataset['test_images'], self.dataset['test_labels'])

    def _change_one_hot_label(self, y, num_class):
        """
        Converts a label vector into a one-hot encoded matrix.
        
        Parameters:
        -----------
        y : np.ndarray
            A 1D array containing class labels.
        num_class : int
            The number of classes (e.g., 10 for MNIST).
            
        Returns:
        --------
        np.ndarray
            A 2D array where each row is the one-hot encoded representation of the corresponding label.
        """
        t = np.zeros((y.size, num_class))
        for idx, row in enumerate(t):
            row[y[idx]] = 1
        return t

# Command line interface
if __name__ == '__main__':
    print("MnistData class is to load MNIST datasets.")
    print("load()")
    print("    Return (train_images, train_labels), (test_images, test_labels)")
    print("    Each image is flattened to 784 bytes. To display an image, reshaping is necessary.")
    print("    Each label is one-hot-encoded. To get a number, use argmax to get the index where 1 is located.")
