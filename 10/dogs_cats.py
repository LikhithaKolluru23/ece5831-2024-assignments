import tensorflow as tf
import pathlib
import os
import shutil
import matplotlib.pyplot as plt

class DogsCatsClassifier:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('/content/drive/MyDrive/dogs-vs-cats')
    SRC_DIR = pathlib.Path('/content/drive/MyDrive/dogs-vs-cats-original/train')
    EPOCHS = 20

    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None


    def make_dataset_folders(self, subset_name, start_index, end_index):
      """Creates directories for subsets and copies images if not already present."""
      # Check if all subfolders exist
      all_folders_exist = all(
          (self.BASE_DIR / subset_name / category).exists() 
          for category in self.CLASS_NAMES
      )

      if all_folders_exist:
          print(f"The folders for the {subset_name} dataset already exist. Skipping dataset creation.")
          return  # Exit early if all subfolders exist

      # Create folders and populate files if not all exist
      for category in self.CLASS_NAMES:
          subset_dir = self.BASE_DIR / subset_name / category
          os.makedirs(subset_dir, exist_ok=True)  # Create the directory if it doesn't exist
          files = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
          for file in files:
              src_file = self.SRC_DIR / file
              dst_file = subset_dir / file
              if not os.path.exists(dst_file):
                  shutil.copy(src_file, dst_file)

      print(f"Created folders and populated the {subset_name} dataset.")



    def _make_dataset(self, subset_name):
        """Creates a tf.data.Dataset object for a given subset."""
        return tf.keras.utils.image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )

    def make_dataset(self):
        """Creates train, validation, and test datasets."""
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        """Builds and compiles the neural network."""
        inputs = tf.keras.Input(shape=self.IMAGE_SHAPE)
        x = inputs
        if augmentation:
            x = tf.keras.Sequential([
                tf.keras.layers.RandomFlip('horizontal'),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2)
            ])(x)
        x = tf.keras.layers.Rescaling(1.0 / 255)(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, model_name):
        """Trains the model and plots training metrics."""
        model_name = f"model.Likhitha-Kolluru.dogs-cats.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_name),
        ]
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.EPOCHS,
            callbacks=callbacks
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.show()

    def load_model(self, model_name):
        """Loads a pre-trained model."""
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        """Predicts class for an image."""
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        predicted_class = self.CLASS_NAMES[int(prediction[0] > 0.5)]
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}")
        plt.show()
        print(f"Prediction: {predicted_class}")
