import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Function to load class names from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Load the trained model from Teachable Machine
model = tf.keras.models.load_model("model/keras_Model.h5")

# Load the class names from the labels.txt file
class_names = load_labels('model/labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Video writer initialization
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  # Modify resolution as needed

# Duration in seconds
duration = 20
end_time = time.time() + duration  # Calculate end time

last_prediction_label = None  # To track the last predicted class
recording = False  # To indicate if we are currently recording

while time.time() < end_time:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction (resize and normalize)
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    prediction_label = class_names[class_idx]
    
    # Display the resulting frame with prediction
    cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Check if the prediction has changed
    if prediction_label != last_prediction_label:
        last_prediction_label = prediction_label
        recording = True  # Start recording when there is a change
    else:
        recording = False  # Stop recording when the prediction is the same

    # If recording, write the frame to the video file
    if recording:
        out.write(frame)

    # Use matplotlib to show the frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {prediction_label}')
    plt.axis('off')  # Hide axis
    plt.show(block=False)  # Show without blocking
    plt.pause(1)  # Pause for 1 second before showing the next frame

# When everything is done, release the capture and video writer
cap.release()
out.release()
#cv2.destroyAllWindows()
