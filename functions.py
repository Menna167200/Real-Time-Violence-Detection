import cv2
import numpy as np
from collections import deque
from keras.models import load_model

# Constants
IMG_SIZE = 128
FRAME_COUNT = 15  # Increased frame count to capture better temporal information
CLASS_NAMES = ['No Violence', 'Violence']
CONFIDENCE_THRESHOLD = 0.5  # Threshold for binary classification
QUEUE_SIZE = 10  # Queue size for smoothing or majority voting

# Load your trained model
model = load_model('bestt.keras')

# Preprocess a frame (resize, normalize, and convert to RGB)
def preprocess_frame(frame, size=(IMG_SIZE, IMG_SIZE)):
    frame = cv2.resize(frame, size)
    frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Smooth predictions using a moving average over the queue
def smooth_predictions(predictions, queue):
    queue.append(predictions)
    average_prediction = np.mean(queue, axis=0)  # Compute the mean of the queue
    return average_prediction

# Apply majority voting based on the queue
def majority_voting(queue, threshold=0.5):
    count_violence = np.sum([1 if pred > threshold else 0 for pred in queue])  # Count violence predictions
    total_count = len(queue)
    return (count_violence / total_count) > threshold  # Majority voting

# Real-time video stream classification with smoothing and majority voting
def classify_video_stream(skip_frames=1, desired_fps=30):
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    frames = []  # To store frames
    last_label = ""  # Variable to hold the last displayed label
    prediction_queue = deque(maxlen=QUEUE_SIZE)  # Queue for smoothing or majority voting

    while True:
        ret, frame = cap.read()  # Capture frame
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the current frame
        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)

        # If enough frames are collected, make a prediction
        if len(frames) == FRAME_COUNT:
            frames_array = np.expand_dims(np.array(frames), axis=0)  # Add batch dimension
            prediction = model.predict(frames_array)[0][0]  # Binary classification returns 1 probability

            # Add prediction to the queue
            smoothed_prediction = smooth_predictions(prediction, prediction_queue)

            # Apply confidence threshold and get predicted label based on threshold > 0.5
            predicted_label = CLASS_NAMES[1] if smoothed_prediction > CONFIDENCE_THRESHOLD else CLASS_NAMES[0]

            # Apply majority voting (optional)
            if len(prediction_queue) == prediction_queue.maxlen:  # Wait until queue is full
                is_violence = majority_voting(prediction_queue, threshold=CONFIDENCE_THRESHOLD)
                predicted_label = CLASS_NAMES[1] if is_violence else CLASS_NAMES[0]

            # Update last displayed label with smoothed prediction
            last_label = f'Class: {predicted_label}, Confidence: {smoothed_prediction:.2f}' if smoothed_prediction > CONFIDENCE_THRESHOLD else "No Violence"

            # Clear the frames list to start over for the next prediction
            frames = []

        # Display the last prediction on the video stream
        cv2.putText(frame, last_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the prediction
        cv2.imshow('Real-Time Video Classification', frame)

        # Skip the specified number of frames
        for _ in range(skip_frames):
            cap.grab()  # Skip frames without reading

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

