import os
import numpy as np
from skimage.feature import hog  # Extract HOG features from images
from skimage.color import rgb2gray  # Convert RGB to grayscale
from skimage.io import imread  # Read image file
from skimage.transform import resize  # Resize image
import cv2

def extract_frames(video_path, max_frames=30): # Extract max 30 frames from video
    frames = []
    cap = cv2.VideoCapture(video_path) # Open video file
    count = 0
    while count < max_frames and cap.isOpened():
        ret, frame = cap.read() # Read a frame
        if not ret:
            break
        frames.append(frame) # Append frame to list
        count += 1
    cap.release() # Release video capture
    return frames

def extract_hog_from_frames(frames):
    features = []
    for frame in frames:
        gray = rgb2gray(frame) # Convert frame to grayscale
        resized = resize(gray, (128, 64)) # Resize frame
        hog_feat = hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True) # Extract HOG
        features.append(hog_feat) # Append features
    return np.mean(features, axis=0) # Return mean HOG vector

def predict_video_class(video_path, model):
    frames = extract_frames(video_path) # Get frames
    if not frames:
        return "Error: No frames extracted" # Return error
    feature = extract_hog_from_frames(frames) # Extract features
    prediction = model.predict([feature])[0] # Predict with model
    return "Violent" if prediction == 1 else "Non-Violent" # Map label
