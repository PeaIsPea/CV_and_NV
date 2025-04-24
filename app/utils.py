import os
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
import cv2
import tempfile
import shutil

def extract_frames(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def extract_hog_from_frames(frames):
    features = []
    for frame in frames:
        gray = rgb2gray(frame)
        resized = resize(gray, (128, 64))
        hog_feat = hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.mean(features, axis=0)

def predict_video_class(video_path, model):
    frames = extract_frames(video_path)
    if not frames:
        return "Error: No frames extracted"
    feature = extract_hog_from_frames(frames)
    prediction = model.predict([feature])[0]
    return "Violent" if prediction == 1 else "Non-Violent"
