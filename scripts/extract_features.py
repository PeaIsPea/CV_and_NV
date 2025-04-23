import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import csv

def extract_hog_features(image_path):
    try:
        image = imread(image_path)
        gray = rgb2gray(image)
        resized = resize(gray, (128, 64), anti_aliasing=True)  # Resize đồng nhất
        feature = hog(resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        return feature
    except Exception as e:
        print(f"[SKIP] Lỗi ảnh {image_path}: {e}")
        return None

def process_split_frames(split_dir):
    x = []
    y = []
    label_encoder = LabelEncoder()
    labels = ["non_violent", "violent"]

    for label in labels:
        label_path = os.path.join(split_dir, label)
        for root, _, files in os.walk(label_path):
            # Gom các ảnh theo tên video gốc
            video_frames = {}
            for f in files:
                if f.endswith(".jpg"):
                    video_name = "_".join(f.split("_")[:-1])
                    video_frames.setdefault(video_name, []).append(os.path.join(root, f))

            for video_name, frame_paths in video_frames.items():
                features = []
                for path in frame_paths:
                    hog_feature = extract_hog_features(path)
                    if hog_feature is not None:
                        features.append(hog_feature)

                if features:
                    try:
                        avg_feature = np.mean(features, axis=0)
                        x.append(avg_feature)
                        y.append(label)
                    except Exception as e:
                        print(f"[SKIP] Video {video_name} gặp lỗi khi tính trung bình: {e}")

    return np.array(x), label_encoder.fit_transform(y)

def save_features_to_csv(x, y, out_path):
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for feature, label in zip(x, y):
            writer.writerow(np.append(feature, label))

# Đường dẫn thư mục frames
base_frames = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frames"))

# xử lý từng tập train/val/test
for split in ["train", "val", "test"]:
    print(f"🔍 Processing {split} set...")
    split_path = os.path.join(base_frames, split)
    x, y = process_split_frames(split_path)

    output_file = os.path.join("..", "features", f"{split}_features.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_features_to_csv(x, y, output_file)
    print(f"✅ Saved {split}_features.csv with {len(x)} samples.")
