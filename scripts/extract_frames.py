import os
import cv2


def extract_frames_from_video(video_path, output_folder, frame_interval=30):
    """
    Trích xuất frame từ video và lưu vào thư mục output.
    frame_interval = số frame bỏ qua giữa 2 lần lưu ảnh (30 ~ mỗi 1 giây nếu video 30fps)
    """
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frame = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{saved_frame}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame += 1

        count += 1

    cap.release()

def process_dataset_split(split_dir, output_root, frame_interval=30):
    for label in ["violent", "non_violent"]:
        input_folder = os.path.join(split_dir, label)
        output_folder = os.path.join(output_root, os.path.basename(split_dir), label)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for video_file in os.listdir(input_folder):
            if video_file.endswith(('.mp4', '.avi', '.mov')):  # các định dạng video phổ biến
                video_path = os.path.join(input_folder, video_file)
                extract_frames_from_video(video_path, output_folder, frame_interval)

# Ví dụ chạy:
process_dataset_split("../data/train", "../frames", frame_interval=30)
process_dataset_split("../data/val", "../frames", frame_interval=30)
process_dataset_split("../data/test", "../frames", frame_interval=30)
