from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
from app.utils import predict_video_class

app = Flask(__name__)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"))

model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    save_path = os.path.join("temp_videos", filename)
    os.makedirs("temp_videos", exist_ok=True)
    video.save(save_path)

    label = predict_video_class(save_path, model)

    return jsonify({'prediction': label})

if __name__ == "__main__":
    app.run(debug=True)
