from flask import Flask, request, jsonify  # Import Flask framework and necessary modules
import os
from werkzeug.utils import secure_filename  # Used to safely store uploaded filenames
import joblib  # Used to load the saved model
from app.utils import predict_video_class  # Import custom function to predict video class

app = Flask(__name__) # Initialize Flask app
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")) # Get absolute path to saved model

model = joblib.load(MODEL_PATH) # Load the trained model

@app.route('/predict', methods=['POST']) # Define POST route for prediction
def predict():
    if 'video' not in request.files: # Check if video file is in request
        return jsonify({'error': 'No video file provided'}), 400 # Return error if not

    video = request.files['video'] # Get the uploaded video file
    filename = secure_filename(video.filename)  # Sanitize the filename
    save_path = os.path.join("temp_videos", filename) # Define path to save temp video
    os.makedirs("temp_videos", exist_ok=True)# Create temp_videos folder if not exists
    video.save(save_path)# Save the video

    label = predict_video_class(save_path, model)# Predict the class of the video

    return jsonify({'prediction': label})# Return the prediction result

if __name__ == "__main__":
    app.run(debug=True)# Run the Flask app
