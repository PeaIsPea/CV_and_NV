import streamlit as st  # Import Streamlit
import requests  # HTTP requests to Flask API

st.title("🎥 Video Violence Classification") # Title of app

video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"]) # Upload video

if video_file:
    with open("temp_input.mp4", "wb") as f:
        f.write(video_file.read()) # Save video to file

    st.video("temp_input.mp4") # Display video in app

    if st.button("🧠 Predict"):
        files = {'video': open("temp_input.mp4", 'rb')} # Prepare file
        response = requests.post("http://localhost:5000/predict", files=files) # Send to Flask
        if response.ok:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}") # Show result
        else:
            st.error("Prediction failed.") # Show error
