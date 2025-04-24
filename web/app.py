import streamlit as st
import requests

st.title("🎥 Video Violence Classification")

video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])

if video_file:
    with open("temp_input.mp4", "wb") as f:
        f.write(video_file.read())

    st.video("temp_input.mp4")

    if st.button("🧠 Predict"):
        files = {'video': open("temp_input.mp4", 'rb')}
        response = requests.post("http://localhost:5000/predict", files=files)
        if response.ok:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
        else:
            st.error("Prediction failed.")
