
# 🎥 Video Violence Classification (Traditional ML Approach)

This project is a web-based demo for detecting **violent** and **non-violent** videos using traditional Machine Learning techniques (without Deep Learning). The system processes a short video, extracts features, and returns whether the content is violent or not.

---

## 📌 Project Highlights

- ✅ **Feature Extraction**: HOG (Histogram of Oriented Gradients) applied on video frames.
- ✅ **Machine Learning Models**: Evaluated multiple models (SVM, KNN, Random Forest), selected best using `GridSearchCV`.
- ✅ **Validation Accuracy**: ~74.5%
- ✅ **Test Accuracy**: ~74.1%
- ✅ **Deployment**: 
  - Flask API for backend inference.
  - Streamlit for frontend interface.

---

## 🚀 Try it yourself

### 🌐 Frontend Interface
Streamlit app allows users to upload videos and view predictions easily.

### 🔁 Backend API
A simple Flask API receives video uploads and returns predictions.

---

## 📊 Model Evaluation

### 🧪 Validation Results
![Validation Metrics](/images/val-metrics.png)

### 🧪 Test Results
![Test Metrics](/images/test-metrics.png)

---

## 🧠 Tech Stack

- Python
- scikit-learn
- OpenCV
- scikit-image
- Flask (API)
- Streamlit (UI)

---

## 📁 Project Structure

```
Project/
├── app/                     # Flask API
│   ├── main.py
│   └── utils.py
├── features/
│   ├── test_features.csv
│   ├── train_features.csv
│   └── val_features.csv
├── models/
│   └── best_model.pkl
├── scripts/                  # Model training
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── extract_features.py
│   └── extract_frames.py
├── web/                      # Streamlit UI
│   └── app.py
├── images/
│   ├── test-metrics.png
│   ├── val-metrics.png
│   ├── test-non.gif
│   ├── test-yes.gif
└── README.md
```

---

## 📦 Dataset

- 📂 Source: [Generalised Real-World Violence Detection Dataset](https://www.kaggle.com/datasets/showravdhar/generalised-real-world-violence-detection)
- 🎞️ Videos of real-world violent and non-violent scenes.
- 📌 Preprocessing: Extracted frames → HOG feature extraction → Model training.

---

## 📸 Demo

![Demo GIF here](images/test-non.gif) 

![Demo GIF here](images/test-yes.gif) <!-- You can replace this with a local path or URL later -->

---

## 🙌 Author

Built with ❤️ by [Pea(Nguyen Ngoc Phuc)]

Feel free to fork, use or extend for your own projects or portfolio!

---

## 📬 Contact

If you'd like to connect: [nnphuc2201@gmail.com] | [LinkedIn: https://www.linkedin.com/in/nguyen-ngoc-phuc-914286318//GitHub Profile]
