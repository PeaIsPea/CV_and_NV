
# 🎥 Video Violence Classification (Traditional ML Approach)

This project is a web-based demo for detecting **violent** and **non-violent** videos using traditional Machine Learning techniques (without Deep Learning). The system processes a short video, extracts features, and returns whether the content is violent or not.

![Demo GIF here](<PLACEHOLDER_FOR_GIF>) <!-- You can replace this with a local path or URL later -->

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
![Validation Metrics](./images/68c9a4c5-0458-44e4-b4e6-45f4bedb79d8.png)

### 🧪 Test Results
![Test Metrics](./images/54d9baad-e129-4461-84d8-765174ad6b6e.png)

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
project/
├── app/               # Flask API
│   ├── main.py
│   └── utils.py
├── model/
│   └── best_model.joblib
├── scripts/           # Model training
│   └── train_model.py
├── web/               # Streamlit UI
│   └── app.py
└── README.md
```

---

## 📦 Dataset

- 📂 Source: [Generalised Real-World Violence Detection Dataset](https://www.kaggle.com/datasets/showravdhar/generalised-real-world-violence-detection)
- 🎞️ Videos of real-world violent and non-violent scenes.
- 📌 Preprocessing: Extracted frames → HOG feature extraction → Model training.

---

## 📸 Demo

> _(Insert your GIF of the demo here once ready)_

---

## 🙌 Author

Built with ❤️ by [Your Name]

Feel free to fork, use or extend for your own projects or portfolio!

---

## 📬 Contact

If you'd like to connect: [your.email@example.com] | [LinkedIn/GitHub Profile]
