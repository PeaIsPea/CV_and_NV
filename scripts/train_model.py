import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from evaluate_model import evaluate_model
import joblib


# Load features
def load_features(csv_path):
    data = pd.read_csv(csv_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Đọc dữ liệu
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "features"))
X_train, y_train = load_features(os.path.join(base_path, "train_features.csv"))
X_val, y_val     = load_features(os.path.join(base_path, "val_features.csv"))
X_test, y_test   = load_features(os.path.join(base_path, "test_features.csv"))

# Pipeline + GridSearch
pipe = Pipeline([
    ('scaler', StandardScaler()),  # scale cho dữ liệu
    ('clf', SVC())                 # tạm thời SVC, sẽ override bằng param_grid
])

param_grid = [
    {
        'clf': [SVC()],
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [0.1, 1, 10]
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    },
    {
        'clf': [KNeighborsClassifier()],
        'clf__n_neighbors': [3, 5, 7, 9]
    }
]

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

print("🔍 Đang chạy GridSearch cho tất cả mô hình...")
grid.fit(X_train, y_train)
print("✅ Best model + params:", grid.best_estimator_)
print("✅ Best CV accuracy:", grid.best_score_)

# Đánh giá trên validation
evaluate_model(grid, X_val, y_val, "Validation")

# Đánh giá trên test
evaluate_model(grid, X_test, y_test, "Test")



# Lưu model tốt nhất sau GridSearch
model_path = os.path.join("..", "models", "best_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(grid.best_estimator_, model_path)
print(f"\n💾 Saved best model to {model_path}")
