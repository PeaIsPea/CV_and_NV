from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Print evaluation metrics for a model
def evaluate_model(model, x, y, dataset_name="Validation"):
    print(f"\n📊 {dataset_name} Results:")
    y_pred = model.predict(x) # Get predictions
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
