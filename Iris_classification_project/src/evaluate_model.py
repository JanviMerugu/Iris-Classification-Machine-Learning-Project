from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the model and prints accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.2f}\n")

    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📄 Classification Report:")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
