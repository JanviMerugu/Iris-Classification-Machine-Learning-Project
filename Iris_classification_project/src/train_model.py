# src/train_model.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Optional: print metrics to terminal
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model, y_pred
