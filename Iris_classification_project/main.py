from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    print("🔍 Loading data...")
    df = load_data("data/IRIS.csv")
    print("🧾 Columns in dataset:", df.columns.tolist())


    print("🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    print("🧠 Training model...")
    model = train_model(X_train, y_train)

    print("📈 Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
