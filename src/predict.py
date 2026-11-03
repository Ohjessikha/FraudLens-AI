
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "fraud_model.joblib"


def load_model():
    """Load the saved model and its expected feature names."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src/train_model.py first."
        )

    saved_data = joblib.load(MODEL_PATH)

    return (
        saved_data["model"],
        saved_data["feature_names"],
        saved_data["threshold"],
    )


def predict_transaction(transaction):
    """
    Predict whether a transaction should be flagged.

    transaction must contain all 30 model input features.
    """
    model, feature_names, threshold = load_model()

    # Convert the input into a one-row DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Ensure the required features exist
    missing = [
        feature for feature in feature_names
        if feature not in transaction_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    # Use the same feature order as during training
    transaction_df = transaction_df[feature_names]

    # Estimate the probability of the fraud-labelled class
    fraud_probability = float(
        model.predict_proba(transaction_df)[0, 1]
    )

    predicted_class = int(
        fraud_probability >= threshold
    )

    # Simple demonstration categories, not real-world guarantees
    if fraud_probability < 0.30:
        risk_category = "Low"
    elif fraud_probability < 0.70:
        risk_category = "Medium"
    else:
        risk_category = "High"

    return {
        "fraud_probability": fraud_probability,
        "predicted_class": predicted_class,
        "prediction": (
            "Potential fraud"
            if predicted_class == 1
            else "Not flagged"
        ),
        "risk_category": risk_category,
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Use a real row from the public dataset for a smoke test.
    # The label is deliberately excluded from the model inputs.
    dataset_path = PROJECT_DIR / "data" / "creditcard.csv"
    df = pd.read_csv(dataset_path, nrows=5)

    example_transaction = (
        df.drop(columns=["Class"]).iloc[0].to_dict()
    )

    result = predict_transaction(example_transaction)

    print("=== Transaction Prediction Test ===")
    print("Fraud probability:", round(result["fraud_probability"], 4))
    print("Prediction:", result["prediction"])
    print("Risk category:", result["risk_category"])
    print("Threshold:", result["threshold"])
