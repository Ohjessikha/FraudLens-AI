
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Find the project folder and required files
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nFraud percentage:", round(df["Class"].mean() * 100, 4), "%")

# Separate input features (X) and target label (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# Split before training; preserve the fraud/non-fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Scale features and train a class-balanced classifier
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Predict probabilities and classes on unseen test data
fraud_probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

# Evaluate the model
print("\n=== Classification Report ===")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not flagged", "Potential fraud"],
        zero_division=0,
    )
)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, predictions))

print("\n=== Average Precision ===")
print(round(average_precision_score(y_test, fraud_probabilities), 6))

# Save the complete preprocessing and prediction pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "threshold": 0.5,
    },
    MODEL_PATH,
)

print("\nModel saved successfully:")
print(MODEL_PATH)
print("\nMODULE 2 TRAINING COMPLETE")
