
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Check required columns
if "target" not in df.columns:
    raise ValueError("The dataset must contain a 'target' column.")

df = df.dropna(subset=["target"]).copy()

X = df.drop(columns=["target"])
y = df["target"]

# This dataset should have binary labels: 0 and 1
if set(y.unique()) != {0, 1}:
    raise ValueError(
        f"Expected target labels 0 and 1, got {sorted(y.unique())}"
    )

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)

print("Dataset shape:", df.shape)
print("Features:", X.columns.tolist())
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Retrain final model on all available rows
model.fit(X, y)

# Save model and feature information
bundle = {
    "model": model,
    "features": X.columns.tolist(),
    "target_column": "target",
    "classes": model.classes_.tolist(),
}

joblib.dump(bundle, MODEL_DIR / "heart_model.joblib")

print("\nModel saved successfully!")
print("File:", MODEL_DIR / "heart_model.joblib")
