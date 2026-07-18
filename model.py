"""
model.py
--------
Train a Random Forest classifier for credit card fraud detection
and predict whether a transaction is Fraud or Genuine.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from utils import DATA_PATH, load_dataset

MODEL_PATH = "fraud_model.pkl"

# Exact feature order used during training (all columns except Class)
FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


def load_data(path=DATA_PATH):
    """Load the credit card dataset from a CSV file."""
    df = load_dataset(path=path)
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df


def split_features_and_target(df):
    """
    Separate features (X) from the target (y).
    X = all columns except Class
    y = Class (0 = Genuine, 1 = Fraud)
    """
    X = df.drop(columns=["Class"])
    # Keep columns in a fixed order so training and prediction match
    X = X[FEATURE_COLUMNS]
    y = df["Class"]
    return X, y


def split_train_test(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    class_weight='balanced' helps the model pay attention to rare fraud cases.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print("Model training complete.")
    print("Model type:", type(model))
    print("Classes (0=Genuine, 1=Fraud):", model.classes_)
    print("Feature names used in training:", list(model.feature_names_in_))
    return model


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for the trained model."""
    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, path=MODEL_PATH):
    """Save the trained model to a file using joblib."""
    joblib.dump(model, path)
    print(f"\nModel saved as {path}")


def load_saved_model(path=MODEL_PATH):
    """Load a previously saved Random Forest model from disk."""
    model = joblib.load(path)
    print(f"Loaded model from {path}: {type(model)}")
    return model


def prepare_features(transaction):
    """
    Convert transaction input into a one-row DataFrame with the SAME
    column names and order used during training.
    """
    if isinstance(transaction, dict):
        row = pd.DataFrame([transaction])
    elif isinstance(transaction, pd.Series):
        row = transaction.to_frame().T
    elif isinstance(transaction, pd.DataFrame):
        row = transaction.iloc[[0]].copy()
    else:
        # list/array must already be in FEATURE_COLUMNS order
        row = pd.DataFrame([list(transaction)], columns=FEATURE_COLUMNS)

    missing = [col for col in FEATURE_COLUMNS if col not in row.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Enforce identical feature order as training
    row = row[FEATURE_COLUMNS]
    return row


def predict_transaction(transaction):
    """
    Predict whether a transaction is Fraud or Genuine.

    Parameters
    ----------
    transaction : list, dict, or pandas DataFrame/Series
        Feature values for one transaction (all columns except Class).

    Returns
    -------
    dict with:
        prediction        – "Fraud" or "Genuine"
        fraud_probability – probability of fraud (class 1)
        risk_score        – fraud probability scaled to 0–100
    """
    model = load_saved_model()

    # Safety: must be a trained RandomForest, not a hardcoded value
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            f"Expected RandomForestClassifier, got {type(model)}. "
            "Retrain with: python model.py"
        )

    row = prepare_features(transaction)

    # Debug information (visible in the terminal running Streamlit / model.py)
    print("\n--- Prediction Debug ---")
    print("Feature names:", list(row.columns))
    print("Feature values:", row.iloc[0].to_dict())

    pred_class = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    # classes_ is [0, 1] → index 0 = Genuine, index 1 = Fraud
    genuine_probability = float(probabilities[0])
    fraud_probability = float(probabilities[1])

    print("Raw predict():", pred_class)
    print("predict_proba():", probabilities)
    print("Genuine probability [0]:", genuine_probability)
    print("Fraud probability [1]:", fraud_probability)

    prediction = "Fraud" if pred_class == 1 else "Genuine"
    risk_score = round(fraud_probability * 100, 2)

    return {
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
    }


def main():
    """Load data, train the model, evaluate it, and save it."""
    df = load_data()
    X, y = split_features_and_target(df)

    print("X columns:", list(X.columns))
    print("y value counts:\n", y.value_counts())

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    # Quick sanity check on a known fraud row
    fraud_row = df[df["Class"] == 1].iloc[0].drop(labels=["Class"])
    sample_result = predict_transaction(fraud_row.to_dict())
    print("\nSanity check on a known fraud transaction:")
    print(sample_result)


if __name__ == "__main__":
    main()
