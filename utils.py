"""
utils.py
--------
Helper functions for the Financial Fraud Intelligence Engine.
Includes data loading, dataset summary, charts, and risk scoring.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/creditcard.csv"


def dataset_exists(path=DATA_PATH):
    """Return True if the dataset file is present on disk."""
    return Path(path).is_file()


def load_dataset(path=None, file_like=None):
    """
    Load the credit card dataset from a path or an uploaded file-like object.

    Parameters
    ----------
    path : str, optional
        Filesystem path to the CSV. Defaults to DATA_PATH when file_like is None.
    file_like : file-like, optional
        An uploaded CSV (e.g. from st.file_uploader). Takes precedence over path.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If loading from path and the file does not exist.
    ValueError
        If neither a valid path nor a file-like object is provided.
    """
    if file_like is not None:
        return pd.read_csv(file_like)

    resolved = DATA_PATH if path is None else path
    if not dataset_exists(resolved):
        raise FileNotFoundError(f"Dataset not found at {resolved}")
    return pd.read_csv(resolved)


def load_data(path=DATA_PATH):
    """
    Load the credit card dataset from a CSV file.

    Thin wrapper around load_dataset for backward compatibility.
    """
    return load_dataset(path=path)


def get_dataset_summary(df):
    """
    Compute basic summary statistics for the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The credit card dataset.

    Returns
    -------
    dict
        A dictionary with:
        - num_rows
        - num_columns
        - genuine_transactions
        - fraudulent_transactions
        - missing_values
    """
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "genuine_transactions": int((df["Class"] == 0).sum()),
        "fraudulent_transactions": int((df["Class"] == 1).sum()),
        "missing_values": int(df.isnull().sum().sum()),
    }


def plot_class_distribution(df):
    """
    Create a bar chart comparing Genuine vs Fraud transactions.

    Parameters
    ----------
    df : pandas.DataFrame
        The credit card dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The bar chart figure.
    """
    counts = df["Class"].value_counts().sort_index()
    labels = ["Genuine", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=["steelblue", "crimson"])
    ax.set_title("Genuine vs Fraud Transactions")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_amount_distribution(df):
    """
    Create a histogram of transaction amounts.

    Parameters
    ----------
    df : pandas.DataFrame
        The credit card dataset.

    Returns
    -------
    matplotlib.figure.Figure
        The histogram figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["Amount"], bins=50, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of Transaction Amounts")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def calculate_risk_score(probability):
    """
    Convert a fraud probability (0 to 1) into a risk score (0 to 100).

    Parameters
    ----------
    probability : float
        Fraud probability between 0 and 1.

    Returns
    -------
    int
        Risk score between 0 and 100.
    """
    return int(round(float(probability) * 100))
