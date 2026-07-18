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
ROOT_DATA_PATH = "creditcard.csv"


def resolve_dataset_path():
    """
    Locate the credit-card CSV on disk.

    Checks, in order:
    1. data/creditcard.csv
    2. creditcard.csv (project root)

    Returns
    -------
    str or None
        Path string if found, otherwise None.
    """
    if Path(DATA_PATH).is_file():
        return DATA_PATH
    if Path(ROOT_DATA_PATH).is_file():
        return ROOT_DATA_PATH
    return None


def load_data(path=None):
    """
    Load the credit card dataset from disk (CLI / training use).

    Parameters
    ----------
    path : str, optional
        Explicit CSV path. If omitted or missing, uses resolve_dataset_path().

    Returns
    -------
    pandas.DataFrame
    """
    if path is not None and Path(path).is_file():
        return pd.read_csv(path)

    resolved = resolve_dataset_path()
    if resolved is None:
        raise FileNotFoundError(
            f"Dataset not found. Place the file at '{DATA_PATH}' or '{ROOT_DATA_PATH}'."
        )
    return pd.read_csv(resolved)


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
