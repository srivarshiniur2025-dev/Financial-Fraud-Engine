"""
utils.py
--------
Helper functions for the Financial Fraud Intelligence Engine.
Includes data loading, dataset summary, charts, and risk scoring.
"""

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/creditcard.csv"


def load_data(path=DATA_PATH):
    """
    Load the credit card dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file (default: data/creditcard.csv).

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    df = pd.read_csv(path)
    return df


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
