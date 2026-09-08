import os
import pandas as pd


BASE_DIR = r"D:\Mastitis"

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "herd_predictions.csv"
)


def load_predictions():
    """
    Load the latest herd predictions generated
    by mastitis_pipeline.py.
    """

    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    return df


def get_cow(cow_id):
    """
    Return prediction data for one cow.
    """

    df = load_predictions()

    result = df[df["cow_id"].astype(str) == str(cow_id)]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


def get_all_cows():
    """
    Return all cow predictions.
    """

    df = load_predictions()

    return df.to_dict(orient="records")