from pathlib import Path
import os

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download


MODEL_FILE = Path("models/trained/temperature_model.joblib")

HF_REPO_ID = "dineshsolanke/oceanembed-temperature-model"
HF_MODEL_FILENAME = "temperature_model.joblib"

FEATURES = [
    "year",
    "month",
    "day",
    "depth_m",
    "latitude",
    "longitude",
]

_MODEL = None


def get_hf_token():
    """
    Get the Hugging Face token.

    Local development:
        Uses HF_TOKEN environment variable if available.

    Streamlit:
        Uses HF_TOKEN from Streamlit secrets.
    """

    token = os.getenv("HF_TOKEN")

    if token:
        return token

    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def get_model_path():
    """
    Use the local model when available.
    Otherwise download the model from Hugging Face.
    """

    if MODEL_FILE.exists():
        return MODEL_FILE

    token = get_hf_token()

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model",
            token=token,
        )
    )


def load_model():
    global _MODEL

    if _MODEL is None:
        model_path = get_model_path()
        _MODEL = joblib.load(model_path)

    return _MODEL


def predict_temperature(
    year: int,
    month: int,
    day: int,
    depth_m: float,
    latitude: float,
    longitude: float,
) -> float:

    model = load_model()

    features = pd.DataFrame([{
        "year": year,
        "month": month,
        "day": day,
        "depth_m": depth_m,
        "latitude": latitude,
        "longitude": longitude,
    }])

    return float(model.predict(features)[0])


def predict_temperature_batch(
    year: int,
    month: int,
    day: int,
    depths,
    latitude: float,
    longitude: float,
):

    model = load_model()

    features = pd.DataFrame({
        "year": [year] * len(depths),
        "month": [month] * len(depths),
        "day": [day] * len(depths),
        "depth_m": depths,
        "latitude": [latitude] * len(depths),
        "longitude": [longitude] * len(depths),
    })

    return model.predict(features)


if __name__ == "__main__":
    prediction = predict_temperature(
        year=2022,
        month=6,
        day=1,
        depth_m=10.0,
        latitude=0.5,
        longitude=60.5,
    )

    print(f"Predicted temperature: {prediction:.4f} C")