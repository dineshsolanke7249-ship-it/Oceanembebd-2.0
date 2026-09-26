from pathlib import Path
import joblib
import pandas as pd

MODEL_FILE = Path("models/trained/temperature_model.joblib")

FEATURES = [
    "year",
    "month",
    "day",
    "depth_m",
    "latitude",
    "longitude",
]

_MODEL = None


def load_model():
    global _MODEL

    if _MODEL is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

        _MODEL = joblib.load(MODEL_FILE)

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
