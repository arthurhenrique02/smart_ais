from datetime import date
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = "model/model.joblib"


def predict_sales_quantity(pred_date: date, color: str) -> dict:
    if not Path(MODEL_PATH).exists():
        raise ValueError("Model not trained yet")

    # get loaded model
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    date_reference = saved["date_reference"]

    # saved accur
    r2 = saved.get("r2_score")
    mae = saved.get("mae")

    days_from_start = (pd.to_datetime(pred_date) - date_reference).days
    input_df = pd.DataFrame([{"color": color, "days_from_start": days_from_start}])
    prediction = model.predict(input_df)[0]

    return {
        "predicted_quantity_sold": max(0, round(prediction)),
        "model_accuracy": {
            "r2_score": round(r2, 4) if r2 is not None else None,
            "mean_absolute_error": round(mae, 2) if mae is not None else None,
        },
    }
