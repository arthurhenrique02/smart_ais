import random
from datetime import datetime, timedelta

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = "model/model.joblib"


def train_model(data: pd.DataFrame) -> dict:
    data["days_from_start"] = (
        pd.to_datetime(data["date"]) - pd.to_datetime(data["date"]).min()
    ).dt.days
    features = ["days_from_start", "color"]
    target = "quantity_sold"

    preprocessor = ColumnTransformer(
        [("color_ohe", OneHotEncoder(handle_unknown="ignore"), ["color"])],
        remainder="passthrough",
    )

    # Pipeline to preprocess data and train model
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    X = data[features]
    y = data[target]

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    # Evaluate accuracy on training data
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Save model and metrics
    joblib.dump(
        {
            "model": pipeline,
            "date_reference": pd.to_datetime(data["date"]).min(),
            "r2_score": r2,
            "mae": mae,
        },
        MODEL_PATH,
    )

    return {
        "message": "Model trained successfully.",
        "r2_score": round(r2, 4),
        "mean_absolute_error": round(mae, 2),
    }


def generate_random_sales_csv(
    filename="sample_quantity_sales.csv", start_date="2024-01-01", num_days=100
):
    """
    A function responsible for generating random data to simulate sales of different colors.

    @param filename: The local/name of the CSV file to save the data.
    @param start_date: The start date for the data generation in YYYY-MM-DD format.
    @param num_days: The number of days to generate data for.
    @return: None
    """

    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "purple",
        "orange",
        "pink",
        "gray",
        "brown",
        "cyan",
        "magenta",
        "navy",
        "teal",
        "maroon",
        "lime",
        "gold",
        "silver",
        "beige",
        "turquoise",
        "indigo",
    ]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for i in range(num_days):
        day = start + timedelta(days=i)
        for color in colors:
            # Simulate demand with color-specific bias
            base = random.randint(0, 10)
            if color in ["red", "black", "gold"]:
                base += 5
            elif color in ["green", "cyan", "beige"]:
                base -= 2

            quantity = max(0, base + random.randint(-1, 2))
            rows.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "color": color,
                    "quantity_sold": quantity,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    generate_random_sales_csv("assets/sample_sales2.csv", num_days=500)
