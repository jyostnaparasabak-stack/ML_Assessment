from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"])
        result["month"] = result["date"].dt.month
        result["day_of_year"] = result["date"].dt.dayofyear
        result["week_of_year"] = result["date"].dt.isocalendar().week.astype(int)
    if {"pickup", "delivery"}.issubset(result.columns):
        result["route"] = result["pickup"] + "->" + result["delivery"]
    if "distance" in result.columns:
        result["is_long_haul"] = (result["distance"] > 1500).astype(int)
    return result


def fill_missing_prediction_columns(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    result = df.copy()
    for col in ["market_index", "quote_signal"]:
        if col not in result.columns:
            result[col] = medians[col]
    return result


def build_pipeline() -> Pipeline:
    numeric_cols = [
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "month",
        "day_of_year",
        "week_of_year",
        "is_long_haul",
    ]
    categorical_cols = ["pickup", "delivery", "equipment", "route"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_model() -> tuple[Pipeline, dict]:
    train = engineer_features(pd.read_csv("train-test.csv"))
    cutoff = train["date"].quantile(0.8)
    train_split = train[train["date"] <= cutoff].copy()
    holdout = train[train["date"] > cutoff].copy()

    features = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "month",
        "day_of_year",
        "week_of_year",
        "route",
        "is_long_haul",
    ]

    model = build_pipeline()
    model.fit(train_split[features], train_split["posted_rate"])
    pred = model.predict(holdout[features])

    metrics = {
        "mae": mean_absolute_error(holdout["posted_rate"], pred),
        "rmse": mean_squared_error(holdout["posted_rate"], pred) ** 0.5,
        "r2": r2_score(holdout["posted_rate"], pred),
    }
    return model, metrics


def generate_outputs() -> None:
    train = engineer_features(pd.read_csv("train-test.csv"))
    validation = engineer_features(pd.read_csv("validation.csv"))
    december = engineer_features(pd.read_csv("december-chart-inputs.csv"))

    medians = train[["market_index", "quote_signal"]].median()
    validation = fill_missing_prediction_columns(validation, medians)
    december = fill_missing_prediction_columns(december, medians)

    features = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "month",
        "day_of_year",
        "week_of_year",
        "route",
        "is_long_haul",
    ]

    model, metrics = evaluate_model()
    model.fit(train[features], train["posted_rate"])

    validation["predicted_rate"] = model.predict(validation[features])
    validation_output = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": validation["predicted_rate"],
    })
    validation_output = validation_output.sort_values("load_id").reset_index(drop=True)
    template = pd.read_csv("validation-predictions-template.csv")
    final_predictions = template[["load_id"]].merge(validation_output, on="load_id", how="left")
    final_predictions["predicted_rate"] = final_predictions["predicted_rate"].clip(lower=1e-6)
    final_predictions.to_csv("validation_predictions.csv", index=False)

    december["predicted_rate"] = model.predict(december[features])
    december = december[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
    december = december.sort_values("date").reset_index(drop=True)
    december["predicted_rate"] = december["predicted_rate"].clip(lower=1e-6)
    december.to_csv("december-chart-inputs.csv", index=False)

    print("Holdout validation metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    print("Saved validation_predictions.csv and filled december-chart-inputs.csv")


if __name__ == "__main__":
    generate_outputs()
