import sys
import os
import pandas as pd

sys.path.append(os.getcwd())

from model.house_model import HouseModel


class ModelValidation:

    def __init__(self):
        self.house_model = HouseModel()

    def validate(self):

        # Load the trained model
        model = self.house_model.load_model()

        # Load validation data
        features, target = self.get_validation_data()

        # Predict
        predictions = model.predict(features)

        # Calculate metrics
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score
        )

        rmse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        r2 = r2_score(target, predictions)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        # Model quality thresholds
        if r2 < 0.60:
            raise RuntimeError(
                f"Model validation failed: R2={r2:.4f} < 0.60"
            )

        print("Model validation passed!")

    def get_validation_data(self):

        features_path = os.path.join(
            os.getcwd(),
            "feature_store",
            "data",
            "house_features.parquet"
        )

        target_path = os.path.join(
            os.getcwd(),
            "feature_store",
            "data",
            "house_target.parquet"
        )

        features = __import__("pandas").read_parquet(
            features_path,
            columns=["area", "mainroad", "bedrooms"]
        )

        target = pd.read_parquet(
            target_path,
            columns=["price"]
        )

        return features, target


if __name__ == "__main__":
    validation = ModelValidation()
    validation.validate()
