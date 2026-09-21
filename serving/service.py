import bentoml
import numpy as np
import csv 
from datetime import datetime 
import pandas as pd 
from itertools import starmap 

@bentoml.service(
    resources={"cpu": "2"}, 
    traffic={"timeout": 10},
    logging={
    "access": {
        "enabled": True,
        "request_content_length": True,
        "request_content_type": True,
        "response_content_length": True,
        "response_content_type": True,
        "skip_paths": ["/metrics", "/healthz", "/livez", "/readyz"],
        "format": {
            "trace_id": "032x",
            "span_id": "016x"
        }
    }
})
class HouseService:
    bento_model = bentoml.models.BentoModel("house_price_model:latest")
    #bento_model = bentoml.models.get("house_price_model:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()
        with open('feedback.csv', 'w', newline='') as file: 
            fieldnames = ["event_timestamp", "area", "bedrooms", "mainroad", "prediction"] 
            writer = csv.DictWriter(file, fieldnames = fieldnames) 
            writer.writeheader()

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        input_df = pd.DataFrame(
            input_data,
            columns=["mainroad", "area", "bedrooms"]
        )

        pred = self.model.predict(input_df)
        print(pred)

        timestamp = pd.Timestamp.now()

        with open("feedback.csv", "a", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)

            data = input_df.iloc[0]

            writer.writerow([
                timestamp,
                data["area"],
                data["bedrooms"],
                data["mainroad"],
                pred[0]
            ])

        return np.asarray(pred)