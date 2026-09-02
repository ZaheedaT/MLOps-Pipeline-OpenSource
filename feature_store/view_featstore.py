import os, sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from feast import FeatureStore
from feature_store import FeastFeatureStore
from feature_repo.definitions import house, house_features

# Define where you want to store your log file ------------------------------------------
LOG_FILE_PATH = os.path.join(os.getcwd(), "feast_pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # 📝 Writes logs directly to a file
        logging.StreamHandler(sys.stdout)   # 💻 Keeps streaming them to your console window
    ]
)
#-----------------------------------------------------------------------------------------------------------

def get_online_with_latency(feast_store, entity_rows: list, features: list) -> tuple:
    """Queries online store dynamically and measures exact round-trip latency."""
    start_time = time.perf_counter()
    response = feast_store.get_online_features(features=features, entity_rows=entity_rows)
    execution_time_ms = (time.perf_counter() - start_time) * 1000

    # Clean, concise metric log
    logging.info(f"ONLINE STORE FETCH | Rows: {len(entity_rows)} | Latency: {execution_time_ms:.2f} ms")
    return response.to_df()

def load_feast():
    # Initialize the feature store with yaml file
    feast = FeastFeatureStore(path=os.path.join(os.getcwd() + "//feature_store//feature_repo"))
    logging.info(f"Loaded current data in the Feature Store:\n {feast.store}")

    # Register structural metadata definitions into Feast's registry to prepare it to track specific data columns
    feast.store.apply([house, house_features])

    # Pull historical data from Offline Store
    entity_df = feast.get_entity_dataframe(
        path=os.path.join(os.getcwd() + "//feature_store//data//house_target.parquet"))

    features = [
        "house_features:area",
        "house_features:bedrooms",
        "house_features:mainroad"
    ]

    # Build a Historical DF by performing a point-in-time correct join to prevent data leakage
    hist_df = feast.get_historical_features(entity_df, features)
    logging.info(f"\n Pulled Historical Offline Data in the Feature Store:\n {hist_df}")

    # Load Offline features to the Low Latency Online Store (Materialization)
    feast.materialize(datetime.now(), datetime.now() - timedelta(days=10))

    # Format entity keys for Real-Time Online Inference
    entity_rows = pd.DataFrame(entity_df["house_id"]).to_dict(orient="records")

    # Query online store & capture both the dataframe and execution time metrics
    online_df = get_online_with_latency(feast.store, entity_rows, features)

    # Incremental update tracking, captures ONLY new updates that occurred between last materialization run and the current timestamp
    # Ensures the online store stays synchronized
    feast.materialize(end_date=datetime.now(), increment=True)
    logging.info("Incremental materialization completed successfully.")

    return online_df


if __name__ == "__main__":
    logging.info("Starting Feast execution pipeline from WSL terminal...")

    # Run the processing logic
    final_online_df = load_feast()

    logging.info(f"Pipeline completed successfully. Generated DF shape: {final_online_df.shape}")


