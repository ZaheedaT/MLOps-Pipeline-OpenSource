import sys
import os
#import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from feature_store.exec_feature_store import ExecuteFeatureStore
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
from monitoring.evidently_monitoring import *

#-------------------------------------------------------------------------
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
ROOT_PATH = os.environ["ROOT_PATH"]
#PYTHON_PATH = os.environ["PYTHON_PATH"]
#SCRIPT_PATH= os.path.join(ROOT_PATH, "airflow", "update_datastore.py")
DATA_PATH = os.path.join(ROOT_PATH, "data", "train.csv")
WORKSPACE = 'monitoring workspace'
PROJECT = 'monitoring project'
#-------------------------------------------------------------------------

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

class MonitorDrift:
    def __init__(self):       
        self.f_store = ExecuteFeatureStore()
        self.monitoring = Monitoring(DataDriftReport())

    def get_db_connection(self):
        engine = create_engine(DB_CONNECTION_STRING)
        return engine


    def get_reference_and_current_data(self):
        store = self.f_store.get_feature_store()

        # --------------------------------------------------
        # REFERENCE DATA
        # --------------------------------------------------


        reference_full = self.f_store.get_historical_features()
        reference = reference_full[
            ["area", "bedrooms", "mainroad"]
        ]
        # Latest timestamp in the historical/reference data
        reference_end = reference_full["event_timestamp"].max()

        print("REFERENCE END:", reference_end)

        # --------------------------------------------------
        # CURRENT DATA
        # --------------------------------------------------

        # Current = recently observed feature data
        engine = self.get_db_connection()

        with engine.connect() as connection:
            result = connection.execute(
                text("""
                    SELECT house_id
                    FROM public.house_features_sql
                    WHERE event_timestamp > :reference_end
                    ORDER BY event_timestamp
                """),
                {"reference_end": reference_end}
            )

            rows = result.fetchall()
            entity_df_cur = pd.DataFrame(
                rows,
                columns=result.keys()
            )

        print("CURRENT ENTITY DATA:")
        print(entity_df_cur)

        if entity_df_cur.empty:
            print("NO NEW DATA DETECTED")
            return reference, pd.DataFrame(columns=reference.columns)

        current = self.f_store.get_online_features(store, entity_df_cur)
        current = current[
            ["area", "bedrooms", "mainroad"]
        ]
        return reference, current


    def monitor_drift(self, reference=None, current=None):
        if reference is None or current is None:
            reference, current = self.get_reference_and_current_data()

        if current.empty:
            print("NO NEW DATA -> NO DRIFT")
            return False

        logging.info("reference:%s", reference)
        logging.info("current:%s", current)

        ws = self.monitoring.create_workspace(WORKSPACE)
        print(self.monitoring.current_strategy)
        self.monitoring.current_strategy = DataDriftTestReport()
        test_suite = self.monitoring.execute_strategy(
            reference, current, ws)
        # Check if drift is detected
        drift_detected = any(test["status"] == "FAIL"
                             for test in test_suite.as_dict()["tests"])
        return drift_detected


if __name__ == "__main__":
    os.chdir("/mnt/c/Users/zahee/coding/mlops-feedback/")
    drift_monitor = MonitorDrift()
    refr, curr = drift_monitor.get_reference_and_current_data()
    drift = drift_monitor.monitor_drift(refr, curr)
    if drift:
        logging.info("Data drift detected! Retraining required.")
        print("Data drift detected! Retraining required.")
        sys.exit(1)
    else:
        logging.info("No drift detected!")
        print("No drift detected!")
        sys.exit(0)
