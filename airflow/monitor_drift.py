import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from feature_store.exec_feature_store import ExecuteFeatureStore
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging
from monitoring.evidently_monitoring import *

#-------------------------------------------------------------------------
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
ROOT_PATH = os.environ["ROOT"]
PYTHON_PATH = os.environ["PYTHON_PATH"]
SCRIPT_PATH= os.path.join(ROOT_PATH, "airflow", "update_datastore.py")
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

        entity_df_cur = pd.read_sql(
            text("""
            SELECT house_id
            FROM public.house_features_sql
            WHERE event_timestamp > :reference_end
            ORDER BY event_timestamp
            """),
            con=engine,
            params={"reference_end": reference_end}
        )

        print("CURRENT ENTITY DATA:")
        print(entity_df_cur)

        current = self.f_store.get_online_features(store, entity_df_cur)
        current = current[
            ["area", "bedrooms", "mainroad"]
        ]
        return reference, current

    def monitor_drift(self, reference=None, current=None):
        if(reference is None or current is None):
            reference, current = self.get_reference_and_current_data()

        # -------------DEBUG -------------------
        print("\n========== REFERENCE ==========")
        print(reference.head())
        print(reference.shape)
        print(reference.describe(include="all"))

        print("\n========== CURRENT ==========")
        print(current.head())
        print(current.shape)
        print(current.describe(include="all"))
        #-----------------DEBUG---------------------
        logging.info("reference:%s", reference)
        logging.info("reference:%s", current)
        ws = self.monitoring.create_workspace(WORKSPACE)
        project = self.monitoring.search_or_create_project(PROJECT, ws)
        #Data drift report
        print(self.monitoring.current_strategy)
        drift = self.monitoring.execute_strategy(reference, current, ws)
        #Data drift test report
        self.monitoring.current_strategy = DataDriftTestReport()
        test_suite = self.monitoring.execute_strategy(reference, current, ws)
        # Check if drift is detected
        drift_detected = any(test["status"] == "FAIL" for test in test_suite.as_dict()["tests"])
        return drift_detected

if __name__ == "__main__":
    print("EDWIN STUFF monitor_drift.py")
    os.chdir("/mnt/c/Users/zahee/coding/mlops-feedback/")
    drift_monitor = MonitorDrift()
    refr, curr = drift_monitor.get_reference_and_current_data()
    drift = drift_monitor.monitor_drift(refr, curr)
    if(drift):
        logging.info("Data drift detected! Retraining required.")
        print("Data drift detected! Retraining required.")
    else:
        logging.info("No drift detected!")
        print("No drift detected!")
