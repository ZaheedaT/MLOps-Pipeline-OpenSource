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
ROOT_PATH = os.environ["ROOT_PATH"]
DATA_PATH = os.path.join(ROOT_PATH, "data", "train.csv")
WORKSPACE = os.path.join(ROOT_PATH, "monitoring workspace")
PROJECT = 'monitoring project'
DB_CONNECTION_STRING= os.environ["DB_CONNECTION_STRING"]
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

        reference_full = self.f_store.get_historical_features()
        reference = reference_full[
            ["area", "bedrooms", "mainroad"]
        ]

        engine = self.get_db_connection()

        with engine.connect() as connection:
            checkpoint = connection.execute(
                text("""
                    SELECT last_processed_timestamp
                    FROM drift_monitor_state
                    WHERE id = 1
                """)
            ).fetchone()

            if checkpoint and checkpoint[0] is not None:
                last_processed_timestamp = checkpoint[0]
            else:
                last_processed_timestamp = reference_full["event_timestamp"].max()

            logging.info(
                "LAST PROCESSED TIMESTAMP: %s",
                last_processed_timestamp
            )

            result = connection.execute(
                text("""
                    SELECT house_id, event_timestamp
                    FROM public.house_features_sql
                    WHERE event_timestamp > :last_processed_timestamp
                    ORDER BY event_timestamp
                """),
                {
                    "last_processed_timestamp": last_processed_timestamp
                }
            )

            rows = result.fetchall()
            entity_df_cur = pd.DataFrame(
                rows,
                columns=result.keys()
            )

        logging.info("CURRENT ENTITY DATA:")
        logging.info(entity_df_cur)

        if entity_df_cur.empty:
            logging.info("NO NEW DATA DETECTED")
            return reference, pd.DataFrame(columns=reference.columns), None

        current = self.f_store.get_online_features(
            store,
            entity_df_cur[["house_id"]]
        )

        current = current[
            ["area", "bedrooms", "mainroad"]
        ]

        newest_timestamp = entity_df_cur["event_timestamp"].max()

        return reference, current, newest_timestamp

    def monitor_drift(self, reference=None, current=None, newest_timestamp=None):
        if reference is None or current is None:
            reference, current, newest_timestamp = self.get_reference_and_current_data()

        if current.empty:
            logging.info("NO NEW DATA -> NO DRIFT")
            return False



        logging.info("reference:%s", reference)
        logging.info("current:%s", current)

        ws = self.monitoring.create_workspace(WORKSPACE)

        # Create the Evidently Data Drift Report
        self.monitoring.current_strategy = DataDriftReport()

        drift_report = self.monitoring.execute_strategy(
            reference,
            current,
            ws
        )

        # Create the Evidently Test Suite
        self.monitoring.current_strategy = DataDriftTestReport()

        test_suite = self.monitoring.execute_strategy(
            reference,
            current,
            ws
        )

        # Determine whether drift was detected
        drift_detected = any(
            test["status"] == "FAIL"
            for test in test_suite.as_dict()["tests"]
        )

        if newest_timestamp is not None:
            with self.get_db_connection().connect() as connection:
                transaction = connection.begin()

                try:
                    connection.execute(
                        text("""
                            INSERT INTO drift_monitor_state
                                (id, last_processed_timestamp)
                            VALUES
                                (1, :timestamp)
                            ON CONFLICT (id)
                            DO UPDATE SET
                                last_processed_timestamp = :timestamp
                        """),
                        {"timestamp": newest_timestamp}
                    )

                    transaction.commit()

                except Exception:
                    transaction.rollback()
                    raise

            logging.info(
                "CHECKPOINT UPDATED: %s",
                newest_timestamp
            )

        return drift_detected

        return drift_detected


if __name__ == "__main__":
    os.chdir("/mnt/c/Users/zahee/coding/mlops-feedback/")
    drift_monitor = MonitorDrift()

    refr, curr, newest_timestamp = drift_monitor.get_reference_and_current_data()
    drift = drift_monitor.monitor_drift(
        refr,
        curr,
        newest_timestamp
    )
    if drift:
        logging.info("Data drift detected! Retraining required.")
        print("Data drift detected! Retraining required.")
        sys.exit(1)
    else:
        logging.info("No drift detected!")
        print("No drift detected!")
        sys.exit(0)
