import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow import AirflowException
from datetime import datetime, timedelta
import subprocess
import logging

#----------------------------------------------------------------------------------------------------------------------
# ENV VARIABLES
ROOT_PATH = os.environ["ROOT_PATH"]
PYTHON_PATH = os.environ["PYTHON_PATH"]

#----------------------------------------------------------------------------------------------------------------------

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

def check_data_push():
    script_path = os.path.join(ROOT_PATH,"/airflow/update_datastore.py")
    result = subprocess.run([PYTHON_PATH, script_path], capture_output=True, text=True)
    
    if result.stdout.endswith("Data pushed successfully\n"):
        logging.info(result.stderr)
        return "update_dashboard"
    else:
        logging.info(result.stderr)
        return "no_update"

def update_live_dashboard():
    print("FIX THESE DAMN PATHS - pushdata-dag")

    script_path = os.path.join(ROOT_PATH,"/airflow/live_dashboard.py")
    result = subprocess.run([PYTHON_PATH, script_path], capture_output=True, text=True)
    

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5),
}

with DAG(
    'push_features',
    default_args=default_args,
    description='A DAG to push data to feature store',
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id='push_feedbac_data',
        python_callable=check_data_push
    )
    update_dashboard_task = PythonOperator(
        task_id="update_dashboard",
        python_callable=update_live_dashboard
    )
    no_update_task = EmptyOperator(
        task_id='no_update'
    )

#check_drift_task >> [update_dashboard_task, no_update_task]