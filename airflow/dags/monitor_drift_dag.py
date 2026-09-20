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

def monitor_drift():
    script_path = os.path.join(
        ROOT_PATH,
        "airflow",
        "monitor_drift.py"
    )

    result = subprocess.run(
        [PYTHON_PATH, script_path],
        capture_output=True,
        text=True
    )

    print("========== MONITOR STDOUT ==========")
    print(result.stdout)

    print("========== MONITOR STDERR ==========")
    print(result.stderr)

    print("========== RETURN CODE ==========")
    print(result.returncode)

    if result.returncode == 1:
        print("DRIFT DETECTED -> RETRAIN")
        return "trigger_retrain"

    if result.returncode == 0:
        print("NO DRIFT -> NO RETRAIN")
        return "no_retrain"


    raise AirflowException(
            f"monitor_drift.py failed with exit code {result.returncode}"
        )
    
def retrain_model():
    script_path = os.path.join(ROOT_PATH,"airflow","train_model.py")

    # Run the command using a list
    result = subprocess.run(
        [PYTHON_PATH, script_path], capture_output=True, text=True)

    logging.info("========== RETRAIN STDOUT ==========")
    logging.info(result.stdout)

    logging.info("========== RETRAIN STDERR ==========")
    logging.info(result.stderr)

    if result.returncode != 0:
        raise AirflowException(
            f"Model retraining failed with exit code {result.returncode}"
        )

    image_uri = None

    for line in result.stdout.splitlines():
        if line.startswith("DEPLOY_IMAGE_URI="):
            image_uri = line.split("=", 1)[1].strip()
            break
    if not image_uri:
        raise AirflowException(
            "Retraining model succeeded but no ECR image URI was returned."
        )

    logging.info("New runtime model image: %s",
                 image_uri)

    return image_uri


def deploy_model(ti):
    image_uri = ti.xcom_pull(
        task_ids="trigger_retrain"
    )

    if not image_uri:
        raise AirflowException(
            "No ECR image URI received from retraining task."
        )

    logging.info(
        "Deploying image to Kubernetes: %s",
        image_uri
    )

    script_path = os.path.join(
        ROOT_PATH,
        "airflow",
        "deploy_model.py"
    )

    env = os.environ.copy()
    env["IMAGE_URI"] = image_uri

    result = subprocess.run(
        [PYTHON_PATH, script_path],
        capture_output=True,
        text=True,
        env=env
    )

    logging.info("========== DEPLOY STDOUT ==========")
    logging.info(result.stdout)

    logging.info("========== DEPLOY STDERR ==========")
    logging.info(result.stderr)

    if result.returncode != 0:
        raise AirflowException(
            f"Kubernetes deployment failed with exit code "
            f"{result.returncode}"
        )

    logging.info(
        "Kubernetes deployment completed successfully."
    )

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    'check_drift_and_retrain',
    default_args=default_args,
    description="Monitor data drift, retrain when required, and deploy to Kubernetes",
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id='check_data_drift',
        python_callable=monitor_drift
    )
    retrain_task = PythonOperator(
        task_id='trigger_retrain',
        python_callable=retrain_model
    )
    no_retrain_task = EmptyOperator(
        task_id='no_retrain'
    )
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model
    )

    pipeline_complete = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success", # Is used because BranchPythonOperator skips the branch that wasn't selected.
    )

    check_drift_task >> [retrain_task, no_retrain_task]
    retrain_task >> deploy_model_task
    [no_retrain_task, deploy_model_task] >> pipeline_complete