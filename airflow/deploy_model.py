import os
import subprocess
import logging

from airflow.exceptions import AirflowException


K8S_DEPLOYMENT = os.environ.get(
    "K8S_DEPLOYMENT",
    "house-service"
)

K8S_NAMESPACE = os.environ.get(
    "K8S_NAMESPACE",
    "default"
)


#----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
#----------------------------------------------------------------------------
IMAGE_URI = os.environ["IMAGE_URI"]


def run_command(command):
    logging.info("Running: %s", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.stdout:
        logging.info("STDOUT:\n%s", result.stdout)

    if result.stderr:
        logging.info("STDERR:\n%s", result.stderr)

    if result.returncode != 0:
        logging.error(
            "Command failed with exit code %s",
            result.returncode
        )

        raise AirflowException(
            f"Command failed with exit code "
            f"{result.returncode}"
        )

    return result

def run_command_with_input(command, input_text):
    logging.info("Running: %s", " ".join(command))

    result = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True
    )

    if result.stdout:
        logging.info("STDOUT:\n%s", result.stdout)

    if result.stderr:
        logging.info("STDERR:\n%s", result.stderr)

    if result.returncode != 0:
        logging.error(
            "Command failed with exit code %s",
            result.returncode
        )

        raise AirflowException(
            f"Command failed with exit code "
            f"{result.returncode}"
        )

    return result


def deploy():

    logging.info(
        "Deploying image to Kubernetes: %s",
        IMAGE_URI
    )

    deployment_path = os.path.join(
        os.environ["ROOT_PATH"],
        "k8s",
        "deployment.yaml"
    )

    with open(deployment_path, "r") as file:
        deployment_yaml = file.read()

    deployment_yaml = deployment_yaml.replace(
        "${IMAGE_URI}",
        IMAGE_URI
    )

    run_command_with_input(
        [
            "kubectl",
            "apply",
            "-f",
            "-",
            "-n",
            K8S_NAMESPACE,
        ],
        deployment_yaml
    )

    run_command([
        "kubectl",
        "apply",
        "-f",
        os.path.join(
            os.environ["ROOT_PATH"],
            "k8s",
            "service.yaml"
        ),
        "-n",
        K8S_NAMESPACE,
    ])

    run_command([
        "kubectl",
        "rollout",
        "status",
        f"deployment/{K8S_DEPLOYMENT}",
        "--timeout=300s",
        "-n",
        K8S_NAMESPACE,
    ])

    logging.info(
        "Kubernetes deployment successful: %s",
        IMAGE_URI
    )

if __name__ == "__main__":
    deploy()
