import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import sqlalchemy as db
import pickle
import logging
import subprocess
import boto3

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#-------------------------------------------------------------------------
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
ROOT_PATH = os.environ["ROOT_PATH"]
PYTHON_PATH = os.environ["PYTHON_PATH"]
DATA_PATH = os.path.join(ROOT_PATH, "data", "train.csv")

AWS_REGION = os.environ["AWS_REGION"]
ECR_REGISTRY = os.environ["ECR_REGISTRY"]
ECR_REPOSITORY = os.environ.get("ECR_REPOSITORY", "house-price-model")
#-------------------------------------------------------------------------

from feature_store.exec_feature_store import ExecuteFeatureStore
from model.house_model import HouseModel
from serving.model_serving import BentoModel

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)


class TrainModel:
    def __init__(self):
        self.params = None
        self.f_store = ExecuteFeatureStore()
        self.house_model = HouseModel()

    def upload_model_to_s3(self):
        model_path = os.path.join(
            ROOT_PATH,
            "model",
            "house_regression_model.pkl"
        )

        bucket = os.environ["MODEL_S3_BUCKET"]
        key = os.environ.get(
            "MODEL_S3_KEY",
            "mlops/house-price/model/house_regression_model.pkl"
        )

        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION
        )

        s3.upload_file(
            model_path,
            bucket,
            key
        )

        print(f"Uploaded retrained model to s3://{bucket}/{key}")

    def get_current_features(self):
        engine = db.create_engine(DB_CONNECTION_STRING)
        logging.info("engine initialized")

        connection = engine.raw_connection()

        try:
            Y_hist = pd.read_sql(
                "SELECT house_id, price FROM public.house_target_sql",
                con=connection
            )
        finally:
            connection.close()
            engine.dispose()

        store = self.f_store.get_feature_store()
        logging.info("feature store initialized")

        X_hist = self.f_store.get_online_features(
            store,
            pd.DataFrame(Y_hist["house_id"])
        )

        X_hist["price"] = Y_hist["price"]

        return X_hist
    def predict_new_data(self):
        path = os.path.join(ROOT_PATH, "serving", "feedback.csv")
        X_new = pd.read_csv(path)
        X_new.drop(["event_timestamp", "prediction"], axis=1, inplace=True)
        lr_model = self.house_model.load_model()
        X_new = X_new[lr_model.feature_names_in_]
        Y_new = self.house_model.predict(X_new)
        X_new["proxy_target"] = Y_new
        return X_new  

    def create_and_train_new_dataset_with_target(self, X_hist, X_new):  
        last_id = int(X_hist.loc[X_hist["house_id"].idxmax()]["house_id"])
        X_new["house_id"] = range(last_id+1, last_id + len(X_new)+1)
        new_data_combined = X_new.copy()
        new_data_combined["price"] = new_data_combined["proxy_target"]      
        historical_data_combined = X_hist.copy()
        combined_data = pd.concat([historical_data_combined, new_data_combined], ignore_index=True)
        X_combined = combined_data.drop(columns=["price", "proxy_target", "house_id"])
        y_combined = combined_data["price"]
        print(combined_data)
        #self.train_model(X_combined, y_combined)
        self.house_model.train_model(X_combined, y_combined, test_size=0.1)
        print("Model re-trained and saved as model.pkl")
        logging.info("Model re-trained and saved as model.pkl")

    def train_model(self, x, y):
        self.params = {
            "fit_intercept": True,
            "positive": False
        }
        model = LinearRegression(**self.params)
        logging.info("train before called")
        model.fit(x, y)
        logging.info("train data called")
        with open("model/house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model re-trained and saved as model.pkl")
        logging.info("Model re-trained and saved as model.pkl")

    def register_model(self):
        self.house_model.configure_mlflow()
        model_info = self.house_model.register()
        return model_info
    
    def serve_model(self, model_info):
        bento_model = BentoModel()
        model_name = bento_model.import_model("house_price_model", model_info.model_uri)
        logging.info(str.format('Imported model {0} to BentoML', model_name))

    def run_command(self, command, input_text=None):
        result = subprocess.run(
            command,
            cwd=ROOT_PATH,
            input=input_text,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logging.info(result.stdout)

        if result.stderr:
            logging.info(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(command)}\n"
                f"Return code: {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        return result.stdout.strip()

    def build_bento(self):
        logging.info("Building BentoML image.")

        output = self.run_command(
            ["bentoml", "build", "serving/", "--output", "tag"]
        )

        bento_tag = output.removeprefix("__tag__:")

        logging.info("BentoML build completed: %s", bento_tag)

        return bento_tag

    def containerize_bento(self, bento_tag):
        logging.info("Containerizing Bento: %s", bento_tag)

        self.run_command(
            ["bentoml", "containerize", bento_tag]
        )

        logging.info("Bento containerization completed.")

    def login_to_ecr(self):
        logging.info("Logging into Amazon ECR.")

        password = self.run_command(
            [
                "aws",
                "ecr",
                "get-login-password",
                "--region",
                AWS_REGION
            ]
        )

        self.run_command(
            [
                "docker",
                "login",
                "--username",
                "AWS",
                "--password-stdin",
                ECR_REGISTRY
            ],
            input_text=password
        )

        logging.info("ECR login completed.")

    def push_to_ecr(self, bento_tag):
        bento_version = bento_tag.split(":", 1)[1]
        image_tag = f"airflow-{bento_version}"

        image_uri = (
            f"{ECR_REGISTRY}/"
            f"{ECR_REPOSITORY}:"
            f"{image_tag}"
        )

        logging.info("Tagging image as: %s", image_uri)

        self.run_command(
            [
                "docker",
                "tag",
                bento_tag,
                image_uri
            ]
        )

        logging.info("Pushing image to ECR: %s", image_uri)

        self.run_command(
            [
                "docker",
                "push",
                image_uri
            ]
        )

        logging.info(
            "Successfully pushed image to ECR: %s",
            image_uri
        )

        return image_uri

    def build_and_push_ecr_image(self):
        bento_tag = self.build_bento()

        self.containerize_bento(bento_tag)

        self.login_to_ecr()

        image_uri = self.push_to_ecr(bento_tag)

        logging.info(
            "Runtime model image ready for deployment: %s",
            image_uri
        )

        return image_uri

if __name__ == "__main__":

    os.chdir(ROOT_PATH)
    trainer = TrainModel()
    X_hist = trainer.get_current_features()
    X_new = trainer.predict_new_data()
    trainer.create_and_train_new_dataset_with_target(X_hist, X_new)
    trainer.upload_model_to_s3()
    m_info = trainer.register_model()
    trainer.serve_model(m_info)
    image_uri = trainer.build_and_push_ecr_image()
    logging.info(
        "Model trained, registered, containerized and pushed to ECR"
    )

    print(f"DEPLOY_IMAGE_URI={image_uri}")

