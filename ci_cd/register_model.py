import sys
import os
import logging

sys.path.append(os.getcwd())

from model.house_model import HouseModel
from serving.model_serving import BentoModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    house_model = HouseModel()

    # Configure MLflow
    house_model.configure_mlflow()

    # Register model
    model_info = house_model.register()

    logger.info(
        f"MLflow model registered: {model_info.model_uri}"
    )

    # Import registered model into BentoML
    bento_model = BentoModel()

    bento_model_name = bento_model.import_model(
        "house_price_model",
        model_info.model_uri
    )

    logger.info(
        f"Bento model imported: {bento_model_name}"
    )
