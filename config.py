ROOT_PATH = os.environ["ROOT"]
PYTHON_PATH = os.environ["PYTHON_PATH"]

#Feast
ENTITY_NAME = ""
OFFLINE_DB = "feast_offline"
DB_CONNECTION_STRING = "postgresql+psycopg2://postgres:73200@localhost:5432/feast_offline"
FEATURE_TABLE = "house_features_sql"
TARGET_TABLE = "house_target_sql"
FEATURE_LIST = [ 
            "house_features:area", 
            "house_features:bedrooms", 
            "house_features:mainroad" 
        ] 


#MLflow
EXPERIMENT_NAME = "House price prediction"
EXPERIMENT_URI = "http://localhost:5000"
REGISTERED_MODEL_NAME = "house_price_prediction"


#evidently ai
WORKSPACE = "monitoring workspace"
PROJECT = "live dashboard"
COLLECTOR_ID = "house_ev"
COLLECTOR_QUAL_ID = "house_ev_qual"
COLLECTOR_TGT_ID = "house_ev_tgt"
COLLECTOR_REG_ID = "house_ev_reg"
COLLECTOR_TEST_ID = "house_ev_test"

#logs
LOG_FILE = "app.log"
