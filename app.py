import os
from datetime import datetime,timedelta
import sys
from importlib import reload
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from eda.data_ingestor import *
from eda.data_inspection import DataInspector, DataTypeInspection, SummaryDataInspection
from eda.data_analysis import AnalysisContext, NumericalUnivariateAnalysis, CategoricalUnivariateAnalysis, BivariateHeatmapAnalysis
from eda.missing_value_handling import *
from eda.data_encoding import DataEncoding

from feast import FeatureStore

import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model, load_model
from model.house_model import HouseModel

import bentoml
from bentoml import HTTPServer
from serving.model_serving import BentoModel

from monitoring.evidently_monitoring import *

from view_featstore import load_feast

def eda_steps():
    path = os.getcwd() + "/data/Housing.zip"
    file_ext = os.path.splitext(path)[1]
    ingestor_type = DataIngestorFactory.get_data_ingestor(file_ext)
    df = ingestor_type.ingest(path)
    print(df.head())

    data_inspector = DataInspector(DataTypeInspection())
    data_inspector.execute_strategy(df)

    data_inspector.set_strategy(SummaryDataInspection())
    data_inspector.execute_strategy(df)

    univariate = AnalysisContext(NumericalUnivariateAnalysis)
    univariate.analyze_strategy(df, "price")

    univariate_cat = AnalysisContext(CategoricalUnivariateAnalysis)
    univariate_cat.analyze_strategy(df, "guestroom")

    int_cols = df.select_dtypes(include=np.number).columns
    heatmap = AnalysisContext(BivariateHeatmapAnalysis)
    heatmap.analyze_strategy(df, int_cols)

    is_missing = MissingValueContext(None)
    print(is_missing.check_missing(df))

    missing_value_handling = MissingValueContext(DropMissingValueStrategy)
    newdf = missing_value_handling.execute(df)


    missing_value_handling.set_strategy(FillMissingValueStrategy)
    out_df = missing_value_handling.execute(df)

    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    cat_columns = ['furnishingstatus']
    numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    encode = DataEncoding()
    bin_df = encode.binary_encoding(out_df, binary_columns)

    cat_df = encode.categorical_encoding(bin_df, cat_columns)

    # num_df = encode.numerical_scaling(cat_df, numerical_columns)
    # num_df

    timestamps = pd.date_range(
        end=pd.Timestamp.now(),
        start=pd.Timestamp.now(),
        periods=len(cat_df),
        freq=None).to_frame(name="event_timestamp", index=False)

    cat_df["event_timestamp"] = timestamps.event_timestamp
    cat_df["house_id"] = range(1, len(cat_df) + 1)  # Assign unique IDs to each house

    # Splitting the dataset into features (X) and target (y)
    X = cat_df.drop(columns=['price'])
    y = pd.DataFrame(cat_df[['house_id', 'event_timestamp', 'price']])

    # Display the first few rows of the preprocessed features
    X.head(), y.head()

    import sqlalchemy as db
    connstr = 'postgresql+psycopg://postgres:73200@localhost:5432/feast_offline'
    engine = db.create_engine(connstr)
    X.to_sql('house_features_sql', engine, if_exists='replace', index=False)
    y.to_sql('house_target_sql', engine, if_exists='replace', index=False)

    # Writing our DataFrames to parquet files
    X.to_parquet(path=os.path.join(os.getcwd(), 'feature_store/data/house_features.parquet'))
    y.to_parquet(path=os.path.join(os.getcwd(), 'feature_store/data/house_target.parquet'))




def mlflower(online_df=load_feast()):
    # Separating the features and labels
    target = entity_df['price']
    online_features = online_df.drop(labels=["house_id"], axis=1)

    model = HouseModel()
    model.train_model(online_features, target)
    print(model.x_train.shape)
    print(model.x_test.shape)

    y_train_pred = model.predict(model.x_train)
    # train_metrics = model.metrics(y_train_pred)
    # train_metrics

    y_pred = model.predict(model.x_test)
    test_metrics = model.metrics(y_pred)
    model.configure_mlflow()
    model_info = model.register()



def bento():
    bento_model = BentoModel()
    model_name = bento_model.import_model("house_price_model", model_info.model_uri)
    # load mlflow model
    b_model = bento_model.load_model(model_name)
    pred = bento_model.predict(b_model, model.x_test[:1])

    # get bento mlflow model
    b_runner = bento_model.get_model(model_name)
    b_runner.predict.run(model.x_test[:1])

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.predict(
            input_data=model.x_test[1:2],
        )
        print(result)

    # start the server from python or from terminal which I did
    '''
    server = HTTPServer("house_service:latest", production=True, port=3000, host='127.0.0.1')
    server.start()
    client = server.get_client()

    with server.start() as client:
        result = client.predict(model.x_test[:1])
        print(result)

    '''

    import requests
    BENTO_API_URL = "http://localhost:3000/predict"
    response = requests.post(BENTO_API_URL, json={
        "input_data": [
            [
                500.2,
                1,
                1.2
            ]
        ]})
    if response.status_code == 200:
        prediction = response.json()[0]
        print(prediction)


def evidentler():
    import pprint
    import importlib
    # importlib.reload(monitoring.evidently_monitoring)
    import monitoring.evidently_monitoring
    monitoring = Monitoring()

    ws = monitoring.create_workspace("monitoring workspace")
    project = monitoring.search_or_create_project("monitoring project", ws)
    print("Project: ", project)

    reference = model.x_train.copy()
    reference["price"] = model.y_train.copy()

    current = model.x_test.copy()
    current["price"] = model.y_test.copy()

    # Data drift report
    print(monitoring.current_strategy)
    drift_report = monitoring.execute_strategy(reference, current, ws)

    pprint.pp("Data Drift Report", drift_report.as_dict())

    monitoring.set_strategy = DataQualityReport()
    qual_report = monitoring.execute_strategy(reference, current, ws)
    print(qual_report)

    # Regression report
    reference_with_pred = reference.copy()
    reference_with_pred["prediction"] = y_train_pred
    reference_with_pred

    current_with_pred = current.copy()
    current_with_pred["prediction"] = y_pred
    current_with_pred

    column_mapping = ColumnMapping()
    column_mapping.target = "price"
    column_mapping.prediction = "prediction"

    monitoring.set_strategy = RegressionReport()
    reg_report = monitoring.execute_strategy(reference_with_pred, current_with_pred, ws, column_mapping)
    reg_report

    # Target Drift Report
    monitoring.set_strategy = TargetDriftReport()
    target_report = monitoring.execute_strategy(reference_with_pred, current_with_pred, ws, column_mapping)
    target_report

    # Test suite
    monitoring.set_strategy = DataDriftTestReport()
    test_report = monitoring.execute_strategy(reference, current, ws)
    test_report

    from evidently.ui.dashboards import CounterAgg, PlotType
    from evidently.renderers.html_widgets import WidgetSize

    # Title panel
    monitoring.add_dashboard_panel(
        project, panel_type="Counter",
        title="House price Monitoring dashboard",
        tags=[],
        metric_id=None,
        field_path="",
        legend="",
        text="",
        agg=CounterAgg.NONE,
        size=WidgetSize.FULL
    )

    # To get the number of columns in dataset
    monitoring.add_dashboard_panel(
        project, panel_type="Counter",
        title="Number of columns",
        tags=[],
        metric_id="DatasetDriftMetric",
        field_path="number_of_columns",
        legend="",
        text="",
        agg=CounterAgg.LAST,
        size=WidgetSize.HALF
    )

    # To get the number of drifted columns
    monitoring.add_dashboard_panel(
        project, panel_type="Counter",
        title="Number of drifted columns",
        tags=[],
        metric_id="DatasetDriftMetric",
        field_path="number_of_drifted_columns",
        legend="",
        text="",
        agg=CounterAgg.LAST,
        size=WidgetSize.HALF
    )

    # To get the target column drift score
    monitoring.add_dashboard_panel(
        project, panel_type="Counter",
        title="Target column drift score",
        tags=[],
        metric_id="ColumnDriftMetric",
        field_path="drift_score",
        legend="",
        text="",
        agg=CounterAgg.LAST,
        size=WidgetSize.HALF
    )

    # To get the number of missing columns
    monitoring.add_dashboard_panel(
        project, panel_type="Counter",
        title="Number of missing values - Current",
        tags=[],
        metric_id="DatasetMissingValuesMetric",
        field_path="current.number_of_missing_values",
        metric_args={},
        legend="Current - missing values",
        size=WidgetSize.HALF,
        agg=CounterAgg.LAST,
        text=""
    )

    # Plot the share of drifted columns
    monitoring.add_dashboard_panel(
        project, panel_type="Plot",
        title="Share of drifted columns",
        tags=[],
        metric_id="DatasetDriftMetric",
        field_path="share_of_drifted_columns",
        metric_args={},
        legend="share",
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
        agg=CounterAgg.SUM
    )

    # Plot R2 score - reference vs current
    monitoring.add_dashboard_panel(
        project, panel_type="MultiPlot",
        title="R2 score - Current vs Reference",
        tags=[],
        metric_id="RegressionQualityMetric",
        field_path="current.r2_score",
        metric_args={},
        legend="R2 Current",
        metric_id_2="RegressionQualityMetric",
        field_path_2="reference.r2_score",
        metric_args_2={},
        legend_2="Reference R2",
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
        agg=CounterAgg.SUM
    )

    # Plot MAE score - reference vs current
    monitoring.add_dashboard_panel(
        project, panel_type="MultiPlot",
        title="MAE score - Current vs Reference",
        tags=[],
        metric_id="RegressionQualityMetric",
        field_path="current.mean_abs_error",
        metric_args={},
        legend="MAE",
        metric_id_2="RegressionQualityMetric",
        field_path_2="reference.mean_abs_error",
        metric_args_2={},
        legend_2="Reference MAE",
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
        agg=CounterAgg.SUM
    )

    # Test suite panel
    monitoring.add_dashboard_panel(project, panel_type="TestSuite")

    project.show_dashboard()

    monitoring.delete_dashboard(project)
























