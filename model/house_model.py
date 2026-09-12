import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model

EXPERIMENT_NAME = "House price prediction"
EXPERIMENT_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"

)


class HouseModel:
    def __init__(self):
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.grid_search = None

    # Custom scorer for MSE
    def mse_scorer(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def train_model(self, features, target, test_size=0.25):
        params = {
            "fit_intercept": [True, False],
            "positive": [True,False]
        }
        model = LinearRegression()

        # Set up GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="neg_mean_squared_error",  # Negative MSE
            cv=5,   
            return_train_score=True
        )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, target, test_size=test_size)
        self.grid_search.fit(self.x_train, self.y_train)

        # Save the trained model
        with open("model/house_regression_model.pkl", "wb") as f:
            # Save the whole grid search wrapper so cv_results_ is preserved
            pickle.dump(self.grid_search, f)
        # ADD THIS: Save the data split so register_model.py can find it later
        with open("model/dataset_splits.pkl", "wb") as f:
            pickle.dump({
                "x_train": self.x_train,
                "x_test": self.x_test,
                "y_train": self.y_train,
                "y_test": self.y_test
            }, f)
            #pickle.dump(self.grid_search.best_estimator_, f)
        print("Model trained and saved as model.pkl")

    # Load model
    def load_model(self):
        # Check if the file exists before trying to open it
        if not os.path.exists("model/house_regression_model.pkl"):
            print("No saved model found on disk.")
            return None

        try:
            with open("model/house_regression_model.pkl", "rb") as f:
                loaded_obj = pickle.load(f)

            if isinstance(loaded_obj, GridSearchCV):
                return loaded_obj.best_estimator_
            return loaded_obj
        except (AttributeError, KeyError, ImportError) as e:
            # If scikit-learn version differences corrupt the pickle, catch it safely
            print(f"Failed to load model due to version mismatch or corruption: {e}")
            return None

    def predict(self, data):
        model = self.load_model()
        # Test data (sample input for prediction)
        #test_data = [5.1, 3.5, 1.4, 0.2]  # Example features
        prediction = model.predict(data)
        #print(f"Prediction for {test_data}: {int(prediction[0])}")
        return prediction

    def metrics(self, y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metric_dict = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        return metric_dict
    
    def configure_mlflow(self):
        mlflow.set_tracking_uri(uri= EXPERIMENT_URI)

        try:
            exp= mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if (exp is not None):
                mlflow.set_experiment(experiment_id=exp.experiment_id)
        except:
            exp_id = mlflow.create_experiment(name =EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_id=exp_id)
        finally:
            return mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    def log_gridsearch(self):
        for i, params in enumerate(self.grid_search.cv_results_["params"]):
            with mlflow.start_run(run_name="child_run_" + str(i), nested=True):  # Use nested=True for sub-runs
                # Get metrics from cv_results_
                mean_test_score = self.grid_search.cv_results_['mean_test_score'][i]
                std_test_score = self.grid_search.cv_results_['std_test_score'][i]
                
                # Log parameters and cross-validation metrics
                mlflow.log_params(params)
                # FIX: Explicitly pass step=i to differentiate database rows
                mlflow.log_metric("mean_cv_score", mean_test_score, step=0) #, step=i)
                mlflow.log_metric("std_cv_score", std_test_score, step=0) #, step=i)

                # Refit model on the best parameters and evaluate on test data
                model = self.grid_search.estimator.set_params(**params)
                model.fit(self.x_train, self.y_train)


                # Save metrics after the model has completed saving
                y_pred = model.predict(self.x_test)

                #New code
                custom_metrics = self.metrics(self.y_test, y_pred)
                for metric_name, metric_value in custom_metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=0)


                #mlflow.log_metrics(self.metrics(y_pred))
                mlflow.sklearn.log_model(model, "model", step=1)

                print(f"Logged run with params: {params},  {mean_test_score:.4f}, std_test_score: {std_test_score:.4f}")
 

    def register(self):  # FIX: Accept original features & target
        if self.grid_search is None:
            with open("model/house_regression_model.pkl", "rb") as f:
                saved_obj = pickle.load(f)

                if isinstance(saved_obj, GridSearchCV):
                    self.grid_search = saved_obj
                else:
                    raise ValueError("Pickle file only contains the best_estimator, cannot log gridsearch history!")

        # Re-split data to reconstruct train/test arrays matching the execution
        #x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size)
        # 2. Reload the saved data splits directly back into the class fields
        if self.x_test is None:
            with open("model/dataset_splits.pkl", "rb") as f:
                splits = pickle.load(f)
                self.x_train = splits["x_train"]
                self.x_test = splits["x_test"]
                self.y_train = splits["y_train"]
                self.y_test = splits["y_test"]
        with mlflow.start_run(run_name="LinearReg_GridSearch_Best", log_system_metrics=True) as run:
            # Log the best parameters and metrics
            best_params = self.grid_search.best_params_
            best_score = self.grid_search.best_score_
            mlflow.log_params(best_params)
            mlflow.log_metric("best_mean_cv_score", best_score, step=0)

            y_pred = self.predict(self.x_test)
            parent_metrics = self.metrics(self.y_test, y_pred)
            for k, v in parent_metrics.items():
                mlflow.log_metric(k, v, step=0)

            self.log_gridsearch()

            #Define signature
            signature = infer_signature(np.array(self.x_train), np.array(self.predict(self.x_test)))

            #Log and Register best model
            model_info = log_model(
                sk_model=self.grid_search.best_estimator_,
                artifact_path="house_model",
                signature=signature,
                input_example= self.x_train,
                registered_model_name="house_price_prediction",
                step=1
            )

            # FIX: Log your parent evaluation metrics AFTER log_model has finished execution.
            # This guarantees MLflow's internal database hook runs against an isolated slate.
            #y_pred = self.predict(self.x_test)
            #mlflow.log_metrics(self.metrics(y_pred))


        return model_info