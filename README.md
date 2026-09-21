<h1>End-to-End MLOps project with Open Source tools</h1>

In this project, we will develop a machine learning workflow utilizing the MLOps pipeline. We will employ some of the open-source tools to construct the MLOps pipeline. This pipeline will encompass the full lifecycle of machine learning model development, which includes data preprocessing, model training, feature engineering, model monitoring, deployment, and implementing CI/CD pipelines.

When discussing the automation of the MLOps pipeline, it is important to highlight the role of Continuous Monitoring. Therefore, we will incorporate a feedback loop to complete the MLOps cycle through Continuous monitoring and re-training of the ML model.

In this project, we will utilize the following open source tools to establish the MLOps pipeline:

Feast
Mlflow
BentoML
Docker
Evidently
Apache Airflow
Project Overview:

The project involves:

1. **Data Preprocessing & Ingestion:** Preparing and processing data with Pandas
2. **Feature Store & Feature Engineering:** Moving the process data to feature store to store and organize them
3. **Experiment tracking & Model Registry:** Manage tracking, versioning, and training of a Scikit-learn model.
4. **Model Evaluation & serving:** Deploy the model through an API.
5. **Model Monitoring:** Assess data drift, concept drift, and model performance using reports and dashboards.
6. **Containerization:** Use Docker to containerize the model
7. **Continuous Integration:** Implement CI to initiate model and data validation with every modification to the source code. (Using DeepChecks, could use Evidently to stay in the same ecosystem)
8. **Model monitoring & retraining:** Consistently evaluate the model with new data and retrain as necessary.

## Project Workflow
                         ┌──────────────────┐
                         │      GitHub      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  GitHub Actions  │
                         │      CI / CD     │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             MODEL LIFECYCLE              CONTAINER DELIVERY
                    │                           │
              ┌─────┴─────┐               ┌───┴────┐
              │           │               │        │
            Feast       MLflow          BentoML   Docker
              │           │               │        │
              ▼           ▼               └───┬────┘
           Training    Registry                │
              │                                ▼
              ▼                               ECR
         Deepchecks                            │
                                               ▼
                                          Kubernetes
                                               │
                                        ┌──────┴──────┐
                                        │             │
                                       Pod          Service
                                        │             │
                                        └──────┬──────┘
                                               │
                                               ▼
                                          BentoML API


                    ┌──────────────────────────────────┐
                    │             AIRFLOW              │
                    │                                  │
                    │       Data / Feature flow        │
                    │                │                 │
                    │                ▼                 │
                    │        Evidently monitoring      │
                    │                │                 │
                    │          data changed?           │
                    │           /          \            │
                    │         NO            YES         │
                    │         │              │          │
                    │         ▼              ▼          │
                    │      No-op          Retrain       │
                    │         │              │          │
                    │         │              ▼          │
                    │         │          Validate       │
                    │         │              │          │
                    │         │              ▼          │
                    │         │         Deploy to      │
                    │         │        Kubernetes      │
                    │         │              │          │
                    │         └──────────────┴──────────┤
                    │                       │            │
                    │                       ▼            │
                    │                  DAG SUCCESS      │
                    └──────────────────────────────────┘

### Airflow DAG workflow
                  check_data_drift
                         │
                    ┌────┴────┐
                    │         │
                   NO        YES
                    │         │
                    ▼         ▼
              no_change     retrain
                    │         │
                    │         ▼
                    │       deploy
                    │         │
                    └────┬────┘
                         ▼
                      complete
## Run the project

### 1. Tag the Deployment Image and push to ECR 

``` Bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl set image deployment/house-service \
  house-service="$ECR_REGISTRY/house-price-model:$IMAGE_TAG"
```

## Kubernetes EKS Clusters

Configure observability

Control plane logging

If you see the individual log types, enable:

✅ API
✅ Audit
✅ Authenticator
✅ Controller manager
✅ Scheduler

These send EKS control-plane logs to Amazon CloudWatch, which is useful for a production-style MLOps project and troubleshooting.

Prometheus / Container Insights / enhanced observability:
If AWS presents optional paid monitoring features, leave them off for now. Your project already uses Evidently for ML/data monitoring, so we don't need to duplicate that with additional AWS monitoring costs.


