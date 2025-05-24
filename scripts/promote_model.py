# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "mileshsoni"
    repo_name = "mlops-mini-project"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"
    
    # get the latest version in production and move it to archived
    try:
        latest_mv = client.get_model_version_by_alias(name="my_model", alias="production")
        client.set_registered_model_alias(model_name, 'archived', latest_mv.version)
    except:
        pass
    
    # Get the latest version in staging
    latest_version_staging = client.get_model_version_by_alias(name="my_model", alias="Staging").version

    client.set_registered_model_alias(model_name, 'production', latest_version_staging)
    
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()