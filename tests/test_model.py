# load test
import unittest
import mlflow
import os

class TestModelLoading(unittest.TestCase):

    # method to connect to dagshub
    @classmethod
    def setUpClass(cls):
        
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

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)


    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        try:
            model_version = client.get_model_version_by_alias(name="MyModel", alias="staging")
            return model_version.version
        except Exception as e:
            return None
    # check if we recieved model from model registry
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

if __name__ == "__main__":
    unittest.main()