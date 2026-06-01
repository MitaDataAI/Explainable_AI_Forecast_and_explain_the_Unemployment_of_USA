import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "unrate_forecast_model"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = mv.run_id

print(f"Model version: {mv.version}")
print(f"Run ID: {run_id}")
print("\nArtifacts du run :")

def list_artifacts_recursive(client, run_id, path=""):
    items = client.list_artifacts(run_id, path)
    for item in items:
        print(item.path)
        if item.is_dir:
            list_artifacts_recursive(client, run_id, item.path)

list_artifacts_recursive(client, run_id)