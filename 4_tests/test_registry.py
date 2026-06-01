import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "unrate_forecast_model"
MODEL_ALIAS = "champion"


def test_registry_full_check():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    rm = client.get_registered_model(MODEL_NAME)
    print("Registered model:", rm.name)

    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    for mv in versions:
        print(
            "version=", mv.version,
            "| run_id=", mv.run_id,
            "| source=", mv.source,
            "| status=", getattr(mv, "status", None),
        )

    mv_alias = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    print("Alias champion -> version:", mv_alias.version)
    print("Alias champion -> run_id:", mv_alias.run_id)

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Loaded model from:", model_uri)

    assert rm.name == MODEL_NAME
    assert mv_alias is not None
    assert model is not None

    print("\n✅ MLflow Registry OK")


if __name__ == "__main__":
    test_registry_full_check()