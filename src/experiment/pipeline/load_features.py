from feast import FeatureStore
from pathlib import Path


def load_features_from_feast(entity_df, feature_refs):
    # Chemin absolu vers feature_repo
    repo_path = (
        Path(__file__)
        .resolve()
        .parents[3]  # ← remonte jusqu’à la racine du projet
        / "2_data_processing"
        / "feature_store"
        / "feast_repo"
        / "feature_repo"
    )

    fs = FeatureStore(repo_path=str(repo_path))

    df = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_refs
    ).to_df()

    return df