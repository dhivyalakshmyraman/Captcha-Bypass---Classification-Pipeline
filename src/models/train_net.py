"""Train network intrusion sub-model (net_model).

Model: IsolationForest (unsupervised, trained on BENIGN only).
Scores converted to [0,1] via MinMaxScaler for probability interpretation.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.net_pipeline import run_pipeline as run_net_pipeline, FEATURE_NAMES


class IsolationForestWrapper:
    """Wrapper to give IsolationForest a predict_proba interface.

    Converts decision_function scores to [0,1] probabilities.
    Higher score = more likely to be bot (anomalous).
    """

    def __init__(self, iso_forest: IsolationForest, score_scaler: MinMaxScaler):
        self.iso_forest = iso_forest
        self.score_scaler = score_scaler

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability scores.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [p_human, p_bot].
        """
        raw_scores = self.iso_forest.decision_function(X)
        # Negate: IsolationForest gives lower scores for anomalies
        # We want higher = more bot-like
        negated = -raw_scores
        # Scale to [0, 1]
        scaled = self.score_scaler.transform(negated.reshape(-1, 1)).flatten()
        scaled = np.clip(scaled, 0, 1)
        return np.column_stack([1 - scaled, scaled])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def train(data_dir: Path, processed_dir: Path, mlflow_uri: str) -> None:
    """Train the network intrusion model."""
    mlflow_uri_formatted = f"file:///{Path(mlflow_uri).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri_formatted)
    mlflow.set_experiment("net_detection")

    features_path = processed_dir / "net_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        print(f"Loaded existing features: {len(features)} rows")
    else:
        features = run_net_pipeline(data_dir, processed_dir)

    X = features[FEATURE_NAMES].values
    y = features["label"].values

    # Split all data 80/20 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="net_model_v1"):
        params = {
            "n_estimators": 200, "contamination": 0.3, "random_state": 42,
            "train_size": len(X_train), "test_size": len(X_test),
            "model_type": "IsolationForest",
        }
        mlflow.log_params(params)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train IsolationForest on BENIGN rows only
        benign_mask = y_train == 1
        X_train_benign = X_train_scaled[benign_mask]
        print(f"Training IsolationForest on {len(X_train_benign)} BENIGN rows")

        iso_forest = IsolationForest(
            n_estimators=200, contamination=0.3, random_state=42
        )
        iso_forest.fit(X_train_benign)

        # Get raw decision function scores on full test set
        raw_scores = -iso_forest.decision_function(X_test_scaled)

        # Fit MinMaxScaler on test scores for probability mapping
        score_scaler = MinMaxScaler()
        score_scaler.fit(raw_scores.reshape(-1, 1))

        # Create wrapper
        model_wrapper = IsolationForestWrapper(iso_forest, score_scaler)

        # Evaluate
        y_pred_proba = model_wrapper.predict_proba(X_test_scaled)[:, 1]
        # For ROC-AUC: label 0 = bot/attack is the positive class
        # But our y has 0=attack, 1=benign. We need p(bot) vs y=0 as positive.
        # Flip: use 1-y as true labels, so 1=attack and p_bot as scores
        roc_auc = roc_auc_score(1 - y_test, y_pred_proba)

        print(f"ROC-AUC (attack detection): {roc_auc:.4f}")

        mlflow.log_metrics({"roc_auc": roc_auc})

        # Save full pipeline
        pipeline_data = {
            "scaler": scaler,
            "iso_forest": iso_forest,
            "score_scaler": score_scaler,
            "wrapper": model_wrapper,
        }

        pipeline_path = processed_dir / "net_pipeline.joblib"
        joblib.dump(pipeline_data, pipeline_path)

        mlflow.log_artifact(str(pipeline_path))

        # Also log the wrapper as a sklearn model
        # We'll log the individual components
        mlflow.sklearn.log_model(iso_forest, "net_model_isoforest")

        # Save test set for meta-learner
        test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
        test_df["label"] = y_test
        test_path = processed_dir / "net_test.parquet"
        test_df.to_parquet(test_path, index=False)
        print(f"Test set saved: {test_path} ({len(test_df)} rows)")
        print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network intrusion model")
    parser.add_argument("--data_dir", type=str, default="data/raw/network")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mlflow_uri", type=str, default="mlflow/")
    args = parser.parse_args()

    train(
        PROJECT_ROOT / args.data_dir,
        PROJECT_ROOT / args.processed_dir,
        str(PROJECT_ROOT / args.mlflow_uri),
    )
