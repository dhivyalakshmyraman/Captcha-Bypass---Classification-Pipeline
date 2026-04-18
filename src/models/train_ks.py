"""Train keystroke sub-model (ks_model).

Model: RandomForestClassifier with isotonic calibration.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ks_pipeline import run_pipeline as run_ks_pipeline, FEATURE_NAMES


def train(data_dir: Path, processed_dir: Path, mlflow_uri: str) -> None:
    """Train the keystroke model.

    Args:
        data_dir: Path to data/raw/keystroke/.
        processed_dir: Path to data/processed/.
        mlflow_uri: MLflow tracking URI.
    """
    mlflow_uri_formatted = f"file:///{Path(mlflow_uri).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri_formatted)
    mlflow.set_experiment("ks_detection")

    # Run feature pipeline if needed
    features_path = processed_dir / "ks_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        print(f"Loaded existing features: {len(features)} rows")
    else:
        features = run_ks_pipeline(data_dir, processed_dir)

    X = features[FEATURE_NAMES].values
    y = features["label"].values

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="ks_model_v1"):
        # Log params
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "random_state": 42,
            "calibration_method": "isotonic",
            "calibration_cv": 5,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        mlflow.log_params(params)

        # Fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train base model
        base_model = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42
        )

        # Calibrate
        calibrated_model = CalibratedClassifierCV(
            base_model, method="isotonic", cv=5
        )
        calibrated_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "f1": f1,
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        })

        # Save confusion matrix as artifact
        cm_path = processed_dir / "ks_confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write(f"ROC-AUC: {roc_auc:.4f}\nF1: {f1:.4f}\n\n{cm}")
        mlflow.log_artifact(str(cm_path))

        # Create pipeline (scaler + model)
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", calibrated_model),
        ])

        # Log model
        mlflow.sklearn.log_model(pipeline, "ks_model")

        # Save pipeline locally too
        pipeline_path = processed_dir / "ks_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Save test set for meta-learner
        test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
        test_df["label"] = y_test
        test_path = processed_dir / "ks_test.parquet"
        test_df.to_parquet(test_path, index=False)
        print(f"Test set saved: {test_path} ({len(test_df)} rows)")

        print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train keystroke model")
    parser.add_argument("--data_dir", type=str, default="data/raw/keystroke")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mlflow_uri", type=str, default="mlflow/")
    args = parser.parse_args()

    train(
        PROJECT_ROOT / args.data_dir,
        PROJECT_ROOT / args.processed_dir,
        str(PROJECT_ROOT / args.mlflow_uri),
    )
