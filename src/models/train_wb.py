"""Train web bot sub-model (wb_model).

Model: XGBClassifier with isotonic calibration.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.wb_pipeline import run_pipeline as run_wb_pipeline, FEATURE_NAMES


def train(data_dir: Path, processed_dir: Path, mlflow_uri: str) -> None:
    """Train the web bot model."""
    mlflow_uri_formatted = f"file:///{Path(mlflow_uri).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri_formatted)
    mlflow.set_experiment("wb_detection")

    features_path = processed_dir / "wb_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        print(f"Loaded existing features: {len(features)} rows")
    else:
        features = run_wb_pipeline(data_dir, processed_dir)

    X = features[FEATURE_NAMES].values
    y = features["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="wb_model_v1"):
        params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
            "eval_metric": "logloss",
            "calibration_method": "isotonic", "calibration_cv": 5,
            "train_size": len(X_train), "test_size": len(X_test),
        }
        mlflow.log_params(params)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", use_label_encoder=False,
        )
        calibrated_model = CalibratedClassifierCV(
            base_model, method="isotonic", cv=5
        )
        calibrated_model.fit(X_train_scaled, y_train)

        y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        mlflow.log_metrics({
            "roc_auc": roc_auc, "f1": f1,
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1]),
        })

        cm_path = processed_dir / "wb_confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write(f"ROC-AUC: {roc_auc:.4f}\nF1: {f1:.4f}\n\n{cm}")
        mlflow.log_artifact(str(cm_path))

        pipeline = Pipeline([("scaler", scaler), ("model", calibrated_model)])
        mlflow.sklearn.log_model(pipeline, "wb_model")

        pipeline_path = processed_dir / "wb_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
        test_df["label"] = y_test
        test_path = processed_dir / "wb_test.parquet"
        test_df.to_parquet(test_path, index=False)
        print(f"Test set saved: {test_path} ({len(test_df)} rows)")
        print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train web bot model")
    parser.add_argument("--data_dir", type=str, default="data/raw/webbot")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mlflow_uri", type=str, default="mlflow/")
    args = parser.parse_args()

    train(
        PROJECT_ROOT / args.data_dir,
        PROJECT_ROOT / args.processed_dir,
        str(PROJECT_ROOT / args.mlflow_uri),
    )
