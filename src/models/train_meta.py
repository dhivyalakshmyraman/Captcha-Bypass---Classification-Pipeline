"""Train meta-learner model.

Fuses calibrated probability scores from all 5 sub-models using LogisticRegression.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, log_loss, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SIGNAL_NAMES = ["ks", "ms", "fp", "net", "wb"]
META_FEATURES = ["p_ks", "p_ms", "p_fp", "p_net", "p_wb"]


def load_sub_model_and_score(signal: str, processed_dir: Path) -> tuple:
    """Load a sub-model's test set and compute probability scores.

    Args:
        signal: Signal name (ks, ms, fp, net, wb).
        processed_dir: Path to data/processed/.

    Returns:
        Tuple of (probabilities, labels) or (None, None) if unavailable.
    """
    test_path = processed_dir / f"{signal}_test.parquet"
    pipeline_path = processed_dir / f"{signal}_pipeline.joblib"

    if not test_path.exists():
        print(f"Warning: {test_path} not found, will impute with 0.5")
        return None, None

    test_df = pd.read_parquet(test_path)
    labels = test_df["label"].values
    features = test_df.drop(columns=["label"]).values

    if not pipeline_path.exists():
        print(f"Warning: {pipeline_path} not found, will impute with 0.5")
        return None, None

    if signal == "net":
        # Network model uses IsolationForest wrapper
        pipeline_data = joblib.load(pipeline_path)
        scaler = pipeline_data["scaler"]
        wrapper = pipeline_data["wrapper"]
        X_scaled = scaler.transform(features)
        proba = wrapper.predict_proba(X_scaled)[:, 1]
    else:
        pipeline = joblib.load(pipeline_path)
        proba = pipeline.predict_proba(features)[:, 1]

    return proba, labels


def build_meta_dataset(processed_dir: Path) -> tuple:
    """Build the meta-learner training matrix.

    For each label class (0 and 1) separately, randomly pairs one probability
    score from each signal's test set. Missing signals imputed with 0.5.

    Args:
        processed_dir: Path to data/processed/.

    Returns:
        Tuple of (X_meta, y_meta) numpy arrays.
    """
    rng = np.random.default_rng(42)

    # Collect probabilities and labels per signal
    signal_data = {}
    for signal in SIGNAL_NAMES:
        proba, labels = load_sub_model_and_score(signal, processed_dir)
        signal_data[signal] = (proba, labels)

    # Separate by label class
    class_probas = {0: {}, 1: {}}
    for signal in SIGNAL_NAMES:
        proba, labels = signal_data[signal]
        if proba is not None and labels is not None:
            for cls in [0, 1]:
                mask = labels == cls
                if mask.any():
                    class_probas[cls][signal] = proba[mask]

    # Determine number of rows per class
    # Use the minimum count across signals for each class
    meta_rows = []
    meta_labels = []

    for cls in [0, 1]:
        counts = []
        for signal in SIGNAL_NAMES:
            if signal in class_probas[cls]:
                counts.append(len(class_probas[cls][signal]))

        if not counts:
            continue

        n_pairs = min(counts) if counts else 0
        if n_pairs == 0:
            continue

        for i in range(n_pairs):
            row = []
            for signal in SIGNAL_NAMES:
                if signal in class_probas[cls]:
                    # Random pairing with seed
                    idx = rng.integers(0, len(class_probas[cls][signal]))
                    row.append(class_probas[cls][signal][idx])
                else:
                    row.append(0.5)  # impute missing signal
            meta_rows.append(row)
            meta_labels.append(cls)

    X_meta = np.array(meta_rows)
    y_meta = np.array(meta_labels)

    return X_meta, y_meta


def train(processed_dir: Path, mlflow_uri: str) -> None:
    """Train the meta-learner model."""
    mlflow_uri_formatted = f"file:///{Path(mlflow_uri).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_uri_formatted)
    mlflow.set_experiment("meta_detection")

    X_meta, y_meta = build_meta_dataset(processed_dir)

    print(f"Meta dataset: {X_meta.shape[0]} rows, {X_meta.shape[1]} features")
    print(f"Label distribution: {dict(zip(*np.unique(y_meta, return_counts=True)))}")

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y_meta, test_size=0.2, stratify=y_meta, random_state=42
    )

    with mlflow.start_run(run_name="meta_model_v1"):
        params = {
            "C": 1.0, "max_iter": 1000, "random_state": 42,
            "train_size": len(X_train), "test_size": len(X_test),
            "n_signals": len(SIGNAL_NAMES),
        }
        mlflow.log_params(params)

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        mlflow.log_metrics({
            "roc_auc": roc_auc, "f1": f1, "log_loss": logloss,
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1]),
        })

        # Log model coefficients
        coefs = dict(zip(META_FEATURES, model.coef_[0]))
        print(f"Model coefficients: {coefs}")
        mlflow.log_params({f"coef_{k}": f"{v:.4f}" for k, v in coefs.items()})

        mlflow.sklearn.log_model(model, "meta_model")

        model_path = processed_dir / "meta_model.joblib"
        joblib.dump(model, model_path)

        # Save meta test set
        test_df = pd.DataFrame(X_test, columns=META_FEATURES)
        test_df["label"] = y_test
        test_path = processed_dir / "meta_test.parquet"
        test_df.to_parquet(test_path, index=False)

        print(f"Meta model saved: {model_path}")
        print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train meta-learner")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--mlflow_uri", type=str, default="mlflow/")
    args = parser.parse_args()

    train(
        PROJECT_ROOT / args.processed_dir,
        str(PROJECT_ROOT / args.mlflow_uri),
    )
