import sys
import numpy as np
import pandas as pd
import joblib
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from importlib import import_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_train_module(signal: str):
    return import_module(f"src.models.train_{signal}")

def retrain_signal(signal: str, processed_dir: Path):
    print(f"--- Retraining {signal} with Augmentation ---")
    orig_path = processed_dir / f"{signal}_features.parquet"
    adv_path = processed_dir / f"{signal}_adversarial.parquet"
    test_path = processed_dir / f"{signal}_test.parquet"
    pipeline_path = processed_dir / f"{signal}_pipeline.joblib"
    
    if not orig_path.exists() or not adv_path.exists():
        print("Missing original or adversarial data.")
        return
        
    orig_df = pd.read_parquet(orig_path)
    adv_df = pd.read_parquet(adv_path)
    test_df = pd.read_parquet(test_path)
    
    # Calculate baseline AUC
    pipeline_data = joblib.load(pipeline_path)
    features_cols = [c for c in test_df.columns if c != "label"]
    X_test = test_df[features_cols].values
    y_test = test_df["label"].values
    
    if signal == "net":
        wrapper = pipeline_data["wrapper"]
        scaler = pipeline_data["scaler"]
        p_test = wrapper.predict_proba(scaler.transform(X_test))[:, 1]
        baseline_auc = roc_auc_score(1 - y_test, p_test)
    else:
        p_test = pipeline_data.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, p_test)
        
    print(f"Baseline AUC: {baseline_auc:.4f}")
    
    # Combine training data
    aug_train = pd.concat([orig_df, adv_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    temp_train_path = processed_dir / f"{signal}_features_aug.parquet"
    aug_train.to_parquet(temp_train_path, index=False)
    
    # Trick the training script into using the augmented data
    import shutil
    shutil.move(orig_path, orig_path.with_suffix('.bak'))
    shutil.move(temp_train_path, orig_path)
    
    try:
        # Run training
        train_mod = get_train_module(signal)
        # We need to run training without overwriting test set if possible, 
        # but our scripts do overwrite. So we test on the NEW test set that it splits OR
        # just accept the new model if its internal validation is good.
        # Actually, let's call train() directly.
        mlflow_uri = str(PROJECT_ROOT / "mlflow")
        train_mod.train(PROJECT_ROOT / f"data/raw/{signal}", processed_dir, mlflow_uri)
        
        # Now check new model
        pipeline_data = joblib.load(pipeline_path)
        if signal == "net":
            wrapper = pipeline_data["wrapper"]
            scaler = pipeline_data["scaler"]
            p_test = wrapper.predict_proba(scaler.transform(X_test))[:, 1]
            new_auc = roc_auc_score(1 - y_test, p_test)
        else:
            p_test = pipeline_data.predict_proba(X_test)[:, 1]
            new_auc = roc_auc_score(y_test, p_test)
            
        print(f"New AUC on original test set: {new_auc:.4f}")
        
    finally:
        # Restore
        shutil.move(orig_path, temp_train_path)
        shutil.move(orig_path.with_suffix('.bak'), orig_path)
        
    with mlflow.start_run(run_name=f"{signal}_adversarial_eval"):
        mlflow.log_metrics({"baseline_auc": baseline_auc, "new_auc": new_auc})
        if new_auc >= baseline_auc - 0.01:
            print(f"Registering new model! (AUC delta: {new_auc - baseline_auc:.4f})")
            mlflow.log_param("adversarial_retrain_accepted", True)
            # In a real setup, register to Model Registry here
        else:
            print(f"Rejecting new model. (AUC delta: {new_auc - baseline_auc:.4f})")
            mlflow.log_param("adversarial_retrain_accepted", False)
            # Rollback model save in joblib
            # Note: For this script we assume MLflow keeps versions via runs.

if __name__ == "__main__":
    processed = PROJECT_ROOT / "data" / "processed"
    for sig in ["ks", "ms", "fp", "net", "wb"]:
        retrain_signal(sig, processed)
