import sys
import numpy as np
import pandas as pd
import torch
import joblib
import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gan.generator import GeneratorMLP
from src.gan.train_gan import get_continuous_features, FP_CATEGORICALS, OracleWrapper
from src.features.fp_pipeline import FEATURE_NAMES as FP_FEATURES
from src.features.net_pipeline import FEATURE_NAMES as NET_FEATURES

def augment_signal(signal: str, processed_dir: Path):
    print(f"--- Adversarial Augment for {signal} ---")
    parquet_path = processed_dir / f"{signal}_features.parquet"
    if not parquet_path.exists():
        return
        
    df = pd.read_parquet(parquet_path)
    bot_df = df[df["label"] == 0]
    human_df = df[df["label"] == 1]
    
    # Load Oracle
    pipeline_path = processed_dir / f"{signal}_pipeline.joblib"
    if not pipeline_path.exists():
        return
    oracle = OracleWrapper(joblib.load(pipeline_path))
    
    if signal == "net":
        # net_pipeline.joblib is a dict with scaler + IsolationForest wrapper
        pipeline_data = joblib.load(pipeline_path)
        net_scaler = pipeline_data["scaler"]
        net_wrapper = pipeline_data["wrapper"]
        # simple boundary attack for Network IsolationForest
        print("Using Gaussian noise boundary attack for net_model")
        X_bots = bot_df[NET_FEATURES].values
        # Generate 10x
        candidates = np.repeat(X_bots, 10, axis=0)
        noise = np.random.normal(0, np.std(X_bots, axis=0) * 0.1, size=candidates.shape)
        candidates += noise

        X_cands_scaled = net_scaler.transform(candidates)
        oracle_pred = net_wrapper.predict_proba(X_cands_scaled)[:, 1]
        keep = oracle_pred > 0.6
        accepted = candidates[keep]

        if len(accepted) > 0:
            aug_df = pd.DataFrame(accepted, columns=NET_FEATURES)
            aug_df["label"] = 0
            out_path = processed_dir / f"{signal}_adversarial.parquet"
            aug_df.to_parquet(out_path, index=False)
            print(f"Saved {len(aug_df)} adversarial samples for {signal}")
        return

    # GAN based signals
    g_path = processed_dir / f"{signal}_generator_v1.pth"
    scaler_path = processed_dir / f"{signal}_gan_scaler.joblib"
    
    if not g_path.exists() or not scaler_path.exists():
        print("Missing GAN generator or scaler.")
        return
        
    features_list = get_continuous_features(signal)
    scaler = joblib.load(scaler_path)
    
    feature_dim = len(features_list)
    netG = GeneratorMLP(feature_dim=feature_dim)
    netG.load_state_dict(torch.load(g_path))
    netG.eval()
    
    n_gen = len(bot_df) * 10
    if n_gen == 0:
        return
        
    # Generate
    noise = torch.randn(n_gen, 64)
    labels = torch.zeros(n_gen, 1)
    with torch.no_grad():
        fake_scaled = netG(noise, labels).numpy()
        
    fake_orig = scaler.inverse_transform(fake_scaled)
    
    full_fake = fake_orig
    final_cols = features_list
    
    if signal == "fp":
        # Handle categoricals
        final_cols = FP_FEATURES
        full_shape = [len(fake_orig), len(FP_FEATURES)]
        full_fake = np.zeros(full_shape)
        
        cont_indices = [FP_FEATURES.index(c) for c in features_list]
        cat_indices = [FP_FEATURES.index(c) for c in FP_CATEGORICALS]
        
        for idx in cont_indices:
            orig_idx = features_list.index(FP_FEATURES[idx])
            full_fake[:, idx] = fake_orig[:, orig_idx]
            
        # Sample categoricals empirically from human distribution
        for c_feat in FP_CATEGORICALS:
            idx = FP_FEATURES.index(c_feat)
            human_vals = human_df[c_feat].values
            sampled = np.random.choice(human_vals, size=n_gen)
            full_fake[:, idx] = sampled

    oracle_pred = oracle.predict_proba(full_fake)[:, 1]
    keep = oracle_pred > 0.6
    accepted = full_fake[keep]
    
    print(f"Generated {n_gen}, Oracle accepted {len(accepted)} (conf > 0.6)")
    
    if len(accepted) > 0:
        aug_df = pd.DataFrame(accepted, columns=final_cols)
        aug_df["label"] = 0
        out_path = processed_dir / f"{signal}_adversarial.parquet"
        aug_df.to_parquet(out_path, index=False)
        print(f"Saved {len(aug_df)} adversarial samples for {signal}")

if __name__ == "__main__":
    processed = PROJECT_ROOT / "data" / "processed"
    for sig in ["ks", "ms", "fp", "net", "wb"]:
        augment_signal(sig, processed)
