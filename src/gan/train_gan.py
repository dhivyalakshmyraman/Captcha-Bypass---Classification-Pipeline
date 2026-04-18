import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gan.generator import GeneratorMLP
from src.gan.discriminator import DiscriminatorMLP
from src.features.ks_pipeline import FEATURE_NAMES as KS_FEATURES
from src.features.ms_pipeline import FEATURE_NAMES as MS_FEATURES
from src.features.fp_pipeline import FEATURE_NAMES as FP_FEATURES
from src.features.wb_pipeline import FEATURE_NAMES as WB_FEATURES

FEATURES_MAP = {
    "ks": KS_FEATURES,
    "ms": MS_FEATURES,
    "fp": FP_FEATURES, # Note: fp needs categorical override later
    "wb": WB_FEATURES
}

FP_CATEGORICALS = ["canvas_hash", "webgl_renderer_hash", "ua_bot_score", "webdriver_flag", "touch_support"]

def get_continuous_features(signal: str):
    if signal == "fp":
        return [f for f in FP_FEATURES if f not in FP_CATEGORICALS]
    return FEATURES_MAP[signal]

class OracleWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def predict_proba(self, X):
        # some pipelines have scale issues with tensor to numpy
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.pipeline.predict_proba(X)

def train_gan_for_signal(signal: str, processed_dir: Path):
    print(f"--- Training GAN for {signal} ---")
    parquet_path = processed_dir / f"{signal}_features.parquet"
    if not parquet_path.exists():
        print(f"Skipping {signal}, raw data missing.")
        return

    features_list = get_continuous_features(signal)
    
    df = pd.read_parquet(parquet_path)
    # We condition on bot rows (label 0)
    bot_df = df[df["label"] == 0]
    
    if len(bot_df) < 10:
        print(f"Not enough bot rows for {signal}. Skipping.")
        return

    # Cap to 500 rows for speed
    MAX_BOT_ROWS = 500
    if len(bot_df) > MAX_BOT_ROWS:
        bot_df = bot_df.sample(n=MAX_BOT_ROWS, random_state=42)
        print(f"Subsampled bot rows to {MAX_BOT_ROWS} for GAN training.")
        
    X_bots = bot_df[features_list].values
    
    # MinMaxScaler to [-1, 1] for Generator Tanh
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_bots_scaled = scaler.fit_transform(X_bots)
    
    # Save the GAN specific scaler
    scaler_path = processed_dir / f"{signal}_gan_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    # DataLoader
    dataset = TensorDataset(torch.FloatTensor(X_bots_scaled), torch.zeros(len(X_bots_scaled), 1))
    dataloader = DataLoader(dataset, batch_size=min(64, len(X_bots_scaled)), shuffle=True)
    
    # Load Oracle (frozen sub-model)
    pipeline_path = processed_dir / f"{signal}_pipeline.joblib"
    if not pipeline_path.exists():
        print(f"Skipping {signal}, Sub-model not found.")
        return
    oracle_pipeline = joblib.load(pipeline_path)
    oracle = OracleWrapper(oracle_pipeline)

    # Init Networks
    feature_dim = len(features_list)
    netG = GeneratorMLP(feature_dim=feature_dim)
    netD = DiscriminatorMLP(feature_dim=feature_dim)
    
    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 300 # reduced for speed
    mlflow.set_experiment(f"{signal}_gan")
    
    with mlflow.start_run(run_name=f"{signal}_gan_training"):
        mlflow.log_params({"epochs": epochs, "feature_dim": feature_dim})
        
        for epoch in range(epochs):
            for i, (real_x, labels) in enumerate(dataloader):
                batch_size = real_x.size(0)
                
                # --- Train Discriminator ---
                netD.zero_grad()
                out_real = netD(real_x, labels).view(-1)
                errD_real = criterion(out_real, torch.ones(batch_size))
                
                noise = torch.randn(batch_size, 64)
                fake_x = netG(noise, labels)
                out_fake = netD(fake_x.detach(), labels).view(-1)
                errD_fake = criterion(out_fake, torch.zeros(batch_size))
                
                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()

                # --- Train Generator ---
                netG.zero_grad()
                out_fake_g = netD(fake_x, labels).view(-1)
                errG_adv = criterion(out_fake_g, torch.ones(batch_size))
                
                # Oracle Loss (needs original scale)
                fake_x_original = scaler.inverse_transform(fake_x.detach().numpy())
                
                if signal == "fp":
                    # For fingerprint, we need to inject categorical before oracle
                    # Use modes/defaults for the continuous GAN part evaluation
                    full_shape = list(fake_x_original.shape)
                    full_shape[1] = len(FP_FEATURES)
                    full_fake = np.zeros(full_shape)
                    
                    cont_indices = [FP_FEATURES.index(c) for c in features_list]
                    cat_indices = [FP_FEATURES.index(c) for c in FP_CATEGORICALS]
                    
                    for row_idx in range(len(fake_x_original)):
                        for idx, c_val in zip(cont_indices, fake_x_original[row_idx]):
                            full_fake[row_idx][idx] = c_val
                        # Defaults for categorical
                        for idx in cat_indices:
                            full_fake[row_idx][idx] = 0
                            
                    oracle_pred = oracle.predict_proba(full_fake)[:, 1]
                else:
                    oracle_pred = oracle.predict_proba(fake_x_original)[:, 1]
                
                oracle_loss = np.mean(oracle_pred) # we want oracle to predict human (1.0)
                
                # Total Generator loss
                errG = errG_adv + 0.5 * (1 - oracle_loss)
                errG.backward()
                optimizerG.step()
                
            if epoch % 100 == 0:
                print(f"[{epoch}/{epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} Oracle_Human_Conf: {oracle_loss:.4f}")
                mlflow.log_metrics({"loss_D": errD.item(), "loss_G": errG.item(), "oracle_conf": oracle_loss}, step=epoch)

        # Save Generator
        g_path = processed_dir / f"{signal}_generator_v1.pth"
        torch.save(netG.state_dict(), g_path)
        mlflow.log_artifact(str(g_path))
        print("Generator saved.")


if __name__ == "__main__":
    processed = PROJECT_ROOT / "data" / "processed"
    # Note: net is handled separately
    for sig in ["ks", "ms", "fp", "wb"]:
        train_gan_for_signal(sig, processed)
