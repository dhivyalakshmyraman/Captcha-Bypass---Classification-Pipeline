"""
Master pipeline runner with resume capability.
Skips already-completed stages based on output file presence.
"""
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
MLFLOW_URI = str(PROJECT_ROOT / "mlflow")


def exists(filename):
    return (PROCESSED / filename).exists()


def step(label):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# STEP 1: Train missing sub-models (fp, net, wb)
# ─────────────────────────────────────────────────────────────

if not exists("fp_pipeline.joblib") or not exists("fp_test.parquet"):
    step("Training FP (Browser Fingerprint) Model")
    from src.models.train_fp import train as train_fp
    train_fp(
        PROJECT_ROOT / "data/raw/fingerprint",
        PROCESSED,
        MLFLOW_URI,
    )
else:
    print("[SKIP] fp_pipeline.joblib already exists")

if not exists("net_pipeline.joblib") or not exists("net_test.parquet"):
    step("Training NET (Network Intrusion) Model")
    from src.models.train_net import train as train_net
    train_net(
        PROJECT_ROOT / "data/raw/network",
        PROCESSED,
        MLFLOW_URI,
    )
else:
    print("[SKIP] net_pipeline.joblib already exists")

if not exists("wb_pipeline.joblib") or not exists("wb_test.parquet"):
    step("Training WB (Web Bot) Model")
    from src.models.train_wb import train as train_wb
    train_wb(
        PROJECT_ROOT / "data/raw/webbot",
        PROCESSED,
        MLFLOW_URI,
    )
else:
    print("[SKIP] wb_pipeline.joblib already exists")

# ─────────────────────────────────────────────────────────────
# STEP 2: Train meta-learner (needs all 5 test sets)
# ─────────────────────────────────────────────────────────────

if not exists("meta_model.joblib") or not exists("meta_test.parquet"):
    step("Training Meta-Learner")
    from src.models.train_meta import train as train_meta
    train_meta(PROCESSED, MLFLOW_URI)
else:
    print("[SKIP] meta_model.joblib already exists")

# ─────────────────────────────────────────────────────────────
# STEP 3: GAN training for ks, ms, fp, wb
# ─────────────────────────────────────────────────────────────

from src.gan.train_gan import train_gan_for_signal

for sig in ["ks", "ms", "fp", "wb"]:
    gen_path = f"{sig}_generator_v1.pth"
    if not exists(gen_path):
        step(f"Training GAN for signal: {sig}")
        import mlflow
        mlflow.set_tracking_uri(f"file:///{Path(MLFLOW_URI).resolve().as_posix()}")
        train_gan_for_signal(sig, PROCESSED)
    else:
        print(f"[SKIP] GAN generator for {sig} already exists")

# ─────────────────────────────────────────────────────────────
# STEP 4: Adversarial augmentation
# ─────────────────────────────────────────────────────────────

from src.gan.adversarial_augment import augment_signal

for sig in ["ks", "ms", "fp", "net", "wb"]:
    adv_path = f"{sig}_adversarial.parquet"
    if not exists(adv_path):
        step(f"Adversarial augmentation for: {sig}")
        augment_signal(sig, PROCESSED)
    else:
        print(f"[SKIP] adversarial parquet for {sig} already exists")

# ─────────────────────────────────────────────────────────────
# STEP 5: Adversarial retrain (update sub-models with augmented data)
# ─────────────────────────────────────────────────────────────

step("Adversarial Retraining")
from src.gan.retrain_with_augment import retrain_signal

for sig in ["ks", "ms", "fp", "net", "wb"]:
    adv_path = PROCESSED / f"{sig}_adversarial.parquet"
    if adv_path.exists():
        retrain_signal(sig, PROCESSED)
    else:
        print(f"[SKIP] No adversarial data for {sig}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Retrain meta post-GAN
# ─────────────────────────────────────────────────────────────

step("Retraining Meta-Learner (post-GAN augmentation)")
from src.models.train_meta import train as train_meta_post
train_meta_post(PROCESSED, MLFLOW_URI)

# ─────────────────────────────────────────────────────────────
# STEP 7: Deepchecks validation reports
# ─────────────────────────────────────────────────────────────

step("Running Deepchecks Validation")
reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(exist_ok=True)

from src.validation.run_deepchecks import run_validation_for_signal

for sig in ["ks", "ms", "fp", "net", "wb"]:
    report_path = reports_dir / f"{sig}_data_validation.html"
    if not report_path.exists():
        run_validation_for_signal(sig, PROCESSED, reports_dir)
    else:
        print(f"[SKIP] Validation report for {sig} already exists")

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  ALL PIPELINE STAGES COMPLETE")
print("="*60)
print(f"\nProcessed files in: {PROCESSED}")
print(f"Reports in:         {reports_dir}")
print(f"MLflow tracking:    {MLFLOW_URI}")
