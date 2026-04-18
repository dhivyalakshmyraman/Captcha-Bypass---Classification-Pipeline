"""Network intrusion feature extraction pipeline.

Extracts flow-based features from the CIC-IDS-2017 dataset.
Works both offline (called by training scripts) and online (imported by API).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional
import warnings

FEATURE_NAMES = [
    "flow_duration_log",
    "fwd_pkt_count", "bwd_pkt_count",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "iat_mean", "iat_std",
]

# Source columns in CIC-IDS-2017 (after stripping whitespace)
SOURCE_COLS = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_pkt_count",
    "Total Backward Packets": "bwd_pkt_count",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_pkts_per_sec",
    "Flow IAT Mean": "iat_mean",
    "Flow IAT Std": "iat_std",
    "Label": "label",
}

MAX_ROWS = 20000  # Subsample limit per user request


def run_pipeline(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run the full network feature extraction pipeline.

    Args:
        data_dir: Path to data/raw/network/ containing CIC-IDS CSVs.
        processed_dir: Path to data/processed/ for output.

    Returns:
        DataFrame with features and labels.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    dfs = []
    for csv_path in csv_files:
        print(f"Loading {csv_path.name}...")
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
        except Exception as e:
            warnings.warn(f"Error reading {csv_path.name}: {e}")
            continue

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    if not dfs:
        raise ValueError("No data loaded!")

    raw = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows: {len(raw)}")

    # Map label: BENIGN -> 1, everything else -> 0
    raw["Label"] = raw["Label"].str.strip()
    raw["label"] = (raw["Label"] == "BENIGN").astype(int)

    # Select and rename columns
    keep_cols = {}
    for src_col, dst_col in SOURCE_COLS.items():
        if src_col in raw.columns and src_col != "Label":
            keep_cols[src_col] = dst_col

    features = raw[list(keep_cols.keys())].rename(columns=keep_cols)
    features["label"] = raw["label"]

    # Convert numeric
    for col in FEATURE_NAMES:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Compute flow_duration_log
    if "flow_duration" in features.columns:
        features["flow_duration_log"] = np.log1p(features["flow_duration"].fillna(0))
        features = features.drop(columns=["flow_duration"])

    # Drop rows with inf or NaN in flow_bytes_per_sec
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=["flow_bytes_per_sec"])

    # Clip outliers at 99th percentile for flow_bytes_per_sec
    p99 = features["flow_bytes_per_sec"].quantile(0.99)
    features["flow_bytes_per_sec"] = features["flow_bytes_per_sec"].clip(upper=p99)

    # Fill remaining NaN
    features = features.fillna(0.0)

    # Stratified subsample to MAX_ROWS
    if len(features) > MAX_ROWS:
        print(f"Subsampling from {len(features)} to {MAX_ROWS} rows (stratified)...")
        from sklearn.model_selection import train_test_split
        features, _ = train_test_split(
            features, train_size=MAX_ROWS, stratify=features["label"],
            random_state=42,
        )
        features = features.reset_index(drop=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "net_features.parquet"
    features.to_parquet(out_path, index=False)

    label_counts = features["label"].value_counts()
    print(f"Network features saved: {out_path} ({len(features)} rows)")
    print(f"  BENIGN (1): {label_counts.get(1, 0)}, Attack (0): {label_counts.get(0, 0)}")

    return features


def transform_online(raw_data: dict, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Transform raw network flow data for online inference.

    Args:
        raw_data: Dict with network flow fields from API request.
        scaler: Fitted StandardScaler (loaded from MLflow).

    Returns:
        Feature vector as numpy array.
    """
    features = np.array([[
        np.log1p(raw_data.get("flow_duration", 0)),
        raw_data.get("fwd_pkt_count", 0),
        raw_data.get("bwd_pkt_count", 0),
        raw_data.get("flow_bytes_per_sec", 0),
        raw_data.get("flow_pkts_per_sec", 0),
        raw_data.get("iat_mean", 0),
        raw_data.get("iat_std", 0),
    ]])

    if scaler is not None:
        features = scaler.transform(features)

    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Network intrusion feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/raw/network")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(
        project_root / args.data_dir,
        project_root / args.processed_dir,
    )
