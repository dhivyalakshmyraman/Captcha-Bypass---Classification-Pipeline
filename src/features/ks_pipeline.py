"""Keystroke feature extraction pipeline.

Extracts timing-based features from the DSL-StrongPasswordData dataset.
Works both offline (called by training scripts) and online (imported by API).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib


# Column groups in the DSL dataset
H_COLS = [
    "H.period", "H.t", "H.i", "H.e", "H.five",
    "H.Shift.r", "H.o", "H.a", "H.n", "H.l", "H.Return",
]
DD_COLS = [
    "DD.period.t", "DD.t.i", "DD.i.e", "DD.e.five",
    "DD.five.Shift.r", "DD.Shift.r.o", "DD.o.a", "DD.a.n", "DD.n.l", "DD.l.Return",
]
UD_COLS = [
    "UD.period.t", "UD.t.i", "UD.i.e", "UD.e.five",
    "UD.five.Shift.r", "UD.Shift.r.o", "UD.o.a", "UD.a.n", "UD.n.l", "UD.l.Return",
]

FEATURE_NAMES = [
    "hold_time_mean", "hold_time_std",
    "flight_time_mean", "flight_time_std",
    "digraph_mean", "digraph_std",
    "error_rate", "session_duration",
]


def extract_session_features(group: pd.DataFrame) -> pd.Series:
    """Extract features from a single session (group of repetitions).

    Args:
        group: DataFrame rows for one subject+sessionIndex combination.

    Returns:
        Series with extracted feature values.
    """
    h_vals = group[H_COLS].values.flatten()
    ud_vals = group[UD_COLS].values.flatten()
    dd_vals = group[DD_COLS].values.flatten()

    return pd.Series({
        "hold_time_mean": np.mean(h_vals),
        "hold_time_std": np.std(h_vals),
        "flight_time_mean": np.mean(ud_vals),
        "flight_time_std": np.std(ud_vals),
        "digraph_mean": np.mean(dd_vals),
        "digraph_std": np.std(dd_vals),
        "error_rate": 0.0,  # DSL dataset has no backspace events
        "session_duration": np.sum(dd_vals),
    })


def generate_synthetic_bots(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic bot keystroke sessions.

    Bots type with uniform, fast timing (5-30ms per key event).

    Args:
        n_rows: Number of synthetic bot rows to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with same feature columns as human data, label=0.
    """
    n_keys_h = len(H_COLS)
    n_keys_ud = len(UD_COLS)
    n_keys_dd = len(DD_COLS)
    reps_per_session = 50  # approximate

    records = []
    for _ in range(n_rows):
        # Simulate bot timing: uniform 5-30ms
        h_vals = rng.uniform(0.005, 0.030, size=n_keys_h * reps_per_session)
        ud_vals = rng.uniform(0.005, 0.030, size=n_keys_ud * reps_per_session)
        dd_vals = rng.uniform(0.005, 0.030, size=n_keys_dd * reps_per_session)

        records.append({
            "hold_time_mean": np.mean(h_vals),
            "hold_time_std": np.std(h_vals),
            "flight_time_mean": np.mean(ud_vals),
            "flight_time_std": np.std(ud_vals),
            "digraph_mean": np.mean(dd_vals),
            "digraph_std": np.std(dd_vals),
            "error_rate": 0.0,
            "session_duration": np.sum(dd_vals),
        })

    df = pd.DataFrame(records)
    df["label"] = 0
    return df


def run_pipeline(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run the full keystroke feature extraction pipeline.

    Args:
        data_dir: Path to data/raw/keystroke/ containing the CSV.
        processed_dir: Path to data/processed/ for output.

    Returns:
        DataFrame with features and labels.
    """
    csv_path = data_dir / "DSL-StrongPasswordData.csv"
    df = pd.read_csv(csv_path)

    # Extract features per session (subject + sessionIndex)
    human_features = df.groupby(["subject", "sessionIndex"]).apply(
        extract_session_features
    ).reset_index(drop=True)
    human_features["label"] = 1

    # Generate synthetic bots (1× human rows)
    rng = np.random.default_rng(42)
    bot_features = generate_synthetic_bots(len(human_features), rng)

    # Combine
    features = pd.concat([human_features, bot_features], ignore_index=True)

    # Save
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "ks_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Keystroke features saved: {out_path} ({len(features)} rows)")

    return features


def transform_online(raw_data: dict, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Transform raw keystroke data for online inference.

    Args:
        raw_data: Dict with timing fields from API request.
        scaler: Fitted StandardScaler (loaded from MLflow).

    Returns:
        Feature vector as numpy array.
    """
    # Extract from raw timing data
    h_vals = [raw_data.get(c, 0.0) for c in H_COLS if c in raw_data]
    ud_vals = [raw_data.get(c, 0.0) for c in UD_COLS if c in raw_data]
    dd_vals = [raw_data.get(c, 0.0) for c in DD_COLS if c in raw_data]

    if not h_vals:
        h_vals = list(raw_data.get("hold_times", [0.1]))
    if not ud_vals:
        ud_vals = list(raw_data.get("flight_times", [0.1]))
    if not dd_vals:
        dd_vals = list(raw_data.get("digraph_times", [0.2]))

    features = np.array([[
        np.mean(h_vals), np.std(h_vals),
        np.mean(ud_vals), np.std(ud_vals),
        np.mean(dd_vals), np.std(dd_vals),
        0.0,  # error_rate
        np.sum(dd_vals),  # session_duration
    ]])

    if scaler is not None:
        features = scaler.transform(features)

    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Keystroke feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/raw/keystroke")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(
        project_root / args.data_dir,
        project_root / args.processed_dir,
    )
