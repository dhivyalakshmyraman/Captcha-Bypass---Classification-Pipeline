"""Mouse dynamics feature extraction pipeline.

Extracts movement-based features from the Balabit Mouse Dynamics Challenge dataset.
Works both offline (called by training scripts) and online (imported by API).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional, List
import warnings

FEATURE_NAMES = [
    "velocity_mean", "velocity_std",
    "acceleration_mean",
    "curvature_mean",
    "idle_ratio",
    "click_count",
    "direction_changes",
]

WINDOW_SIZE_SEC = 30.0  # sliding window size in seconds


def load_session_file(filepath: Path) -> pd.DataFrame:
    """Load a single mouse session file.

    Args:
        filepath: Path to the session CSV/text file.

    Returns:
        DataFrame with columns: record_timestamp, client_timestamp, button, state, x, y
    """
    df = pd.read_csv(filepath, header=0)
    df.columns = [c.strip() for c in df.columns]

    # Standardize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if "record" in cl and "time" in cl:
            col_map[c] = "record_timestamp"
        elif "client" in cl and "time" in cl:
            col_map[c] = "client_timestamp"
        elif cl == "button":
            col_map[c] = "button"
        elif cl == "state":
            col_map[c] = "state"
        elif cl == "x":
            col_map[c] = "x"
        elif cl == "y":
            col_map[c] = "y"

    df = df.rename(columns=col_map)

    # Convert numeric columns
    for col in ["record_timestamp", "client_timestamp", "x", "y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["record_timestamp", "x", "y"])
    return df


def extract_window_features(window: pd.DataFrame) -> dict:
    """Extract features from a time window of mouse data.

    Args:
        window: DataFrame slice for a single time window.

    Returns:
        Dict of feature values.
    """
    if len(window) < 3:
        return None

    t = window["record_timestamp"].values
    x = window["x"].values.astype(float)
    y = window["y"].values.astype(float)

    # Time deltas
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-6, dt)

    # Distances
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    # Velocity
    velocity = dist / dt
    velocity_mean = np.mean(velocity) if len(velocity) > 0 else 0.0
    velocity_std = np.std(velocity) if len(velocity) > 0 else 0.0

    # Acceleration
    if len(velocity) > 1:
        dv = np.diff(velocity)
        dt2 = dt[1:]
        dt2 = np.where(dt2 <= 0, 1e-6, dt2)
        acceleration = dv / dt2
        acceleration_mean = np.mean(np.abs(acceleration))
    else:
        acceleration_mean = 0.0

    # Curvature (angular change per unit distance)
    if len(dx) > 1:
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        # Wrap to [0, pi]
        angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
        cumulative_dist = np.cumsum(dist[:-1])
        cumulative_dist = np.where(cumulative_dist <= 0, 1e-6, cumulative_dist)
        curvature = angle_changes / np.maximum(dist[1:], 1e-6)
        curvature_mean = np.nanmean(curvature)
    else:
        curvature_mean = 0.0

    # Idle ratio
    total_time = t[-1] - t[0] if t[-1] != t[0] else 1.0
    zero_movement = np.sum(dist < 1.0)  # less than 1 pixel
    idle_ratio = zero_movement / max(len(dist), 1)

    # Click count
    states = window["state"].values
    click_count = 0
    for i in range(1, len(states)):
        s_prev = str(states[i - 1]).lower()
        s_curr = str(states[i]).lower()
        if "pressed" in s_curr or ("drag" in s_curr and "drag" not in s_prev):
            click_count += 1

    # Direction changes (angle reversals > 45 degrees)
    if len(dx) > 1:
        angles = np.arctan2(dy, dx)
        angle_diffs = np.abs(np.diff(angles))
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        direction_changes = int(np.sum(angle_diffs > np.pi / 4))
    else:
        direction_changes = 0

    return {
        "velocity_mean": velocity_mean,
        "velocity_std": velocity_std,
        "acceleration_mean": acceleration_mean,
        "curvature_mean": float(np.nan_to_num(curvature_mean, nan=0.0)),
        "idle_ratio": idle_ratio,
        "click_count": click_count,
        "direction_changes": direction_changes,
    }


def extract_session_windows(df: pd.DataFrame, window_size: float = WINDOW_SIZE_SEC) -> List[dict]:
    """Extract features from sliding windows across a session.

    Args:
        df: Full session DataFrame.
        window_size: Window size in seconds.

    Returns:
        List of feature dicts, one per window.
    """
    if df.empty:
        return []

    t = df["record_timestamp"].values
    t_start = t[0]
    t_end = t[-1]

    if t_end - t_start < 1.0:
        return []

    windows = []
    step = window_size / 2  # 50% overlap
    current = t_start

    while current + window_size <= t_end + step:
        mask = (t >= current) & (t < current + window_size)
        window_df = df[mask]
        if len(window_df) >= 3:
            feats = extract_window_features(window_df)
            if feats is not None:
                windows.append(feats)
        current += step

    # If no windows produced, try the whole session as one window
    if not windows and len(df) >= 3:
        feats = extract_window_features(df)
        if feats is not None:
            windows.append(feats)

    return windows


def run_pipeline(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run the full mouse dynamics feature extraction pipeline.

    Args:
        data_dir: Path to data/raw/mouse/ containing training/test files.
        processed_dir: Path to data/processed/ for output.

    Returns:
        DataFrame with features and labels.
    """
    all_records = []

    # 1. Training files — all legal sessions (label=1)
    training_dir = data_dir / "training_files"
    if training_dir.exists():
        for user_dir in sorted(training_dir.iterdir()):
            if user_dir.is_dir():
                for session_file in sorted(user_dir.iterdir()):
                    if session_file.is_file():
                        try:
                            df = load_session_file(session_file)
                            windows = extract_session_windows(df)
                            for w in windows:
                                w["label"] = 1  # legal/human
                                w["source"] = f"train_{user_dir.name}"
                                all_records.append(w)
                        except Exception as e:
                            warnings.warn(f"Error processing {session_file}: {e}")

    # 2. Test files — labels from public_labels.csv
    test_dir = data_dir / "test_files"
    labels_path = data_dir / "public_labels.csv"

    labels_map = {}
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        for _, row in labels_df.iterrows():
            # is_illegal: 1 = illegal (bot), 0 = legal (human)
            labels_map[str(row["filename"])] = 0 if row["is_illegal"] == 1 else 1

    if test_dir.exists():
        for user_dir in sorted(test_dir.iterdir()):
            if user_dir.is_dir():
                for session_file in sorted(user_dir.iterdir()):
                    if session_file.is_file():
                        session_name = session_file.name
                        label = labels_map.get(session_name)
                        if label is None:
                            continue  # skip unlabeled sessions
                        try:
                            df = load_session_file(session_file)
                            windows = extract_session_windows(df)
                            for w in windows:
                                w["label"] = label
                                w["source"] = f"test_{user_dir.name}"
                                all_records.append(w)
                        except Exception as e:
                            warnings.warn(f"Error processing {session_file}: {e}")

    if not all_records:
        raise ValueError("No mouse data was extracted!")

    features = pd.DataFrame(all_records)
    features = features.drop(columns=["source"], errors="ignore")

    # Replace inf/nan
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "ms_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Mouse features saved: {out_path} ({len(features)} rows)")

    return features


def transform_online(raw_data: dict, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Transform raw mouse data for online inference.

    Args:
        raw_data: Dict with mouse movement fields from API request.
        scaler: Fitted StandardScaler (loaded from MLflow).

    Returns:
        Feature vector as numpy array.
    """
    # Expect raw_data to have x, y, timestamps arrays
    timestamps = raw_data.get("timestamps", [0.0, 0.1])
    x_coords = raw_data.get("x", [0, 100])
    y_coords = raw_data.get("y", [0, 100])

    df = pd.DataFrame({
        "record_timestamp": timestamps,
        "x": x_coords,
        "y": y_coords,
        "button": "NoButton",
        "state": "Move",
    })

    windows = extract_session_windows(df)
    if windows:
        # Average across windows
        feat_df = pd.DataFrame(windows)
        features = feat_df[FEATURE_NAMES].mean().values.reshape(1, -1)
    else:
        features = np.zeros((1, len(FEATURE_NAMES)))

    if scaler is not None:
        features = scaler.transform(features)

    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mouse dynamics feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/raw/mouse")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(
        project_root / args.data_dir,
        project_root / args.processed_dir,
    )
