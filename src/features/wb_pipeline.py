"""Web bot detection feature extraction pipeline.

Extracts web traffic features from the m4d web bot detection dataset.
Works both offline (called by training scripts) and online (imported by API).
"""
import re
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Any
from datetime import datetime
import warnings
from collections import Counter

FEATURE_NAMES = [
    "request_rate",
    "ua_entropy",
    "method_diversity",
    "js_enabled",
    "timing_regularity",
    "referrer_present",
    "unique_urls",
    "avg_response_size",
    "error_rate",
]

# Apache Combined Log Format regex
LOG_PATTERN = re.compile(
    r'^(.*?) - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" (.*?) "(.*?)"$'
)

DATETIME_FMT = "%d/%b/%Y:%H:%M:%S %z"


def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string's characters.

    Args:
        s: Input string.

    Returns:
        Entropy value.
    """
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def parse_log_line(line: str) -> Optional[dict]:
    """Parse a single Apache access log line.

    Args:
        line: Raw log line string.

    Returns:
        Dict with parsed fields, or None on failure.
    """
    m = LOG_PATTERN.match(line.strip())
    if not m:
        return None

    ip = m.group(1)
    timestamp_str = m.group(2)
    request = m.group(3)
    status = int(m.group(4))
    size = int(m.group(5))
    referrer = m.group(6)
    session_id = m.group(7)
    user_agent = m.group(8)

    # Parse timestamp
    try:
        timestamp = datetime.strptime(timestamp_str, DATETIME_FMT)
    except ValueError:
        timestamp = None

    # Parse request method and URL
    parts = request.split(" ")
    method = parts[0] if parts else "GET"
    url = parts[1] if len(parts) > 1 else "/"

    return {
        "ip": ip,
        "timestamp": timestamp,
        "method": method,
        "url": url,
        "status": status,
        "size": size,
        "referrer": referrer,
        "session_id": session_id,
        "user_agent": user_agent,
    }


def extract_session_features(session_logs: List[dict]) -> dict:
    """Extract features from a session's log entries.

    Args:
        session_logs: List of parsed log entry dicts for one session.

    Returns:
        Feature dict.
    """
    if len(session_logs) < 2:
        return None

    # Sort by timestamp
    session_logs = sorted(session_logs, key=lambda x: x["timestamp"] or datetime.min)

    timestamps = [l["timestamp"] for l in session_logs if l["timestamp"]]
    if len(timestamps) < 2:
        return None

    # Session duration in seconds
    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    if duration <= 0:
        duration = 1.0

    # Request rate
    request_rate = len(session_logs) / duration

    # User-agent entropy
    ua = session_logs[0].get("user_agent", "")
    ua_entropy = shannon_entropy(ua)

    # Method diversity (hash of unique methods)
    methods = set(l["method"] for l in session_logs)
    method_diversity = len(methods)

    # JS enabled: check if session requested any /js/ resources
    js_urls = [l for l in session_logs if "/js/" in l.get("url", "")]
    js_enabled = 1 if js_urls else 0

    # Timing regularity: std of inter-request gaps
    time_diffs = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                  for i in range(len(timestamps) - 1)]
    timing_regularity = np.std(time_diffs) if time_diffs else 0.0

    # Referrer presence
    refs_present = sum(1 for l in session_logs if l["referrer"] not in ("-", "", "None"))
    referrer_present = refs_present / len(session_logs)

    # Unique URLs
    unique_urls = len(set(l["url"] for l in session_logs))

    # Average response size
    sizes = [l["size"] for l in session_logs]
    avg_response_size = np.mean(sizes) if sizes else 0.0

    # Error rate (4xx/5xx)
    errors = sum(1 for l in session_logs if l["status"] >= 400)
    error_rate = errors / len(session_logs)

    return {
        "request_rate": request_rate,
        "ua_entropy": ua_entropy,
        "method_diversity": method_diversity,
        "js_enabled": js_enabled,
        "timing_regularity": timing_regularity,
        "referrer_present": referrer_present,
        "unique_urls": unique_urls,
        "avg_response_size": avg_response_size,
        "error_rate": error_rate,
    }


def load_annotations(webbot_dir: Path) -> dict:
    """Load session labels from annotation files.

    Args:
        webbot_dir: Path to data/raw/webbot/.

    Returns:
        Dict mapping session_id -> label (1=human, 0=bot).
    """
    labels = {}

    for phase in ["phase1", "phase2"]:
        annot_dir = webbot_dir / phase / "annotations"
        if not annot_dir.exists():
            continue

        for subdir in annot_dir.iterdir():
            if subdir.is_dir():
                for annot_file in subdir.iterdir():
                    if annot_file.is_file():
                        with open(annot_file) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    sid = parts[0]
                                    lbl_str = parts[1]
                                    if lbl_str == "human":
                                        labels[sid] = 1
                                    elif "bot" in lbl_str.lower():
                                        labels[sid] = 0

    return labels


def run_pipeline(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run the full web bot feature extraction pipeline.

    Args:
        data_dir: Path to data/raw/webbot/ containing log files and annotations.
        processed_dir: Path to data/processed/ for output.

    Returns:
        DataFrame with features and labels.
    """
    # Load labels
    labels = load_annotations(data_dir)
    print(f"Loaded {len(labels)} session annotations")

    # Parse all log files grouped by session
    sessions: Dict[str, List[dict]] = {}

    for phase in ["phase1", "phase2"]:
        logs_dir = data_dir / phase / "data" / "web_logs"
        if not logs_dir.exists():
            continue

        for category in ["humans", "bots"]:
            cat_dir = logs_dir / category
            if not cat_dir.exists():
                continue

            for log_file in sorted(cat_dir.iterdir()):
                if log_file.is_file() and log_file.suffix == ".log":
                    with open(log_file, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            parsed = parse_log_line(line)
                            if parsed and parsed["session_id"] != "-":
                                sid = parsed["session_id"]
                                if sid not in sessions:
                                    sessions[sid] = []
                                sessions[sid].append(parsed)

    print(f"Parsed {len(sessions)} unique sessions from logs")

    # Extract features for labeled sessions
    records = []
    for sid, logs in sessions.items():
        if sid in labels:
            feats = extract_session_features(logs)
            if feats is not None:
                feats["label"] = labels[sid]
                records.append(feats)

    # Also include labeled sessions that might not have web logs
    # (they may only have mouse movement data — skip those)

    if not records:
        raise ValueError("No web bot data extracted!")

    features = pd.DataFrame(records)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "wb_features.parquet"
    features.to_parquet(out_path, index=False)

    label_counts = features["label"].value_counts()
    print(f"Web bot features saved: {out_path} ({len(features)} rows)")
    print(f"  Human (1): {label_counts.get(1, 0)}, Bot (0): {label_counts.get(0, 0)}")

    return features


def transform_online(raw_data: dict, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Transform raw web request data for online inference.

    Args:
        raw_data: Dict with web request fields from API request.
        scaler: Fitted StandardScaler (loaded from MLflow).

    Returns:
        Feature vector as numpy array.
    """
    ua = raw_data.get("user_agent", "")

    features = np.array([[
        raw_data.get("request_rate", 1.0),
        shannon_entropy(ua),
        raw_data.get("method_diversity", 1),
        raw_data.get("js_enabled", 1),
        raw_data.get("timing_regularity", 1.0),
        raw_data.get("referrer_present", 0.5),
        raw_data.get("unique_urls", 5),
        raw_data.get("avg_response_size", 1000),
        raw_data.get("error_rate", 0.0),
    ]])

    if scaler is not None:
        features = scaler.transform(features)

    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Web bot feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/raw/webbot")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(
        project_root / args.data_dir,
        project_root / args.processed_dir,
    )
