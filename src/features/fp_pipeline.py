"""Browser fingerprint feature extraction pipeline.

Extracts fingerprint features from the Browser Fingerprint dataset (JSON + HTML).
Works both offline (called by training scripts) and online (imported by API).
"""
import hashlib
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup

FEATURE_NAMES = [
    "canvas_hash", "webgl_renderer_hash",
    "ua_bot_score", "timezone_offset",
    "language_count", "plugin_count",
    "screen_resolution_category",
    "touch_support", "webdriver_flag", "font_count",
]

# Timezone to UTC offset mapping (common ones)
TZ_OFFSETS = {
    "America/New_York": -5, "America/Chicago": -6, "America/Denver": -7,
    "America/Los_Angeles": -8, "America/Anchorage": -9, "Pacific/Honolulu": -10,
    "Europe/London": 0, "Europe/Paris": 1, "Europe/Berlin": 1,
    "Europe/Moscow": 3, "Asia/Dubai": 4, "Asia/Kolkata": 5.5,
    "Asia/Shanghai": 8, "Asia/Tokyo": 9, "Australia/Sydney": 11,
    "Pacific/Auckland": 12, "UTC": 0, "Etc/GMT": 0,
}


def hash_to_int(value: str) -> int:
    """Convert a string to a stable integer hash.

    Args:
        value: String to hash.

    Returns:
        Integer hash value.
    """
    if not value or value in ("None", "null", "undefined"):
        return 0
    return int(hashlib.md5(value.encode()).hexdigest()[:8], 16) % 100000


def get_tz_offset(tz_string: str) -> float:
    """Convert timezone string to UTC offset.

    Args:
        tz_string: Timezone string like 'Asia/Kolkata'.

    Returns:
        UTC offset as float.
    """
    if not tz_string:
        return 0.0
    return TZ_OFFSETS.get(tz_string, 0.0)


def categorize_screen(width: int) -> int:
    """Categorize screen resolution into buckets.

    Args:
        width: Screen width in pixels.

    Returns:
        Category integer: 0=mobile, 1=tablet, 2=desktop, 3=unusual.
    """
    if width < 768:
        return 0  # mobile
    elif width < 1024:
        return 1  # tablet
    elif width <= 1920:
        return 2  # desktop
    else:
        return 3  # unusual


def extract_from_json_entry(entry: Dict[str, Any]) -> dict:
    """Extract features from a single dataset.json entry.

    Args:
        entry: Dict from dataset.json.

    Returns:
        Feature dict.
    """
    ua_info = entry.get("userAgent", {})
    screen_info = entry.get("screen", {})
    feature_info = entry.get("feature", {})
    lang_info = entry.get("language", {})
    device_info = entry.get("devices", {})

    ua_string = ua_info.get("ua", "")
    ua_bot_score = 0
    if "HeadlessChrome" in ua_string:
        ua_bot_score = 1
    elif "PhantomJS" in ua_string:
        ua_bot_score = 1

    canvas_hash = hash_to_int(feature_info.get("canvasHash", ""))
    webgl_hash = hash_to_int(feature_info.get("webglHash", ""))

    languages = lang_info.get("languages", [])
    timezone = lang_info.get("timezone", "")

    screen_width = screen_info.get("width", 1920)

    return {
        "canvas_hash": canvas_hash,
        "webgl_renderer_hash": webgl_hash,
        "ua_bot_score": ua_bot_score,
        "timezone_offset": get_tz_offset(timezone),
        "language_count": len(languages) if isinstance(languages, list) else 1,
        "plugin_count": device_info.get("hardwareConcurrency", 4),
        "screen_resolution_category": categorize_screen(screen_width),
        "touch_support": 0,  # not available in dataset
        "webdriver_flag": 1 if ua_info.get("webdriver", False) else 0,
        "font_count": device_info.get("deviceMemory", 8),
        "label": 1 if entry.get("label") == "human" else 0,
    }


def extract_from_html(filepath: Path) -> dict:
    """Extract features from a single HTML fingerprint file.

    Args:
        filepath: Path to the HTML file.

    Returns:
        Feature dict.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")
    cards = soup.find_all("div", class_="card")

    data = {}
    for card in cards:
        text = card.get_text()
        if ":" in text:
            key, val = text.split(":", 1)
            data[key.strip().lower()] = val.strip()

    # Determine label from filename or meta tag
    label_meta = soup.find("meta", attrs={"name": "label"})
    if label_meta:
        label = 1 if label_meta.get("content") == "human" else 0
    else:
        label = 1 if "human" in filepath.stem.lower() else 0

    ua_string = data.get("user agent", "")
    ua_bot_score = 0
    if "HeadlessChrome" in ua_string:
        ua_bot_score = 1
    elif "PhantomJS" in ua_string:
        ua_bot_score = 1

    # Parse screen
    screen_str = data.get("screen", "1920x1080")
    try:
        screen_w = int(screen_str.split("x")[0])
    except (ValueError, IndexError):
        screen_w = 1920

    # Parse languages
    langs = data.get("languages", "en-US")
    language_count = len(langs.split(","))

    webdriver_str = data.get("webdriver", "false")
    webdriver_flag = 1 if webdriver_str.lower() == "true" else 0

    cpu = int(data.get("cpu", "4"))
    ram = int(data.get("ram", "8"))

    return {
        "canvas_hash": hash_to_int(data.get("canvas hash", "")),
        "webgl_renderer_hash": hash_to_int(data.get("webgl hash", "")),
        "ua_bot_score": ua_bot_score,
        "timezone_offset": get_tz_offset(data.get("timezone", "")),
        "language_count": language_count,
        "plugin_count": cpu,
        "screen_resolution_category": categorize_screen(screen_w),
        "touch_support": 0,
        "webdriver_flag": webdriver_flag,
        "font_count": ram,
        "label": label,
    }


def run_pipeline(data_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Run the full fingerprint feature extraction pipeline.

    Args:
        data_dir: Path to data/raw/fingerprint/ containing HTML files and dataset.json.
        processed_dir: Path to data/processed/ for output.

    Returns:
        DataFrame with features and labels.
    """
    records = []

    # Prefer dataset.json if available
    json_path = data_dir / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            dataset = json.load(f)
        for entry in dataset:
            rec = extract_from_json_entry(entry)
            records.append(rec)
        print(f"Extracted {len(records)} entries from dataset.json")
    else:
        # Fallback: parse HTML files
        for html_file in sorted(data_dir.glob("*.html")):
            if html_file.stem.startswith(("bot_", "human_")):
                try:
                    rec = extract_from_html(html_file)
                    records.append(rec)
                except Exception as e:
                    print(f"Warning: Failed to parse {html_file.name}: {e}")
        print(f"Extracted {len(records)} entries from HTML files")

    if not records:
        raise ValueError("No fingerprint data extracted!")

    features = pd.DataFrame(records)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "fp_features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Fingerprint features saved: {out_path} ({len(features)} rows)")

    return features


def transform_online(raw_data: dict, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Transform raw fingerprint data for online inference.

    Args:
        raw_data: Dict with fingerprint fields from API request.
        scaler: Fitted StandardScaler (loaded from MLflow).

    Returns:
        Feature vector as numpy array.
    """
    ua_string = raw_data.get("user_agent", "")
    ua_bot_score = 0
    if "HeadlessChrome" in ua_string:
        ua_bot_score = 1
    elif "PhantomJS" in ua_string:
        ua_bot_score = 1

    features = np.array([[
        hash_to_int(raw_data.get("canvas_hash", "")),
        hash_to_int(raw_data.get("webgl_hash", "")),
        ua_bot_score,
        get_tz_offset(raw_data.get("timezone", "")),
        raw_data.get("language_count", 1),
        raw_data.get("plugin_count", 4),
        categorize_screen(raw_data.get("screen_width", 1920)),
        raw_data.get("touch_support", 0),
        1 if raw_data.get("webdriver", False) else 0,
        raw_data.get("font_count", 8),
    ]])

    if scaler is not None:
        features = scaler.transform(features)

    return features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fingerprint feature extraction")
    parser.add_argument("--data_dir", type=str, default="data/raw/fingerprint")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    run_pipeline(
        project_root / args.data_dir,
        project_root / args.processed_dir,
    )
