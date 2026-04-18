"""Data validation reports for each signal.

Uses deepchecks' individual checks (avoiding broken scorer suites),
plus a fallback pandas-based summary if deepchecks fails.
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _pandas_report(df: pd.DataFrame, signal_name: str, reports_dir: Path):
    """Fallback: generate a simple HTML report using pandas profiling."""
    report_path = reports_dir / f"{signal_name}_data_validation.html"

    label_counts = df["label"].value_counts().to_dict()
    null_rates = (df.isnull().mean() * 100).round(2).to_dict()
    desc = df.describe().round(4)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{signal_name} Data Validation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }}
    h1 {{ color: #333; }} h2 {{ color: #555; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
    th {{ background: #4a90d9; color: white; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
              background:#4a90d9; color:white; font-size:0.85em; }}
  </style>
</head>
<body>
  <h1>Data Validation Report: <span class="badge">{signal_name}</span></h1>
  <p><b>Total rows:</b> {len(df):,} &nbsp;|&nbsp;
     <b>Features:</b> {df.shape[1] - 1} &nbsp;|&nbsp;
     <b>Label counts:</b> {label_counts}</p>

  <h2>Null Rates (%)</h2>
  <table>
    <tr><th>Column</th><th>Null %</th></tr>
    {"".join(f"<tr><td>{c}</td><td>{v}</td></tr>" for c, v in null_rates.items())}
  </table>

  <h2>Descriptive Statistics</h2>
  {desc.to_html(classes="", border=0)}
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Pandas report saved: {report_path}")


def run_validation_for_signal(signal_name: str, processed_dir: Path, reports_dir: Path):
    """Run data integrity checks and save an HTML report."""
    parquet_path = processed_dir / f"{signal_name}_features.parquet"
    if not parquet_path.exists():
        logger.warning(f"File {parquet_path} missing, skipping")
        return

    df = pd.read_parquet(parquet_path)
    if df.empty or "label" not in df.columns:
        logger.warning(f"Empty or no label column in {parquet_path}, skipping")
        return

    logger.info(f"Checking {signal_name} data integrity...")

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.checks import (
            DataDuplicates,
            StringMismatch,
            MixedNulls,
            FeatureLabelCorrelation,
            ClassImbalance,
            OutlierSampleDetection,
        )
        from deepchecks.core import CheckResult

        ds = Dataset(df, label="label", cat_features=[])
        checks = [
            DataDuplicates(),
            MixedNulls(),
            ClassImbalance(),
            FeatureLabelCorrelation(),
        ]

        results = []
        for check in checks:
            try:
                r = check.run(ds)
                results.append(r)
            except Exception as ce:
                logger.warning(f"  Check {check.__class__.__name__} failed: {ce}")

        # Save combined HTML
        if results:
            report_path = reports_dir / f"{signal_name}_data_validation.html"
            # build a simple combined HTML
            sections = ""
            for r in results:
                try:
                    sections += f"<h2>{r.check.__class__.__name__}</h2>"
                    sections += f"<pre>{r.value}</pre>"
                except Exception:
                    pass
            html = f"<html><body><h1>{signal_name} Validation</h1>{sections}</body></html>"
            report_path.write_text(html, encoding="utf-8")
            logger.info(f"Deepchecks report saved: {report_path}")
        else:
            _pandas_report(df, signal_name, reports_dir)

    except Exception as e:
        logger.warning(f"Deepchecks failed ({e}), falling back to pandas report")
        _pandas_report(df, signal_name, reports_dir)


if __name__ == "__main__":
    processed = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    signals = ["ks", "ms", "fp", "net", "wb"]
    for sig in signals:
        run_validation_for_signal(sig, processed, reports_dir)
