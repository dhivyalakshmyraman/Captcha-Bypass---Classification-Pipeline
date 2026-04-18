"""Setup script to organize raw data into data/raw/ directory structure."""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"


def setup_dirs() -> None:
    """Create all required directories."""
    for d in [
        RAW_DIR / "keystroke",
        RAW_DIR / "mouse",
        RAW_DIR / "fingerprint",
        RAW_DIR / "network",
        RAW_DIR / "webbot",
        PROCESSED_DIR,
        REPORTS_DIR,
        MLFLOW_DIR,
        PROJECT_ROOT / "src" / "features",
        PROJECT_ROOT / "src" / "models",
        PROJECT_ROOT / "src" / "gan",
        PROJECT_ROOT / "src" / "api",
        PROJECT_ROOT / "src" / "validation",
    ]:
        d.mkdir(parents=True, exist_ok=True)
        init_file = d / "__init__.py"
        if "src" in str(d) and not init_file.exists():
            init_file.touch()

    # Also create src/__init__.py
    src_init = PROJECT_ROOT / "src" / "__init__.py"
    if not src_init.exists():
        src_init.touch()


def link_data() -> None:
    """Copy/symlink raw datasets into data/raw/ structure."""
    mappings = [
        (
            PROJECT_ROOT / "Keyboard_Strokes_DSL-StrongPasswordData.csv",
            RAW_DIR / "keystroke" / "DSL-StrongPasswordData.csv",
        ),
        (
            PROJECT_ROOT / "Mouse-Dynamics-Challenge-master",
            RAW_DIR / "mouse",
        ),
        (
            PROJECT_ROOT / "Browser Fingerprint Dataset",
            RAW_DIR / "fingerprint",
        ),
        (
            PROJECT_ROOT / "Network Intrusion Dataset",
            RAW_DIR / "network",
        ),
        (
            PROJECT_ROOT / "web_bot_detection_dataset",
            RAW_DIR / "webbot",
        ),
    ]

    for src, dst in mappings:
        if not src.exists():
            print(f"WARNING: Source not found: {src}")
            continue

        if src.is_file():
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"Copied {src.name} -> {dst}")
            else:
                print(f"Already exists: {dst}")
        else:
            # For directories, copy contents into dst
            if dst.name == "mouse":
                # Copy specific subdirs
                for sub in ["training_files", "test_files"]:
                    src_sub = src / sub
                    dst_sub = dst / sub
                    if src_sub.exists() and not dst_sub.exists():
                        shutil.copytree(src_sub, dst_sub)
                        print(f"Copied {sub} -> {dst_sub}")
                # Copy labels file
                labels_src = src / "public_labels.csv"
                labels_dst = dst / "public_labels.csv"
                if labels_src.exists() and not labels_dst.exists():
                    shutil.copy2(labels_src, labels_dst)
                    print(f"Copied public_labels.csv -> {labels_dst}")
            elif dst.name == "fingerprint":
                # Copy all html files and dataset.json
                for f in src.iterdir():
                    target = dst / f.name
                    if not target.exists():
                        if f.is_file():
                            shutil.copy2(f, target)
                        else:
                            shutil.copytree(f, target)
                print(f"Copied fingerprint data -> {dst}")
            elif dst.name == "network":
                for f in src.iterdir():
                    target = dst / f.name
                    if f.is_file() and not target.exists():
                        shutil.copy2(f, target)
                print(f"Copied network data -> {dst}")
            elif dst.name == "webbot":
                for sub in src.iterdir():
                    target = dst / sub.name
                    if not target.exists():
                        if sub.is_file():
                            shutil.copy2(sub, target)
                        else:
                            shutil.copytree(sub, target)
                print(f"Copied webbot data -> {dst}")


if __name__ == "__main__":
    print("Setting up project directories...")
    setup_dirs()
    print("\nLinking raw data...")
    link_data()
    print("\nDone! Project structure ready.")
