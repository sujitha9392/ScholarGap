from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def create_project_folders():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df, path):
    df.to_csv(path, index=False)


def load_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(text))