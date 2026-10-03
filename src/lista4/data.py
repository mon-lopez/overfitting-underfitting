from pathlib import Path
import pandas as pd

# Ruta raíz del repo (…/lista_4)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def load_csv(name: str, data_dir: Path | None = None) -> pd.DataFrame:
   
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    return pd.read_csv(base / name)

