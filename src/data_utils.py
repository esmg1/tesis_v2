from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd


_COLUMN_ALIASES = {
    "fecha_aprehension": "fecha_aprehension",
    "fecha_infraccion": "fecha_aprehension",
    "fecha_acta": "fecha_aprehension",
    "provincia": "provincia",
    "provincia_decomiso": "provincia",
    "subzona": "subzona",
    "zona": "zona",
}


def load_seizure_data(path: str | Path) -> pd.DataFrame:
    """Load BDD_HI or C4I2 Excel files and normalize key column names."""

    file_path = Path(path)
    df = pd.read_excel(file_path, dtype=object)
    df.columns = [_normalize_column_name(col) for col in df.columns]
    df = df.rename(columns={col: _COLUMN_ALIASES.get(col, col) for col in df.columns})
    return df


def _normalize_column_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for ch in (" ", "-", "/", "\\"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")
