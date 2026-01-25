from __future__ import annotations

from pathlib import Path


FOLDERS = [
    Path("data/raw"),
    Path("data/processed"),
    Path("src/models"),
    Path("src/features"),
    Path("notebooks"),
    Path("thesis/figures"),
]

REQUIREMENTS = "\n".join(
    [
        "pymc",
        "pytensor",
        "torch",
        "geopandas",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "dagshub",
    ]
)

AGENTS_CONTENT = "\n".join(
    [
        "# AGENTS",
        "- Use PEP8.",
        "- Prioritize PyTorch for neural components.",
        "- Use PyMC for Bayesian inference.",
        "- Always include uncertainty estimation in model outputs.",
    ]
)

DATA_UTILS_CONTENT = """from __future__ import annotations

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
    \"\"\"Load BDD_HI or C4I2 Excel files and normalize key column names.\"\"\"

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
    for ch in (" ", "-", "/", "\\\\"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")
"""


def _write_file(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    for folder in FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)

    _write_file(Path("requirements.txt"), REQUIREMENTS + "\n")
    _write_file(Path("AGENTS.md"), AGENTS_CONTENT + "\n")
    _write_file(Path("src/data_utils.py"), DATA_UTILS_CONTENT + "\n")


if __name__ == "__main__":
    main()
