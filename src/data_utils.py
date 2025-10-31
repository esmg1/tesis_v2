from __future__ import annotations

import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
RAW_DATA_DIRS: List[Path] = [
    Path("samples"),
    Path("data_samples"),
    Path("data/raw"),
    Path("../samples"),
    Path("../data_samples"),
]


def ensure_curated_events(target: Path | str = Path("data/curated/events.parquet")) -> Path:
    """Create the curated events parquet if it does not exist."""

    target_path = Path(target)
    if target_path.exists():
        return target_path

    df = build_curated_events()
    if df.empty:
        raise FileNotFoundError(
            "Unable to build events dataset; no usable raw Excel files were found."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False)
    return target_path


def build_curated_events(raw_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    """Aggregate the raw Excel files into region/date counts."""

    frames: List[pd.DataFrame] = []
    for file_path in _discover_excel_files(raw_dirs):
        frame = _load_events_from_excel(file_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["region", "date", "y"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["date"].notna() & combined["region"].notna()]
    combined["date"] = combined["date"].dt.floor("D")
    combined["region"] = combined["region"].astype("string").str.strip()
    grouped = (
        combined.groupby(["region", "date"], as_index=False)
        .size()
        .rename(columns={"size": "y"})
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
    grouped["y"] = grouped["y"].astype("int64")
    return grouped


def _discover_excel_files(raw_dirs: Optional[Iterable[Path]]) -> Iterable[Path]:
    candidates = list(raw_dirs) if raw_dirs is not None else RAW_DATA_DIRS
    seen = set()
    for root in candidates:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        for pattern in ("*.xlsx", "*.xls"):
            for file_path in sorted(root.glob(pattern)):
                if file_path.name.startswith("~$"):
                    continue
                real = file_path.resolve()
                if real in seen:
                    continue
                seen.add(real)
                yield real


def _load_events_from_excel(path: Path) -> pd.DataFrame:
    if path.name.lower().startswith("book1"):
        # ``Book1.xlsx`` is a toy duplicate of the homicide data; skip to avoid double counting.
        return pd.DataFrame(columns=["region", "date"])

    raw = _read_excel_with_fallback(path)
    if raw.empty:
        return pd.DataFrame(columns=["region", "date"])

    rename_map = {col: _normalize_column(col) for col in raw.columns if isinstance(col, str)}
    raw = raw.rename(columns=rename_map)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    date_col = None
    for candidate in ("FECHA_APREHENSION", "FECHA_INFRACCION", "FECHA_ACTA"):
        if candidate in raw.columns:
            date_col = candidate
            break
    if date_col is None:
        return pd.DataFrame(columns=["region", "date"])

    region_col = None
    for candidate in ("PROVINCIA", "SUBZONA", "ZONA", "CANTON", "DISTRITO"):
        if candidate in raw.columns:
            region_col = candidate
            break
    if region_col is None:
        return pd.DataFrame(columns=["region", "date"])

    dates = _coerce_dates(raw[date_col])
    regions = _clean_region(raw[region_col])
    out = pd.DataFrame({"region": regions, "date": dates})
    return out


def _normalize_column(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("/", " ").replace("\\", " ").replace("-", " ")
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = "_".join(part for part in text.split() if part)
    return text.upper()


def _coerce_dates(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    if s.empty:
        return pd.Series([], dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None).dt.floor("D")

    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    numeric = pd.to_numeric(s, errors="coerce")
    mask = numeric.notna()
    if mask.any():
        result.loc[mask] = (EXCEL_EPOCH + pd.to_timedelta(numeric[mask], unit="D")).values
    remaining = result.isna()
    if remaining.any():
        result.loc[remaining] = pd.to_datetime(
            s.loc[remaining], errors="coerce", dayfirst=True
        ).values
    return result.dt.floor("D")


def _clean_region(series: pd.Series) -> pd.Series:
    def _normalize(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"nan", "none", "na"}:
            return None
        return " ".join(text.split()).upper()

    return series.apply(_normalize)


def _read_excel_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, dtype=object)
    except Exception:
        return _read_excel_via_xml(path)


def _read_excel_via_xml(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            with zf.open("xl/sharedStrings.xml") as handle:
                for event, elem in ET.iterparse(handle, events=("end",)):
                    if elem.tag.endswith("}t"):
                        shared_strings.append(elem.text or "")
                    elem.clear()

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            alternatives = sorted(
                name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")
            )
            if not alternatives:
                return pd.DataFrame()
            sheet_name = alternatives[0]

        rows: List[dict[int, Optional[str]]] = []
        max_col = -1
        with zf.open(sheet_name) as handle:
            for event, elem in ET.iterparse(handle, events=("end",)):
                if elem.tag.endswith("}row"):
                    row_map: dict[int, Optional[str]] = {}
                    current_idx = -1
                    for cell in elem:
                        if not cell.tag.endswith("}c"):
                            continue
                        ref = cell.attrib.get("r")
                        idx = _cell_index(ref)
                        if idx is None:
                            idx = current_idx + 1
                        current_idx = idx

                        value: Optional[str] = None
                        cell_type = cell.attrib.get("t")
                        for child in cell:
                            tag = child.tag
                            if tag.endswith("}v"):
                                value = child.text
                            elif tag.endswith("}is"):
                                value = "".join(
                                    node.text or "" for node in child.iter() if node.tag.endswith("}t")
                                )
                        if cell_type == "s" and value is not None:
                            try:
                                value = shared_strings[int(value)]
                            except (ValueError, IndexError):
                                value = None
                        row_map[idx] = value
                        if idx > max_col:
                            max_col = idx
                    if any(v is not None for v in row_map.values()):
                        rows.append(row_map)
                    elem.clear()

        if not rows:
            return pd.DataFrame()

        width = max_col + 1
        matrix = [[row.get(i) for i in range(width)] for row in rows]
        header, *data = matrix
        # Drop completely empty columns
        if not any(isinstance(col, str) and col.strip() for col in header):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=header)
        df = df.loc[:, [col for col in df.columns if isinstance(col, str) and col.strip()]]
        return df


def _cell_index(cell_ref: Optional[str]) -> Optional[int]:
    if not cell_ref:
        return None
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    if not letters:
        return None
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1
