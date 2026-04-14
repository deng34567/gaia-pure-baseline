from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import pandas as pd
except ImportError:
    pd = None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if pd is not None:
        try:
            return bool(pd.isna(value))
        except Exception:
            pass
    return isinstance(value, float) and math.isnan(value)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_parquet(path: Path) -> List[Dict[str, Any]]:
    if pq is not None:
        return pq.read_table(path).to_pylist()
    if pd is not None:
        return pd.read_parquet(path).to_dict("records")
    raise ImportError(
        "Found GAIA parquet metadata at '{}', but neither pyarrow nor pandas is installed. "
        "Install pyarrow or provide metadata.jsonl instead.".format(path)
    )


def _candidate_split_dirs(data_dir: str, split: str) -> List[Path]:
    root = Path(data_dir).expanduser().resolve()
    return [root / "2023" / split, root / split]


def _candidate_metadata_paths(split_dir: Path, level: Optional[int]) -> List[Path]:
    suffix = f".level{level}" if level in {1, 2, 3} else ""
    return [
        split_dir / f"metadata{suffix}.parquet",
        split_dir / f"metadata{suffix}.jsonl",
        split_dir / "metadata.parquet",
        split_dir / "metadata.jsonl",
    ]


def _load_split_records(split_dir: Path, level: Optional[int]) -> List[Dict[str, Any]]:
    metadata_path = next((path for path in _candidate_metadata_paths(split_dir, level) if path.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError(
            "Could not find GAIA metadata under split dir '{}'. Checked: {}".format(
                split_dir,
                [str(path) for path in _candidate_metadata_paths(split_dir, level)],
            )
        )

    if metadata_path.suffix == ".parquet":
        records = _load_parquet(metadata_path)
    else:
        records = _read_jsonl(metadata_path)

    if level in {1, 2, 3} and metadata_path.name in {"metadata.parquet", "metadata.jsonl"}:
        records = [row for row in records if str(row.get("Level")) == str(level)]

    return records


def resolve_attachment_path(row: Dict[str, Any], data_dir: str, split: str) -> Optional[str]:
    dataset_root = Path(data_dir).expanduser().resolve()

    file_path = row.get("file_path")
    if not _is_missing(file_path):
        candidate = str(file_path).strip()
        if candidate:
            if os.path.isabs(candidate):
                return candidate
            return str((dataset_root / candidate).resolve())

    file_name = row.get("file_name")
    if _is_missing(file_name):
        return None

    basename = str(file_name).strip()
    if not basename:
        return None

    for candidate in (
        dataset_root / "2023" / split / basename,
        dataset_root / split / basename,
        dataset_root / basename,
    ):
        if candidate.exists():
            return str(candidate.resolve())
    return str((dataset_root / "2023" / split / basename).resolve())


def format_task(row: Dict[str, Any], split: str, data_dir: str) -> Dict[str, Any]:
    answer = row.get("Final answer")
    attachment_path = resolve_attachment_path(row, data_dir=data_dir, split=split)

    level = row.get("Level")
    if not _is_missing(level):
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = str(level)
    else:
        level = None

    return {
        "id": row.get("task_id", row.get("id", "N/A")),
        "level": level,
        "question": str(row.get("Question", "")).strip(),
        "answer": None if _is_missing(answer) else str(answer),
        "has_attachment": attachment_path is not None,
        "attachment_path": attachment_path,
    }


def load_dataset(
    split: str,
    level: Optional[int] = 1,
    data_limit: Optional[int] = None,
    data_dir: str = "data/GAIA",
) -> List[Dict[str, Any]]:
    split_dir = next((path for path in _candidate_split_dirs(data_dir, split) if path.is_dir()), None)
    if split_dir is None:
        raise FileNotFoundError(
            "GAIA split directory not found under '{}'. Expected one of: {}".format(
                Path(data_dir).expanduser().resolve(),
                [str(path) for path in _candidate_split_dirs(data_dir, split)],
            )
        )

    records = _load_split_records(split_dir, level)
    if data_limit is not None:
        records = records[: data_limit]
    return [format_task(row, split=split, data_dir=data_dir) for row in records]
