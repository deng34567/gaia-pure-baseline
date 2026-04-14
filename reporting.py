from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metric_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("correct") is not None]
    correct_count = sum(1 for row in scored if row.get("correct") is True)
    accuracy = (correct_count / len(scored)) if scored else None
    return {
        "count": len(rows),
        "scored_count": len(scored),
        "correct_count": correct_count,
        "accuracy": accuracy,
    }


def build_summary(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    summary = _metric_payload(rows)
    summary.update(
        {
            "model_name": config["model_name"],
            "split": config["split"],
            "level": config["level"],
            "limit": config["limit"],
            "gaia_data_dir": config["gaia_data_dir"],
            "output_dir": str(Path(config["output_dir"]).expanduser().resolve()),
        }
    )
    return summary


def build_metrics_by_level(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    levels = sorted({row.get("level") for row in rows}, key=lambda value: (value is None, value))
    return {str(level): _metric_payload([row for row in rows if row.get("level") == level]) for level in levels}


def build_metrics_by_attachment_presence(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    with_attachment = [row for row in rows if row.get("has_attachment")]
    without_attachment = [row for row in rows if not row.get("has_attachment")]
    return {
        "with_attachment": _metric_payload(with_attachment),
        "without_attachment": _metric_payload(without_attachment),
    }
