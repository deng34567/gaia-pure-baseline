from __future__ import annotations

import math
import re
from typing import Optional

FLOAT_TOLERANCE = 1e-3


def normalize_string(text: str) -> str:
    return "".join(str(text).split()).lower()


def extract_final_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    normalized = str(text).strip()
    parts = re.split(r"FINAL ANSWER:\s*", normalized, flags=re.IGNORECASE)
    had_final_answer_marker = len(parts) > 1
    if len(parts) > 1:
        normalized = parts[-1].strip()

    normalized = normalized.strip().strip("`").strip()
    normalized = re.sub(r"^\*\*|\*\*$", "", normalized).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if lines:
        normalized = lines[0] if had_final_answer_marker else lines[-1]

    normalized = normalized.strip().strip("`").strip("\"' ")
    normalized = re.sub(r"^(?:final answer\s*:\s*|answer\s*:\s*)", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(" .")
    return normalized


def _parse_full_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return float(text)
    return None


def _normalize_gaia_list(text: str) -> list[str]:
    items = [item.strip() for item in str(text).split(",")]
    return [normalize_string(item.strip("\"' ").rstrip(".")) for item in items if item.strip()]


def check_gaia(final_ans: Optional[str], true_ans: Optional[str]) -> bool:
    if final_ans is None or true_ans is None:
        return False

    pred = extract_final_answer(final_ans)
    gold = extract_final_answer(true_ans)
    if not pred or not gold:
        return False

    pred_num = _parse_full_number(pred)
    gold_num = _parse_full_number(gold)
    if pred_num is not None and gold_num is not None:
        if not (math.isfinite(pred_num) and math.isfinite(gold_num)):
            return False
        if abs(pred_num - gold_num) < FLOAT_TOLERANCE:
            return True
        return round(pred_num) == round(gold_num)

    if normalize_string(pred) == normalize_string(gold):
        return True

    if "," in pred and "," in gold:
        return _normalize_gaia_list(pred) == _normalize_gaia_list(gold)

    return False
