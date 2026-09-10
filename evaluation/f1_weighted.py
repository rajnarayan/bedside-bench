"""Weighted F1 for ``f1_weighted_rubric`` cases."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

POINTS_TO_CLASS = {3: 9, 2: 8, 1: 7, 0: 5, -1: 3, -2: 2, -3: 1}
SEVERITY_WEIGHTS = {
    "severe": 72.0,
    "moderate": 24.0,
    "mild": 3.0,
    "uncertain": 1.0,
}


def points_to_class(points: float) -> int:
    return POINTS_TO_CLASS.get(int(points), 5)


def severity_weight(severity_class: int) -> float:
    score = int(severity_class)
    if score in (1, 9):
        return SEVERITY_WEIGHTS["severe"]
    if score in (2, 8):
        return SEVERITY_WEIGHTS["moderate"]
    if score in (3, 7):
        return SEVERITY_WEIGHTS["mild"]
    if score in (4, 5, 6):
        return SEVERITY_WEIGHTS["uncertain"]
    return 0.0


def _severity_class(item: Mapping[str, Any]) -> int:
    if "severity_class" in item:
        return int(item["severity_class"])
    return points_to_class(float(item.get("points") or 0))


def precision_recall_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _item_id(item: Mapping[str, Any], index: int) -> str:
    return str(item.get("item_id") or f"item_{index}")


def _severe_error(items: list[Mapping[str, Any]], matched: set[str]) -> bool:
    for index, item in enumerate(items):
        score = _severity_class(item)
        is_matched = _item_id(item, index) in matched
        if score == 9 and not is_matched:
            return True
        if score == 1 and is_matched:
            return True
    return False


def _recall_weighted(
    items: list[Mapping[str, Any]], matched: set[str]
) -> float:
    numerator = 0.0
    denominator = 0.0
    for index, item in enumerate(items):
        score = _severity_class(item)
        if score < 7:
            continue
        weight = severity_weight(score)
        denominator += weight
        if _item_id(item, index) in matched:
            numerator += weight
    return numerator / denominator if denominator > 0.0 else 0.0


def _precision_weighted(
    items: list[Mapping[str, Any]], matched: Iterable[str]
) -> float:
    by_id = {_item_id(item, index): item for index, item in enumerate(items)}
    numerator = 0.0
    denominator = 0.0
    for raw_id in matched:
        item = by_id.get(str(raw_id))
        if item is None:
            continue
        score = _severity_class(item)
        weight = severity_weight(score)
        if weight == 0.0:
            continue
        denominator += weight
        if score >= 7:
            numerator += weight
    if denominator <= 0.0:
        return math.nan
    return numerator / denominator


def compute_f1_weighted(
    items: list[Mapping[str, Any]],
    matched_ids: Iterable[str],
) -> dict[str, float | None]:
    """Return weighted F1 for one bedside case.

    ``severe_rate`` is reported as a diagnostic but does not gate the score. A
    severe omission still costs its class-9 weight in the recall denominator,
    and a severe commission still costs its class-1 weight in the precision
    denominator, so severity is priced through the weights alone.
    """
    matched = {str(item_id) for item_id in matched_ids}
    severe = _severe_error(items, matched)
    recall = _recall_weighted(items, matched)
    precision = _precision_weighted(items, matched)
    if precision is None or math.isnan(precision):
        f1 = 0.0
    else:
        f1 = precision_recall_f1(precision, recall)
    return {
        "f1_weighted": round(f1, 4),
        "precision_weighted": None if math.isnan(precision) else round(precision, 4),
        "recall_weighted": round(recall, 4),
        "severe_rate": 1.0 if severe else 0.0,
    }
