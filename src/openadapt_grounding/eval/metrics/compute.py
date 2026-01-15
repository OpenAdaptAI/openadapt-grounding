"""Metrics computation functions."""

from collections import defaultdict
from typing import List, Tuple

from openadapt_grounding.eval.metrics.types import ElementResult, MethodMetrics


def compute_iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """Compute IoU between two bboxes in (x, y, w, h) format.

    Args:
        bbox1: First bounding box (x, y, width, height)
        bbox2: Second bounding box (x, y, width, height)

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to (x1, y1, x2, y2) format
    box1 = (x1, y1, x1 + w1, y1 + h1)
    box2 = (x2, y2, x2 + w2, y2 + h2)

    # Compute intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_point_distance(
    point: Tuple[float, float],
    target_center: Tuple[float, float],
) -> float:
    """Compute Euclidean distance between point and target center.

    Args:
        point: Predicted point (x, y) normalized
        target_center: Target center (x, y) normalized

    Returns:
        Euclidean distance (normalized units)
    """
    px, py = point
    tx, ty = target_center
    return ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5


def aggregate_metrics(
    results: List[ElementResult],
    method_name: str,
    dataset_name: str,
) -> MethodMetrics:
    """Aggregate individual results into method metrics.

    Args:
        results: List of element evaluation results
        method_name: Name of the evaluation method
        dataset_name: Name of the dataset

    Returns:
        Aggregated MethodMetrics
    """
    if not results:
        return MethodMetrics(
            method_name=method_name,
            dataset_name=dataset_name,
            detection_rate=0.0,
            mean_iou=0.0,
            mean_latency_ms=0.0,
            mean_attempts=0.0,
            detection_rate_small=0.0,
            detection_rate_medium=0.0,
            detection_rate_large=0.0,
            detection_rate_by_type={},
            total_elements=0,
            detected_elements=0,
            results=results,
        )

    # Overall detection rate
    detected = sum(1 for r in results if r.found)
    detection_rate = detected / len(results)

    # Mean IoU (only for detected elements with IoU > 0)
    detected_ious = [r.iou for r in results if r.found and r.iou > 0]
    mean_iou = sum(detected_ious) / len(detected_ious) if detected_ious else 0.0

    # Latency and attempts
    mean_latency_ms = sum(r.latency_ms for r in results) / len(results)
    mean_attempts = sum(r.attempts for r in results) / len(results)

    # Breakdown by size
    by_size: dict = defaultdict(list)
    for r in results:
        by_size[r.size_category].append(r.found)

    def rate_for_category(items: List[bool]) -> float:
        return sum(items) / len(items) if items else 0.0

    detection_rate_small = rate_for_category(by_size.get("small", []))
    detection_rate_medium = rate_for_category(by_size.get("medium", []))
    detection_rate_large = rate_for_category(by_size.get("large", []))

    # Breakdown by element type
    by_type: dict = defaultdict(list)
    for r in results:
        by_type[r.element_type].append(r.found)

    detection_rate_by_type = {t: rate_for_category(items) for t, items in by_type.items()}

    return MethodMetrics(
        method_name=method_name,
        dataset_name=dataset_name,
        detection_rate=detection_rate,
        mean_iou=mean_iou,
        mean_latency_ms=mean_latency_ms,
        mean_attempts=mean_attempts,
        detection_rate_small=detection_rate_small,
        detection_rate_medium=detection_rate_medium,
        detection_rate_large=detection_rate_large,
        detection_rate_by_type=detection_rate_by_type,
        total_elements=len(results),
        detected_elements=detected,
        results=results,
    )
