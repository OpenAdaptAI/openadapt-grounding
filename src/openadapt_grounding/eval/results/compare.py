"""Compare evaluation results across methods."""

from collections import defaultdict
from typing import Any, Dict, List

from openadapt_grounding.eval.metrics.types import MethodMetrics


def compare_methods(all_metrics: List[MethodMetrics]) -> Dict[str, Any]:
    """Generate comparison summary across all methods.

    Args:
        all_metrics: List of MethodMetrics from different method/dataset runs

    Returns:
        Comparison summary dict
    """
    if not all_metrics:
        return {"error": "No metrics provided"}

    # Group by dataset
    by_dataset: Dict[str, List[MethodMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_dataset[m.dataset_name].append(m)

    # Find best method per dataset
    best_per_dataset = {}
    for dataset, methods in by_dataset.items():
        best = max(methods, key=lambda x: x.detection_rate)
        best_per_dataset[dataset] = {
            "method": best.method_name,
            "detection_rate": best.detection_rate,
            "latency_ms": best.mean_latency_ms,
        }

    # Overall rankings
    by_rate = sorted(all_metrics, key=lambda x: x.detection_rate, reverse=True)
    by_latency = sorted(all_metrics, key=lambda x: x.mean_latency_ms)

    # Find best for each size category
    size_analysis = {}
    if all_metrics:
        size_analysis = {
            "small_best": max(all_metrics, key=lambda x: x.detection_rate_small).method_name,
            "medium_best": max(
                all_metrics, key=lambda x: x.detection_rate_medium
            ).method_name,
            "large_best": max(all_metrics, key=lambda x: x.detection_rate_large).method_name,
        }

    # Summary statistics
    summary = {
        "total_methods": len(set(m.method_name for m in all_metrics)),
        "total_datasets": len(by_dataset),
        "best_overall": by_rate[0].method_name if by_rate else None,
        "best_overall_rate": by_rate[0].detection_rate if by_rate else 0.0,
        "fastest_method": by_latency[0].method_name if by_latency else None,
        "fastest_latency_ms": by_latency[0].mean_latency_ms if by_latency else 0.0,
    }

    return {
        "summary": summary,
        "best_per_dataset": best_per_dataset,
        "ranking_by_detection_rate": [
            {
                "method": m.method_name,
                "dataset": m.dataset_name,
                "rate": m.detection_rate,
            }
            for m in by_rate
        ],
        "ranking_by_latency": [
            {
                "method": m.method_name,
                "dataset": m.dataset_name,
                "latency_ms": m.mean_latency_ms,
            }
            for m in by_latency
        ],
        "size_analysis": size_analysis,
    }
