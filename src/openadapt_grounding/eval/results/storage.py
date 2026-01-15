"""Save and load evaluation results."""

import json
from pathlib import Path

from openadapt_grounding.eval.metrics.types import MethodMetrics


def save_results(metrics: MethodMetrics, path: Path) -> None:
    """Save evaluation results to JSON.

    Args:
        metrics: MethodMetrics object to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)


def load_results(path: Path) -> MethodMetrics:
    """Load evaluation results from JSON.

    Args:
        path: Input file path

    Returns:
        MethodMetrics object
    """
    with open(path) as f:
        data = json.load(f)

    return MethodMetrics.from_dict(data)
