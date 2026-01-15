"""Evaluation framework for comparing UI grounding methods.

This module provides tools for benchmarking OmniParser and UI-TARS
across synthetic and real datasets.

Usage:
    python -m openadapt_grounding.eval generate --type synthetic --count 500
    python -m openadapt_grounding.eval run --method omniparser --dataset synthetic
    python -m openadapt_grounding.eval compare --charts-dir evaluation/charts
"""

from openadapt_grounding.eval.dataset.schema import (
    AnnotatedElement,
    Dataset,
    ElementSize,
    ElementType,
    Sample,
)
from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.metrics.types import ElementResult, MethodMetrics

__all__ = [
    # Dataset
    "AnnotatedElement",
    "Dataset",
    "ElementSize",
    "ElementType",
    "Sample",
    # Methods
    "EvaluationMethod",
    "EvaluationPrediction",
    # Metrics
    "ElementResult",
    "MethodMetrics",
]
