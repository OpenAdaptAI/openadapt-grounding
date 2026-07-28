"""Evaluation methods for different grounding approaches."""

from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.methods.cropping import (
    CroppingStrategy,
    CropRegion,
    FixedCropping,
    NoCropping,
    ScreenSeekeRCropping,
)

__all__ = [
    "CropRegion",
    "CroppingStrategy",
    "EvaluationMethod",
    "EvaluationPrediction",
    "FixedCropping",
    "NoCropping",
    "ScreenSeekeRCropping",
]
