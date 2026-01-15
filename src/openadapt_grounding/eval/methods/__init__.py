"""Evaluation methods for different grounding approaches."""

from openadapt_grounding.eval.methods.base import EvaluationMethod, EvaluationPrediction
from openadapt_grounding.eval.methods.cropping import (
    CropRegion,
    CroppingStrategy,
    FixedCropping,
    NoCropping,
    ScreenSeekeRCropping,
)

__all__ = [
    "EvaluationMethod",
    "EvaluationPrediction",
    "CropRegion",
    "CroppingStrategy",
    "NoCropping",
    "FixedCropping",
    "ScreenSeekeRCropping",
]
