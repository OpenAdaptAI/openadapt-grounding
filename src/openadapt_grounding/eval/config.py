"""Configuration settings for evaluation."""

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    raise ImportError(
        "pydantic-settings not installed. Run: uv pip install openadapt-grounding[eval]"
    )


class EvalSettings(BaseSettings):
    """Configuration settings for evaluation."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        extra="ignore",
    )

    # Server URLs
    OMNIPARSER_URL: str = "http://localhost:8000"
    UITARS_URL: str = "http://localhost:8001/v1"

    # Default paths
    DATASETS_DIR: str = "evaluation/datasets"
    RESULTS_DIR: str = "evaluation/results"
    CHARTS_DIR: str = "evaluation/charts"

    # Evaluation settings
    DEFAULT_IOU_THRESHOLD: float = 0.3
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5

    # Synthetic generation
    SYNTHETIC_WIDTH: int = 1920
    SYNTHETIC_HEIGHT: int = 1080
    SYNTHETIC_SEED: int = 42


def get_settings() -> EvalSettings:
    """Get evaluation settings, creating with defaults if pydantic unavailable."""
    try:
        return EvalSettings()
    except Exception:
        # Return object with defaults if settings can't be loaded
        class DefaultSettings:
            OMNIPARSER_URL = "http://localhost:8000"
            UITARS_URL = "http://localhost:8001/v1"
            DATASETS_DIR = "evaluation/datasets"
            RESULTS_DIR = "evaluation/results"
            CHARTS_DIR = "evaluation/charts"
            DEFAULT_IOU_THRESHOLD = 0.3
            DEFAULT_CONFIDENCE_THRESHOLD = 0.5
            SYNTHETIC_WIDTH = 1920
            SYNTHETIC_HEIGHT = 1080
            SYNTHETIC_SEED = 42

        return DefaultSettings()  # type: ignore
