"""Configuration settings for evaluation."""

from pathlib import Path
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    raise ImportError(
        "pydantic-settings not installed. Run: uv pip install openadapt-grounding[eval]"
    )


def _discover_instance_url(project_name: str, port: int) -> str | None:
    """Discover running EC2 instance URL by Name tag.

    Args:
        project_name: Name tag value to search for (e.g., "omniparser", "uitars")
        port: Port the service runs on

    Returns:
        URL string like "http://1.2.3.4:8000" or None if not found
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        ec2 = boto3.resource("ec2")
        instances = ec2.instances.filter(
            Filters=[
                {"Name": "tag:Name", "Values": [project_name]},
                {"Name": "instance-state-name", "Values": ["running"]},
            ]
        )

        for instance in instances:
            if instance.public_ip_address:
                return f"http://{instance.public_ip_address}:{port}"
        return None
    except (ImportError, BotoCoreError, ClientError, NoCredentialsError):
        return None


@lru_cache(maxsize=1)
def _get_omniparser_url() -> str:
    """Get OmniParser URL, auto-detecting from AWS if not set."""
    import os

    # Check env var first
    env_url = os.environ.get("EVAL_OMNIPARSER_URL")
    if env_url:
        return env_url

    # Try AWS discovery
    discovered = _discover_instance_url("omniparser", 8000)
    if discovered:
        return discovered

    return "http://localhost:8000"


@lru_cache(maxsize=1)
def _get_uitars_url() -> str:
    """Get UI-TARS URL, auto-detecting from AWS if not set."""
    import os

    # Check env var first
    env_url = os.environ.get("EVAL_UITARS_URL")
    if env_url:
        return env_url

    # Try AWS discovery
    discovered = _discover_instance_url("uitars", 8001)
    if discovered:
        return f"{discovered}/v1"

    return "http://localhost:8001/v1"


class EvalSettings(BaseSettings):
    """Configuration settings for evaluation.

    Server URLs are auto-detected from AWS EC2 instances by Name tag.
    Set EVAL_OMNIPARSER_URL or EVAL_UITARS_URL environment variables to override.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def OMNIPARSER_URL(self) -> str:
        """OmniParser URL - auto-detected from AWS or env var."""
        return _get_omniparser_url()

    @property
    def UITARS_URL(self) -> str:
        """UI-TARS URL - auto-detected from AWS or env var."""
        return _get_uitars_url()

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
            @property
            def OMNIPARSER_URL(self) -> str:
                return _get_omniparser_url()

            @property
            def UITARS_URL(self) -> str:
                return _get_uitars_url()

            DATASETS_DIR = "evaluation/datasets"
            RESULTS_DIR = "evaluation/results"
            CHARTS_DIR = "evaluation/charts"
            DEFAULT_IOU_THRESHOLD = 0.3
            DEFAULT_CONFIDENCE_THRESHOLD = 0.5
            SYNTHETIC_WIDTH = 1920
            SYNTHETIC_HEIGHT = 1080
            SYNTHETIC_SEED = 42

        return DefaultSettings()  # type: ignore
