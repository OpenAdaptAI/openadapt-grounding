"""Parsers for extracting UI elements from screenshots."""

from openadapt_grounding.parsers.base import Parser
from openadapt_grounding.parsers.omniparser import OmniParserClient
from openadapt_grounding.parsers.uitars import GroundingResult, UITarsClient

__all__ = ["Parser", "OmniParserClient", "UITarsClient", "GroundingResult"]
