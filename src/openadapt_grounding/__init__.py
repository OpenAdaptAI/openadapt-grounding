"""OpenAdapt Grounding: Robust UI element localization."""

__version__ = "0.1.0"

from openadapt_grounding.builder import Registry, RegistryBuilder
from openadapt_grounding.collector import (
    analyze_stability,
    collect_frames,
    collect_live_frames,
)
from openadapt_grounding.locator import ElementLocator
from openadapt_grounding.parsers import OmniParserClient, Parser
from openadapt_grounding.types import Bounds, Element, LocatorResult, RegistryEntry

__all__ = [
    # Types
    "Bounds",
    "Element",
    "LocatorResult",
    "RegistryEntry",
    # Core
    "Registry",
    "RegistryBuilder",
    "ElementLocator",
    # Parsers
    "Parser",
    "OmniParserClient",
    # Collectors
    "collect_frames",
    "collect_live_frames",
    "analyze_stability",
]
