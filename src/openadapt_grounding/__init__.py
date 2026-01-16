"""OpenAdapt Grounding: Robust UI element localization."""

__version__ = "0.1.0"

from openadapt_grounding.builder import Registry, RegistryBuilder
from openadapt_grounding.collector import (
    analyze_stability,
    collect_frames,
    collect_live_frames,
)
from openadapt_grounding.locator import ElementLocator
from openadapt_grounding.parsers import (
    GroundingResult,
    OmniParserClient,
    Parser,
    UITarsClient,
)
from openadapt_grounding.types import Bounds, Element, LocatorResult, RegistryEntry

# Lazy import for providers (requires optional dependencies)
def get_provider(name: str):
    """Get a VLM API provider by name.

    This is a convenience re-export. For full functionality, import from
    openadapt_grounding.providers directly.

    Args:
        name: Provider name ('anthropic', 'openai', 'google') or alias
              ('claude', 'gpt', 'gemini').

    Returns:
        Provider instance.

    Note:
        Requires optional dependencies. Install with:
        - pip install openadapt-grounding[providers-anthropic]
        - pip install openadapt-grounding[providers-openai]
        - pip install openadapt-grounding[providers-google]
        - pip install openadapt-grounding[providers]  # all providers
    """
    from openadapt_grounding.providers import get_provider as _get_provider
    return _get_provider(name)


__all__ = [
    # Types
    "Bounds",
    "Element",
    "GroundingResult",
    "LocatorResult",
    "RegistryEntry",
    # Core
    "Registry",
    "RegistryBuilder",
    "ElementLocator",
    # Parsers
    "Parser",
    "OmniParserClient",
    "UITarsClient",
    # Collectors
    "collect_frames",
    "collect_live_frames",
    "analyze_stability",
    # Providers
    "get_provider",
]
