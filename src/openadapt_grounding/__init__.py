"""OpenAdapt Grounding: Robust UI element localization."""

__version__ = "0.1.0"

from openadapt_grounding.builder import Registry, RegistryBuilder
from openadapt_grounding.locator import ElementLocator
from openadapt_grounding.types import Bounds, Element, LocatorResult, RegistryEntry

__all__ = [
    "Bounds",
    "Element",
    "ElementLocator",
    "LocatorResult",
    "Registry",
    "RegistryBuilder",
    "RegistryEntry",
]
