"""Base protocol for UI element parsers."""

from typing import List, Protocol, runtime_checkable

from PIL import Image

from openadapt_grounding.types import Element


@runtime_checkable
class Parser(Protocol):
    """Protocol for UI element parsers.

    Implementations should extract UI elements from screenshots.
    This allows swapping between different backends (OmniParser, etc.)
    """

    def parse(self, image: Image.Image) -> List[Element]:
        """Parse a screenshot and return detected UI elements.

        Args:
            image: PIL Image of the screenshot

        Returns:
            List of detected Element objects with normalized bounds (0-1)
        """
        ...

    def is_available(self) -> bool:
        """Check if the parser is available and ready.

        Returns:
            True if parser can accept requests
        """
        ...
