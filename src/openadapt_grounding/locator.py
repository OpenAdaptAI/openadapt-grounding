"""Runtime element locator using OCR."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image

from openadapt_grounding.builder import Registry
from openadapt_grounding.types import Bounds, Element, LocatorResult


class ElementLocator:
    """Find registered elements in screenshots using OCR."""

    def __init__(
        self,
        registry: Union[str, Path, Registry],
        fuzzy_match: bool = True,
    ):
        """
        Args:
            registry: Path to registry JSON or Registry object
            fuzzy_match: Allow substring matching for text
        """
        if isinstance(registry, Registry):
            self.registry = registry
        else:
            self.registry = Registry.load(registry)

        self.fuzzy_match = fuzzy_match

    def find(
        self,
        query: str,
        screenshot: Image.Image,
    ) -> LocatorResult:
        """
        Find an element by text query in the screenshot.

        Args:
            query: Text to search for (e.g., "Save", "Submit")
            screenshot: PIL Image of current screen

        Returns:
            LocatorResult with coordinates if found
        """
        # 1. Look up in registry
        entry = self.registry.get_by_text(query)
        if not entry and self.fuzzy_match:
            entry = self.registry.find_similar_text(query)

        if not entry:
            return LocatorResult(
                found=False,
                debug={"reason": "not_in_registry", "query": query},
            )

        # 2. OCR the screenshot
        ocr_results = self._run_ocr(screenshot)

        # 3. Find matching text
        for ocr_elem in ocr_results:
            if self._text_matches(ocr_elem.text, entry.text):
                cx, cy = ocr_elem.center
                return LocatorResult(
                    found=True,
                    x=cx,
                    y=cy,
                    confidence=0.9,
                    matched_entry=entry,
                    debug={"method": "ocr", "ocr_text": ocr_elem.text},
                )

        # 4. Fallback: return registry position if no OCR match
        # This is risky but better than nothing for stable UIs
        cx, cy = entry.center
        return LocatorResult(
            found=True,
            x=cx,
            y=cy,
            confidence=0.5,  # Lower confidence for fallback
            matched_entry=entry,
            debug={"method": "fallback_position", "reason": "no_ocr_match"},
        )

    def find_by_uid(
        self,
        uid: str,
        screenshot: Image.Image,
    ) -> LocatorResult:
        """Find element by registry UID."""
        entry = self.registry.get_by_uid(uid)
        if not entry:
            return LocatorResult(
                found=False,
                debug={"reason": "uid_not_found", "uid": uid},
            )

        if entry.text:
            return self.find(entry.text, screenshot)

        # For non-text elements, just return stored position
        cx, cy = entry.center
        return LocatorResult(
            found=True,
            x=cx,
            y=cy,
            confidence=0.5,
            matched_entry=entry,
            debug={"method": "stored_position"},
        )

    def _run_ocr(self, image: Image.Image) -> List[Element]:
        """Run OCR on image and return detected text elements."""
        try:
            import pytesseract
        except ImportError:
            return []

        # Get OCR data with bounding boxes
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return []

        elements = []
        width, height = image.size

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty or low confidence
            if not text or conf < 30:
                continue

            # Convert to normalized coordinates
            x = data["left"][i] / width
            y = data["top"][i] / height
            w = data["width"][i] / width
            h = data["height"][i] / height

            elements.append(
                Element(
                    bounds=(x, y, w, h),
                    text=text,
                    element_type="text",
                    confidence=conf / 100.0,
                )
            )

        return elements

    def _text_matches(self, ocr_text: Optional[str], registry_text: Optional[str]) -> bool:
        """Check if OCR text matches registry text."""
        if not ocr_text or not registry_text:
            return False

        ocr_lower = ocr_text.lower().strip()
        reg_lower = registry_text.lower().strip()

        # Exact match
        if ocr_lower == reg_lower:
            return True

        # Fuzzy: one contains the other
        if self.fuzzy_match:
            if ocr_lower in reg_lower or reg_lower in ocr_lower:
                return True

        return False
