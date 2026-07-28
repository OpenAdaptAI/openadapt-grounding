"""Runtime element locator using OCR."""

import logging
from pathlib import Path

from PIL import Image

from openadapt_grounding.builder import Registry
from openadapt_grounding.types import Element, LocatorResult

logger = logging.getLogger(__name__)


class ElementLocator:
    """Find registered elements in screenshots using OCR."""

    def __init__(
        self,
        registry: str | Path | Registry,
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
        ocr_results, ocr_error = self._run_ocr(screenshot)

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
        # The reason distinguishes "OCR ran and disagreed" from "OCR never
        # ran". Both used to report `reason="no_ocr_match"`, so a machine with
        # no tesseract installed got `found=True` at the coordinate recorded at
        # build time, labelled as if OCR had confirmed no match -- an unchecked
        # stale click that the caller had no way to detect.
        debug: dict[str, object] = {"method": "fallback_position"}
        if ocr_error is not None:
            debug["reason"] = "ocr_unavailable"
            debug["ocr_error"] = ocr_error
        else:
            debug["reason"] = "no_ocr_match"
        return LocatorResult(
            found=True,
            x=cx,
            y=cy,
            confidence=0.5,  # Lower confidence for fallback
            matched_entry=entry,
            debug=debug,
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

    def _run_ocr(self, image: Image.Image) -> tuple[list[Element], str | None]:
        """Run OCR on image and return detected text elements.

        Returns:
            ``(elements, error)``. ``error`` is None when OCR actually ran; it
            is a short description when OCR could not run at all, in which case
            ``elements`` is empty for a reason that has nothing to do with the
            screenshot.

        An empty list used to be returned for both "OCR ran and saw no text"
        and "OCR never ran", which let `find()` report a stale registry
        coordinate as `found=True, reason="no_ocr_match"`. See `find()`.
        """
        try:
            import pytesseract
        except ImportError as exc:
            logger.warning("OCR unavailable, pytesseract is not installed: %s", exc)
            return [], f"pytesseract not installed: {exc}"

        # Get OCR data with bounding boxes. The catch stays broad because
        # pytesseract raises TesseractNotFoundError (an OSError), TesseractError
        # and PIL decoding errors with no shared base -- but the reason is
        # returned now instead of discarded.
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as exc:
            logger.warning("OCR failed: %s: %s", type(exc).__name__, exc)
            return [], f"{type(exc).__name__}: {exc}"

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

        return elements, None

    def _text_matches(self, ocr_text: str | None, registry_text: str | None) -> bool:
        """Check if OCR text matches registry text."""
        if not ocr_text or not registry_text:
            return False

        ocr_lower = ocr_text.lower().strip()
        reg_lower = registry_text.lower().strip()

        # Exact match
        if ocr_lower == reg_lower:
            return True

        # Fuzzy: one contains the other
        return self.fuzzy_match and (
            ocr_lower in reg_lower or reg_lower in ocr_lower
        )
