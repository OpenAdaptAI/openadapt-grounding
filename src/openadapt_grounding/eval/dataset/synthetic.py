"""Synthetic UI dataset generator for evaluation."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from openadapt_grounding.eval.dataset.schema import (
    AnnotatedElement,
    Dataset,
    ElementType,
    Sample,
)

# Common UI texts
BUTTON_TEXTS = [
    "Submit",
    "Cancel",
    "OK",
    "Save",
    "Delete",
    "Close",
    "Open",
    "New",
    "Edit",
    "Copy",
    "Paste",
    "Undo",
    "Redo",
    "Apply",
    "Next",
    "Back",
    "Continue",
    "Finish",
    "Start",
    "Stop",
    "Pause",
    "Resume",
    "Refresh",
    "Update",
    "Download",
    "Upload",
    "Send",
    "Share",
    "Print",
    "Export",
]

LINK_TEXTS = [
    "Learn more",
    "Sign up",
    "Log in",
    "Forgot password?",
    "Help",
    "Contact us",
    "Privacy Policy",
    "Terms of Service",
    "View details",
    "See all",
    "Read more",
    "Click here",
    "Get started",
    "Try it free",
    "Sign out",
]

ICON_LABELS = [
    "settings",
    "search",
    "home",
    "menu",
    "close",
    "add",
    "remove",
    "share",
    "download",
    "upload",
    "refresh",
    "star",
    "heart",
    "bookmark",
    "edit",
    "delete",
    "copy",
    "paste",
    "undo",
    "redo",
]

TEXT_LABELS = [
    "Username",
    "Password",
    "Email",
    "Phone",
    "Address",
    "Name",
    "Title",
    "Description",
    "Message",
    "Comment",
]

# Color palettes
LIGHT_BACKGROUNDS = ["#ffffff", "#f5f5f5", "#fafafa", "#f0f0f0", "#e8e8e8"]
DARK_BACKGROUNDS = ["#1e1e1e", "#2d2d2d", "#333333", "#424242", "#212121"]
BUTTON_COLORS = [
    "#2196f3",
    "#4caf50",
    "#f44336",
    "#ff9800",
    "#9c27b0",
    "#607d8b",
    "#00bcd4",
    "#795548",
    "#3f51b5",
    "#009688",
]


class SyntheticGenerator:
    """Generate synthetic UI screenshots with ground truth annotations."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        seed: Optional[int] = None,
    ):
        """Initialize the generator.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)

        self._font = self._load_font()
        self._font_small = self._load_font(size=12)

    def _load_font(self, size: int = 14) -> Optional[ImageFont.FreeTypeFont]:
        """Try to load a TrueType font."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue

        return None

    def generate_sample(
        self,
        sample_id: str,
        difficulty: str = "medium",
        theme: str = "light",
    ) -> Tuple[Image.Image, Sample]:
        """Generate a single synthetic sample.

        Args:
            sample_id: Unique identifier for this sample
            difficulty: "easy", "medium", "hard" - affects element density and size
            theme: "light" or "dark"

        Returns:
            Tuple of (PIL Image, Sample annotation)
        """
        # Create background
        bg_color = random.choice(
            LIGHT_BACKGROUNDS if theme == "light" else DARK_BACKGROUNDS
        )
        img = Image.new("RGB", (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(img)

        # Determine element counts based on difficulty
        element_params = {
            "easy": {"count": (3, 8), "min_size": 80, "max_size": 200},
            "medium": {"count": (8, 20), "min_size": 40, "max_size": 150},
            "hard": {"count": (20, 50), "min_size": 16, "max_size": 80},
        }
        params = element_params.get(difficulty, element_params["medium"])
        num_elements = random.randint(*params["count"])

        elements: List[AnnotatedElement] = []
        occupied_regions: List[Tuple[int, int, int, int]] = []

        for i in range(num_elements):
            elem = self._generate_element(
                draw=draw,
                elem_id=f"{sample_id}_elem_{i:03d}",
                min_size=params["min_size"],
                max_size=params["max_size"],
                theme=theme,
                occupied=occupied_regions,
            )
            if elem:
                elements.append(elem)
                # Mark region as occupied (in pixels)
                x, y, w, h = elem.bbox
                occupied_regions.append(
                    (
                        int(x * self.width),
                        int(y * self.height),
                        int(w * self.width),
                        int(h * self.height),
                    )
                )

        sample = Sample(
            id=sample_id,
            image_path=f"samples/{sample_id}.png",
            width=self.width,
            height=self.height,
            elements=elements,
            metadata={
                "difficulty": difficulty,
                "theme": theme,
                "num_elements": len(elements),
            },
        )

        return img, sample

    def _generate_element(
        self,
        draw: ImageDraw.ImageDraw,
        elem_id: str,
        min_size: int,
        max_size: int,
        theme: str,
        occupied: List[Tuple[int, int, int, int]],
    ) -> Optional[AnnotatedElement]:
        """Generate a single UI element."""
        # Choose element type with weighted distribution
        elem_type = random.choices(
            [
                ElementType.BUTTON,
                ElementType.ICON,
                ElementType.TEXT,
                ElementType.LINK,
                ElementType.TEXT_FIELD,
            ],
            weights=[0.35, 0.20, 0.20, 0.15, 0.10],
        )[0]

        # Try to find non-overlapping position
        for _ in range(20):  # Max attempts
            w = random.randint(min_size, max_size)
            h = random.randint(min_size // 3, min_size)

            # Adjust dimensions based on element type
            if elem_type == ElementType.ICON:
                h = w  # Icons are square
            elif elem_type == ElementType.TEXT_FIELD:
                w = max(w, 150)
                h = max(30, min_size // 2)

            x = random.randint(20, self.width - w - 20)
            y = random.randint(20, self.height - h - 20)

            # Check for overlap
            if not self._overlaps(x, y, w, h, occupied):
                break
        else:
            return None  # Could not find non-overlapping position

        # Draw element based on type
        text: Optional[str] = None
        instruction: Optional[str] = None

        if elem_type == ElementType.BUTTON:
            text = random.choice(BUTTON_TEXTS)
            color = random.choice(BUTTON_COLORS)
            draw.rounded_rectangle([x, y, x + w, y + h], radius=4, fill=color)
            text_color = "white"
            self._draw_centered_text(draw, x, y, w, h, text, text_color)
            instruction = f"Click on the '{text}' button"

        elif elem_type == ElementType.LINK:
            text = random.choice(LINK_TEXTS)
            text_color = "#1976d2" if theme == "light" else "#64b5f6"
            self._draw_centered_text(draw, x, y, w, h, text, text_color, underline=True)
            instruction = f"Click on the '{text}' link"

        elif elem_type == ElementType.ICON:
            label = random.choice(ICON_LABELS)
            icon_color = "#757575" if theme == "light" else "#bdbdbd"
            # Draw simple icon placeholder (circle with initial)
            draw.ellipse([x + 2, y + 2, x + w - 2, y + h - 2], fill=icon_color)
            initial = label[0].upper()
            self._draw_centered_text(draw, x, y, w, h, initial, "white")
            text = label
            instruction = f"Click on the {label} icon"

        elif elem_type == ElementType.TEXT:
            text = random.choice(TEXT_LABELS)
            text_color = "#333333" if theme == "light" else "#eeeeee"
            self._draw_centered_text(draw, x, y, w, h, text, text_color)
            instruction = f"Click on the text '{text}'"

        elif elem_type == ElementType.TEXT_FIELD:
            label = random.choice(TEXT_LABELS)
            border_color = "#cccccc" if theme == "light" else "#555555"
            fill_color = "#ffffff" if theme == "light" else "#2d2d2d"
            text_color = "#999999" if theme == "light" else "#666666"

            # Draw text field with border
            draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=border_color)
            placeholder = f"Enter {label.lower()}..."
            self._draw_text_left(draw, x + 8, y, h, placeholder, text_color)
            text = label
            instruction = f"Click on the {label} text field"

        # Normalize coordinates
        norm_x = x / self.width
        norm_y = y / self.height
        norm_w = w / self.width
        norm_h = h / self.height

        return AnnotatedElement(
            id=elem_id,
            bbox=(norm_x, norm_y, norm_w, norm_h),
            text=text,
            element_type=elem_type,
            instruction=instruction,
        )

    def _overlaps(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        occupied: List[Tuple[int, int, int, int]],
        margin: int = 10,
    ) -> bool:
        """Check if rectangle overlaps with any occupied region."""
        for ox, oy, ow, oh in occupied:
            if not (
                x + w + margin < ox
                or x > ox + ow + margin
                or y + h + margin < oy
                or y > oy + oh + margin
            ):
                return True
        return False

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        color: str,
        underline: bool = False,
    ) -> None:
        """Draw text centered in a rectangle."""
        font = self._font

        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(text) * 8
            text_h = 12

        tx = x + (w - text_w) // 2
        ty = y + (h - text_h) // 2

        draw.text((tx, ty), text, fill=color, font=font)

        if underline and font:
            draw.line([(tx, ty + text_h + 2), (tx + text_w, ty + text_h + 2)], fill=color)

    def _draw_text_left(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        h: int,
        text: str,
        color: str,
    ) -> None:
        """Draw text left-aligned and vertically centered."""
        font = self._font_small or self._font

        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_h = bbox[3] - bbox[1]
        else:
            text_h = 12

        ty = y + (h - text_h) // 2
        draw.text((x, ty), text, fill=color, font=font)

    def generate_dataset(
        self,
        output_dir: Path,
        num_samples: int = 500,
        distribution: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Dataset:
        """Generate a complete synthetic dataset.

        Args:
            output_dir: Directory to save images and annotations
            num_samples: Total number of samples
            distribution: Dict specifying sample distribution by (difficulty, theme)

        Returns:
            Dataset object with all samples
        """
        output_dir = Path(output_dir)
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        if distribution is None:
            distribution = {
                ("easy", "light"): 0.15,
                ("easy", "dark"): 0.05,
                ("medium", "light"): 0.30,
                ("medium", "dark"): 0.15,
                ("hard", "light"): 0.20,
                ("hard", "dark"): 0.15,
            }

        samples: List[Sample] = []
        sample_idx = 0

        for (difficulty, theme), fraction in distribution.items():
            count = int(num_samples * fraction)
            for _ in range(count):
                sample_id = f"synthetic_{sample_idx:04d}"
                img, sample = self.generate_sample(sample_id, difficulty, theme)

                # Save image
                img.save(samples_dir / f"{sample_id}.png")
                samples.append(sample)
                sample_idx += 1

        dataset = Dataset(
            name="synthetic",
            version="1.0",
            samples=samples,
            metadata={
                "num_samples": len(samples),
                "distribution": {f"{k[0]}_{k[1]}": v for k, v in distribution.items()},
                "width": self.width,
                "height": self.height,
            },
        )

        # Save annotations
        dataset.save(output_dir / "annotations.json")

        return dataset
