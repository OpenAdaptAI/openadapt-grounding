"""Basic tests for openadapt-grounding."""

import pytest

from openadapt_grounding import Element, Registry, RegistryBuilder, RegistryEntry


class TestElement:
    def test_center(self):
        elem = Element(bounds=(0.1, 0.2, 0.3, 0.4), text="Test")
        cx, cy = elem.center
        assert cx == pytest.approx(0.25)  # 0.1 + 0.3/2
        assert cy == pytest.approx(0.4)  # 0.2 + 0.4/2

    def test_iou_no_overlap(self):
        e1 = Element(bounds=(0.0, 0.0, 0.1, 0.1))
        e2 = Element(bounds=(0.5, 0.5, 0.1, 0.1))
        assert e1.iou(e2) == 0.0

    def test_iou_full_overlap(self):
        e1 = Element(bounds=(0.1, 0.1, 0.2, 0.2))
        e2 = Element(bounds=(0.1, 0.1, 0.2, 0.2))
        assert e1.iou(e2) == pytest.approx(1.0)

    def test_iou_partial_overlap(self):
        e1 = Element(bounds=(0.0, 0.0, 0.2, 0.2))
        e2 = Element(bounds=(0.1, 0.1, 0.2, 0.2))
        # Intersection: 0.1 * 0.1 = 0.01
        # Union: 0.04 + 0.04 - 0.01 = 0.07
        assert e1.iou(e2) == pytest.approx(0.01 / 0.07, rel=0.01)


class TestRegistryBuilder:
    def test_empty_builder(self):
        builder = RegistryBuilder()
        registry = builder.build()
        assert len(registry) == 0

    def test_single_frame(self):
        builder = RegistryBuilder()
        builder.add_frame([Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Button")])
        registry = builder.build(min_stability=0.5)
        # With only 1 frame, stability = 100%
        assert len(registry) == 1
        assert registry.entries[0].text == "Button"

    def test_stability_filtering(self):
        builder = RegistryBuilder()

        # "Stable" appears in all 3 frames
        # "Unstable" appears in only 1 frame
        builder.add_frame([
            Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Stable"),
            Element(bounds=(0.5, 0.5, 0.1, 0.1), text="Unstable"),
        ])
        builder.add_frame([
            Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Stable"),
        ])
        builder.add_frame([
            Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Stable"),
        ])

        registry = builder.build(min_stability=0.5)

        # Only "Stable" should survive (3/3 = 100%)
        # "Unstable" (1/3 = 33%) should be filtered
        assert len(registry) == 1
        assert registry.entries[0].text == "Stable"

    def test_text_clustering(self):
        builder = RegistryBuilder()

        # Same text, slightly different positions
        builder.add_frame([Element(bounds=(0.10, 0.10, 0.1, 0.1), text="Save")])
        builder.add_frame([Element(bounds=(0.11, 0.11, 0.1, 0.1), text="Save")])
        builder.add_frame([Element(bounds=(0.09, 0.09, 0.1, 0.1), text="Save")])

        registry = builder.build(min_stability=0.5)

        # Should cluster into single entry
        assert len(registry) == 1
        entry = registry.entries[0]
        assert entry.text == "Save"
        assert entry.detection_count == 3


class TestRegistry:
    def test_lookup_by_text(self):
        entries = [
            RegistryEntry(
                uid="1",
                text="Login",
                bounds=(0.1, 0.1, 0.1, 0.1),
                element_type="button",
                detection_count=10,
                total_frames=10,
            ),
            RegistryEntry(
                uid="2",
                text="Cancel",
                bounds=(0.2, 0.2, 0.1, 0.1),
                element_type="button",
                detection_count=10,
                total_frames=10,
            ),
        ]
        registry = Registry(entries)

        # Case-insensitive lookup
        assert registry.get_by_text("Login") is not None
        assert registry.get_by_text("login") is not None
        assert registry.get_by_text("LOGIN") is not None
        assert registry.get_by_text("NotFound") is None

    def test_similar_text(self):
        entries = [
            RegistryEntry(
                uid="1",
                text="Forgot Password?",
                bounds=(0.1, 0.1, 0.1, 0.1),
                element_type="link",
                detection_count=10,
                total_frames=10,
            ),
        ]
        registry = Registry(entries)

        # Should find "Forgot Password?" when searching for "Forgot"
        assert registry.find_similar_text("Forgot") is not None
        assert registry.find_similar_text("Password") is not None
        assert registry.find_similar_text("Other") is None

    def test_save_load(self, tmp_path):
        entries = [
            RegistryEntry(
                uid="test",
                text="Button",
                bounds=(0.1, 0.2, 0.3, 0.4),
                element_type="button",
                detection_count=5,
                total_frames=10,
            ),
        ]
        registry = Registry(entries)

        path = tmp_path / "registry.json"
        registry.save(path)

        loaded = Registry.load(path)
        assert len(loaded) == 1
        assert loaded.entries[0].uid == "test"
        assert loaded.entries[0].text == "Button"
        assert loaded.entries[0].bounds == (0.1, 0.2, 0.3, 0.4)


class TestDemo:
    def test_demo_runs(self, tmp_path):
        """Smoke test that demo runs without errors."""
        from openadapt_grounding.demo import run_demo

        results = run_demo(output_dir=str(tmp_path))

        assert results["registry_size"] > 0
        assert results["raw_metrics"]["avg_detection_rate"] < 1.0  # Has dropout
        # Stabilized rate should be >= raw (filtering helps)
        assert results["stable_metrics"]["avg_detection_rate"] >= results["raw_metrics"]["avg_detection_rate"]
        assert (tmp_path / "registry.json").exists()
        assert (tmp_path / "base_ui.png").exists()


class TestLocatorOCRFailure:
    """`find()` must not present a stale coordinate as an OCR-checked result.

    Regression for the bug where `_run_ocr` returned `[]` both when OCR ran and
    matched nothing and when OCR could not run at all. On a machine without
    tesseract, `find()` returned `found=True` at the coordinate recorded at
    build time, tagged `reason="no_ocr_match"` -- i.e. it claimed OCR had
    confirmed there was no match. Callers had no way to tell the difference, so
    an unchecked click landed on whatever now occupies the old position.
    """

    @staticmethod
    def _locator():
        from openadapt_grounding.locator import ElementLocator

        registry = Registry(
            [
                RegistryEntry(
                    uid="save",
                    text="Save",
                    bounds=(0.1, 0.2, 0.3, 0.4),
                    element_type="button",
                    detection_count=10,
                    total_frames=10,
                )
            ]
        )
        return ElementLocator(registry)

    @staticmethod
    def _screenshot():
        from PIL import Image

        return Image.new("RGB", (64, 64), "white")

    def test_ocr_failure_is_reported_as_unavailable(self, monkeypatch):
        locator = self._locator()
        monkeypatch.setattr(
            locator,
            "_run_ocr",
            lambda image: ([], "TesseractNotFoundError: tesseract is not installed"),
        )

        result = locator.find("Save", self._screenshot())

        assert result.debug["method"] == "fallback_position"
        assert result.debug["reason"] == "ocr_unavailable"
        assert "TesseractNotFoundError" in result.debug["ocr_error"]

    def test_ocr_ran_but_matched_nothing_is_distinct(self, monkeypatch):
        locator = self._locator()
        monkeypatch.setattr(locator, "_run_ocr", lambda image: ([], None))

        result = locator.find("Save", self._screenshot())

        assert result.debug["method"] == "fallback_position"
        assert result.debug["reason"] == "no_ocr_match"
        assert "ocr_error" not in result.debug

    def test_missing_pytesseract_returns_a_reason(self, monkeypatch):
        """The import probe must report why, not silently yield no elements."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("No module named 'pytesseract'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        elements, error = self._locator()._run_ocr(self._screenshot())

        assert elements == []
        assert error is not None
        assert "pytesseract" in error


class TestProviderModelAllowList:
    """`SUPPORTED_MODELS` is shared by every instance, so it must be immutable.

    Regression for `list_all_models()` handing back the providers' own list
    objects: a caller that sorted or filtered the result in place permanently
    changed what `validate_model()` accepted process-wide.
    """

    def test_supported_models_is_immutable(self):
        from openadapt_grounding.providers import AnthropicProvider

        with pytest.raises(AttributeError):
            AnthropicProvider.SUPPORTED_MODELS.append("not-a-real-model")

    def test_list_all_models_returns_copies(self):
        from openadapt_grounding.providers import AnthropicProvider, list_all_models

        before = AnthropicProvider().get_supported_models()

        models = list_all_models()
        models["anthropic"].append("not-a-real-model")
        models["anthropic"].sort()

        assert AnthropicProvider().get_supported_models() == before
        assert not AnthropicProvider().is_model_supported("not-a-real-model")

    def test_get_supported_models_returns_a_copy(self):
        from openadapt_grounding.providers import OpenAIProvider

        provider = OpenAIProvider()
        provider.get_supported_models().clear()

        assert provider.get_supported_models()


class TestFixedCroppingEmptySizes:
    """An explicit empty `crop_sizes` means "full image only", not "defaults".

    Regression for `crop_sizes or [200, 300, 500]`, which substituted the three
    default sizes for a run that had deliberately disabled fixed crops -- so the
    results reported crops the operator had switched off.
    """

    def test_empty_crop_sizes_is_respected(self):
        from openadapt_grounding.eval.methods.cropping import FixedCropping

        assert FixedCropping(crop_sizes=[]).crop_sizes == []

    def test_none_crop_sizes_uses_defaults(self):
        from openadapt_grounding.eval.methods.cropping import FixedCropping

        assert FixedCropping().crop_sizes == [200, 300, 500]


class TestEvalMethodBackendErrors:
    """A backend outage must not be recorded as a legitimate 0% score.

    Regression for `except Exception: continue` in both evaluation methods.
    An unreachable or erroring OmniParser/UI-TARS server produced
    `EvaluationPrediction(found=False)` for every element, indistinguishable
    from a genuine miss -- so a comparison chart published "0% detection rate"
    for a method whose server had simply been down.
    """

    @staticmethod
    def _target():
        from openadapt_grounding.eval.dataset.schema import AnnotatedElement

        return AnnotatedElement(id="e1", bbox=(0.1, 0.1, 0.2, 0.2), text="Save")

    @staticmethod
    def _screenshot():
        from PIL import Image

        return Image.new("RGB", (64, 64), "white")

    def test_omniparser_records_backend_failure(self):
        from openadapt_grounding.eval.methods.omniparser import OmniParserMethod

        class BrokenClient:
            def parse(self, image):
                raise ConnectionError("connection refused")

        prediction = OmniParserMethod(BrokenClient()).evaluate_element(
            self._screenshot(), self._target()
        )

        assert prediction.found is False
        assert prediction.method_info["region_errors"]
        assert "ConnectionError" in prediction.method_info["region_errors"][0]

    def test_uitars_records_backend_failure(self):
        from openadapt_grounding.eval.methods.uitars import UITarsMethod

        class BrokenClient:
            def ground(self, image, instruction):
                raise ConnectionError("connection refused")

        prediction = UITarsMethod(BrokenClient()).evaluate_element(
            self._screenshot(), self._target()
        )

        assert prediction.found is False
        assert prediction.method_info["region_errors"]
        assert "ConnectionError" in prediction.method_info["region_errors"][0]

    def test_no_region_errors_key_on_a_clean_miss(self):
        from openadapt_grounding.eval.methods.omniparser import OmniParserMethod

        class EmptyClient:
            def parse(self, image):
                return []

        prediction = OmniParserMethod(EmptyClient()).evaluate_element(
            self._screenshot(), self._target()
        )

        assert prediction.found is False
        assert "region_errors" not in prediction.method_info
