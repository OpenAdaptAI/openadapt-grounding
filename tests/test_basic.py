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
