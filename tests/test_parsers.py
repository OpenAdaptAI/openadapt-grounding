"""Tests for parser implementations."""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from openadapt_grounding import Element, OmniParserClient, Parser
from openadapt_grounding.collector import analyze_stability, collect_frames


class TestParserProtocol:
    """Test that Parser protocol works correctly."""

    def test_omniparser_client_is_parser(self):
        """OmniParserClient should implement Parser protocol."""
        client = OmniParserClient("http://localhost:8000")
        assert isinstance(client, Parser)

    def test_custom_parser_protocol(self):
        """Custom parsers can implement the protocol."""

        class MockParser:
            def parse(self, image):
                return [Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Mock")]

            def is_available(self):
                return True

        parser = MockParser()
        assert isinstance(parser, Parser)


class TestOmniParserClient:
    """Tests for OmniParserClient."""

    def test_init(self):
        """Test client initialization."""
        client = OmniParserClient("http://example.com:8000/")
        assert client.server_url == "http://example.com:8000"  # Trailing slash removed
        assert client.timeout == 60.0

    def test_init_custom_timeout(self):
        """Test custom timeout."""
        client = OmniParserClient("http://localhost:8000", timeout=120.0)
        assert client.timeout == 120.0

    @patch("openadapt_grounding.parsers.omniparser.requests.get")
    def test_is_available_success(self, mock_get):
        """Test is_available when server responds."""
        mock_get.return_value.status_code = 200
        client = OmniParserClient("http://localhost:8000")
        assert client.is_available() is True
        mock_get.assert_called_once_with("http://localhost:8000/probe/", timeout=5.0)

    @patch("openadapt_grounding.parsers.omniparser.requests.get")
    def test_is_available_failure(self, mock_get):
        """Test is_available when server is down."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("Connection refused")
        client = OmniParserClient("http://localhost:8000")
        assert client.is_available() is False

    @patch("openadapt_grounding.parsers.omniparser.requests.post")
    def test_parse_success(self, mock_post):
        """Test successful parse."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parsed_content_list": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "content": "Login"},
                {"bbox": [0.5, 0.5, 0.7, 0.6], "content": "Cancel"},
            ],
            "latency": 1.5,
        }
        mock_post.return_value = mock_response

        client = OmniParserClient("http://localhost:8000")
        image = Image.new("RGB", (100, 100), color="white")
        elements = client.parse(image)

        assert len(elements) == 2
        # Check first element - bounds converted from [x1,y1,x2,y2] to (x,y,w,h)
        assert elements[0].text == "Login"
        x, y, w, h = elements[0].bounds
        assert (x, y) == pytest.approx((0.1, 0.2))
        assert (w, h) == pytest.approx((0.2, 0.2))  # w=0.3-0.1, h=0.4-0.2

        # Check second element
        assert elements[1].text == "Cancel"
        x, y, w, h = elements[1].bounds
        assert (x, y) == pytest.approx((0.5, 0.5))
        assert (w, h) == pytest.approx((0.2, 0.1))  # w=0.7-0.5, h=0.6-0.5

    @patch("openadapt_grounding.parsers.omniparser.requests.post")
    def test_parse_with_text_box_prefix(self, mock_post):
        """Test parsing elements with 'Text Box ID' prefix."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parsed_content_list": [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "content": "Text Box ID 0: Login"},
                {"bbox": [0.3, 0.3, 0.4, 0.4], "content": "Icon Box ID 1: search icon"},
            ],
        }
        mock_post.return_value = mock_response

        client = OmniParserClient("http://localhost:8000")
        image = Image.new("RGB", (100, 100))
        elements = client.parse(image)

        assert len(elements) == 2
        assert elements[0].text == "Login"
        assert elements[0].element_type == "text"
        assert elements[1].text == "search icon"
        assert elements[1].element_type == "icon"

    @patch("openadapt_grounding.parsers.omniparser.requests.post")
    def test_parse_filters_invalid_bounds(self, mock_post):
        """Test that invalid bounds are filtered out."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parsed_content_list": [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "content": "Valid"},
                {"bbox": [1.5, 0.1, 0.2, 0.2], "content": "Out of bounds"},  # x1 > 1
                {"bbox": [0.1, 0.1, 0.05, 0.2], "content": "Negative width"},  # x2 < x1
                {"bbox": [], "content": "Empty bbox"},
                {"content": "No bbox"},
            ],
        }
        mock_post.return_value = mock_response

        client = OmniParserClient("http://localhost:8000")
        image = Image.new("RGB", (100, 100))
        elements = client.parse(image)

        assert len(elements) == 1
        assert elements[0].text == "Valid"

    @patch("openadapt_grounding.parsers.omniparser.requests.post")
    def test_parse_connection_error(self, mock_post):
        """Test connection error handling."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = OmniParserClient("http://localhost:8000")
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ConnectionError) as exc_info:
            client.parse(image)

        assert "Could not connect" in str(exc_info.value)


class TestCollectFrames:
    """Tests for frame collection utilities."""

    def test_collect_frames_basic(self):
        """Test basic frame collection."""

        class MockParser:
            def __init__(self):
                self.call_count = 0

            def parse(self, image):
                self.call_count += 1
                # Return slightly different results each time
                if self.call_count % 2 == 0:
                    return [
                        Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Stable"),
                    ]
                else:
                    return [
                        Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Stable"),
                        Element(bounds=(0.5, 0.5, 0.1, 0.1), text="Unstable"),
                    ]

            def is_available(self):
                return True

        parser = MockParser()
        image = Image.new("RGB", (100, 100))
        registry = collect_frames(parser, image, num_frames=4, min_stability=0.5)

        assert parser.call_count == 4
        # "Stable" appears in 4/4 frames (100%), should be kept
        # "Unstable" appears in 2/4 frames (50%), should be kept at 0.5 threshold
        assert len(registry) >= 1

    def test_collect_frames_with_callback(self):
        """Test frame collection with callback."""
        frames_collected = []

        class MockParser:
            def parse(self, image):
                return [Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Test")]

            def is_available(self):
                return True

        def on_frame(idx, elements):
            frames_collected.append((idx, len(elements)))

        parser = MockParser()
        image = Image.new("RGB", (100, 100))
        collect_frames(parser, image, num_frames=3, on_frame=on_frame)

        assert len(frames_collected) == 3
        assert frames_collected[0] == (0, 1)
        assert frames_collected[1] == (1, 1)
        assert frames_collected[2] == (2, 1)


class TestAnalyzeStability:
    """Tests for stability analysis."""

    def test_analyze_stability(self):
        """Test stability analysis output."""

        class MockParser:
            def __init__(self):
                self.call_count = 0

            def parse(self, image):
                self.call_count += 1
                elements = [Element(bounds=(0.1, 0.1, 0.1, 0.1), text="Always")]
                if self.call_count <= 5:
                    elements.append(Element(bounds=(0.5, 0.5, 0.1, 0.1), text="Sometimes"))
                return elements

            def is_available(self):
                return True

        parser = MockParser()
        image = Image.new("RGB", (100, 100))
        stats = analyze_stability(parser, image, num_frames=10)

        assert stats["num_frames"] == 10
        assert stats["unique_elements"] == 2

        # Find element stats
        always_elem = next(e for e in stats["elements"] if e["text"] == "always")
        sometimes_elem = next(e for e in stats["elements"] if e["text"] == "sometimes")

        assert always_elem["stability"] == 1.0  # 10/10
        assert sometimes_elem["stability"] == 0.5  # 5/10
