"""Tests for VLM API provider implementations.

These tests verify:
1. Provider instantiation and basic functionality
2. Image encoding for each provider
3. Real API calls with actual API keys (integration tests)

Integration tests require API keys in environment:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY

Run with: uv run pytest tests/test_providers.py -v
"""

import os
from pathlib import Path

import pytest
from PIL import Image

from openadapt_grounding.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    get_provider,
    get_provider_for_model,
    list_all_models,
    list_providers,
)
from openadapt_grounding.providers.base import BaseAPIProvider


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def real_image():
    """Load a real UI image from assets."""
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / "base_ui.png"
    if image_path.exists():
        return Image.open(image_path)
    # Fallback to generated image
    return Image.new("RGB", (800, 600), color="white")


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def google_api_key():
    """Get Google API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


# ============================================================================
# Unit Tests - Provider Registry
# ============================================================================


class TestProviderRegistry:
    """Test provider registry and factory functions."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers

    def test_get_provider_by_name(self):
        """Test getting provider by canonical name."""
        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"

    def test_get_provider_by_alias(self):
        """Test getting provider by alias."""
        provider = get_provider("claude")
        assert isinstance(provider, AnthropicProvider)

        provider = get_provider("gpt")
        assert isinstance(provider, OpenAIProvider)

        provider = get_provider("gemini")
        assert isinstance(provider, GoogleProvider)

    def test_get_provider_invalid(self):
        """Test error on invalid provider name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")

    def test_list_all_models(self):
        """Test listing all models by provider."""
        models = list_all_models()
        assert "anthropic" in models
        assert "openai" in models
        assert "google" in models
        assert len(models["anthropic"]) > 0
        assert len(models["openai"]) > 0
        assert len(models["google"]) > 0

    def test_get_provider_for_model(self):
        """Test getting provider for a specific model."""
        provider = get_provider_for_model("claude-opus-4-5-20251101")
        assert provider is not None
        assert provider.name == "anthropic"

        provider = get_provider_for_model("gpt-4o")
        assert provider is not None
        assert provider.name == "openai"

        provider = get_provider_for_model("gemini-2.5-pro")
        assert provider is not None
        assert provider.name == "google"

        provider = get_provider_for_model("unknown-model")
        assert provider is None


# ============================================================================
# Unit Tests - AnthropicProvider
# ============================================================================


class TestAnthropicProvider:
    """Test AnthropicProvider implementation."""

    def test_name(self):
        """Test provider name."""
        provider = AnthropicProvider()
        assert provider.name == "anthropic"

    def test_supported_models(self):
        """Test supported models list."""
        provider = AnthropicProvider()
        models = provider.get_supported_models()
        assert "claude-opus-4-5-20251101" in models
        assert "claude-sonnet-4-5-20250929" in models

    def test_is_model_supported(self):
        """Test model support check."""
        provider = AnthropicProvider()
        assert provider.is_model_supported("claude-opus-4-5-20251101")
        assert not provider.is_model_supported("gpt-4o")

    def test_validate_model_raises(self):
        """Test model validation raises on unsupported model."""
        provider = AnthropicProvider()
        with pytest.raises(ValueError, match="not supported"):
            provider.validate_model("unsupported-model")

    def test_create_client_empty_key(self):
        """Test error on empty API key."""
        provider = AnthropicProvider()
        with pytest.raises(ValueError, match="API key must be provided"):
            provider.create_client("")

        with pytest.raises(ValueError, match="API key must be provided"):
            provider.create_client("   ")

    def test_encode_image(self, test_image):
        """Test image encoding for Anthropic format."""
        provider = AnthropicProvider()
        encoded = provider.encode_image(test_image)

        assert encoded["type"] == "image"
        assert encoded["source"]["type"] == "base64"
        assert encoded["source"]["media_type"] == "image/png"
        assert len(encoded["source"]["data"]) > 0

    def test_encode_image_jpeg(self, test_image):
        """Test image encoding with JPEG media type."""
        provider = AnthropicProvider()
        encoded = provider.encode_image(test_image, media_type="image/jpeg")

        assert encoded["source"]["media_type"] == "image/jpeg"


# ============================================================================
# Unit Tests - OpenAIProvider
# ============================================================================


class TestOpenAIProvider:
    """Test OpenAIProvider implementation."""

    def test_name(self):
        """Test provider name."""
        provider = OpenAIProvider()
        assert provider.name == "openai"

    def test_supported_models(self):
        """Test supported models list."""
        provider = OpenAIProvider()
        models = provider.get_supported_models()
        assert "gpt-4o" in models
        assert "gpt-5.2" in models

    def test_is_model_supported(self):
        """Test model support check."""
        provider = OpenAIProvider()
        assert provider.is_model_supported("gpt-4o")
        assert not provider.is_model_supported("claude-opus-4-5-20251101")

    def test_create_client_empty_key(self):
        """Test error on empty API key."""
        provider = OpenAIProvider()
        with pytest.raises(ValueError, match="API key must be provided"):
            provider.create_client("")

    def test_encode_image(self, test_image):
        """Test image encoding for OpenAI format."""
        provider = OpenAIProvider()
        encoded = provider.encode_image(test_image)

        assert encoded["type"] == "image_url"
        assert "url" in encoded["image_url"]
        assert encoded["image_url"]["url"].startswith("data:image/png;base64,")
        assert encoded["image_url"]["detail"] == "auto"

    def test_encode_image_with_detail(self, test_image):
        """Test image encoding with custom detail level."""
        provider = OpenAIProvider()
        encoded = provider.encode_image(test_image, detail="high")

        assert encoded["image_url"]["detail"] == "high"


# ============================================================================
# Unit Tests - GoogleProvider
# ============================================================================


class TestGoogleProvider:
    """Test GoogleProvider implementation."""

    def test_name(self):
        """Test provider name."""
        provider = GoogleProvider()
        assert provider.name == "google"

    def test_supported_models(self):
        """Test supported models list."""
        provider = GoogleProvider()
        models = provider.get_supported_models()
        assert "gemini-3-pro" in models
        assert "gemini-2.5-pro" in models
        assert "gemini-2.5-flash" in models

    def test_is_model_supported(self):
        """Test model support check."""
        provider = GoogleProvider()
        assert provider.is_model_supported("gemini-2.5-pro")
        assert not provider.is_model_supported("gpt-4o")

    def test_create_client_empty_key(self):
        """Test error on empty API key."""
        provider = GoogleProvider()
        with pytest.raises(ValueError, match="API key must be provided"):
            provider.create_client("")

    def test_encode_image(self, test_image):
        """Test image encoding for Google format."""
        provider = GoogleProvider()
        encoded = provider.encode_image(test_image)

        # Google uses PIL images directly
        assert encoded["type"] == "image"
        assert encoded["image"] is test_image
        assert encoded["media_type"] == "image/png"


# ============================================================================
# Unit Tests - Base Provider
# ============================================================================


class TestBaseProvider:
    """Test BaseAPIProvider utilities."""

    def test_image_to_base64(self, test_image):
        """Test base64 encoding utility."""
        base64_data = BaseAPIProvider.image_to_base64(test_image)
        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        # Should be valid base64
        import base64
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0


# ============================================================================
# Integration Tests - Real API Calls
# ============================================================================


class TestAnthropicIntegration:
    """Integration tests for Anthropic API."""

    def test_create_client(self, anthropic_api_key):
        """Test creating a real client."""
        provider = AnthropicProvider()
        client = provider.create_client(anthropic_api_key)
        assert client is not None

    def test_send_message_text_only(self, anthropic_api_key):
        """Test sending a simple text message."""
        provider = AnthropicProvider()
        client = provider.create_client(anthropic_api_key)

        response = provider.send_message(
            client=client,
            model="claude-sonnet-4-5-20250929",
            system="You are a helpful assistant.",
            content=[{"type": "text", "text": "Say 'hello' and nothing else."}],
            max_tokens=100,
        )

        assert "hello" in response.lower()

    def test_send_message_with_image(self, anthropic_api_key, real_image):
        """Test sending a message with an image."""
        provider = AnthropicProvider()
        client = provider.create_client(anthropic_api_key)

        response = provider.send_message(
            client=client,
            model="claude-sonnet-4-5-20250929",
            system="You are a UI analyst. Describe UI elements you see.",
            content=[
                {"type": "text", "text": "What is in this image? Be brief."},
                provider.encode_image(real_image),
            ],
            max_tokens=200,
        )

        assert len(response) > 10
        print(f"\nAnthropic response: {response[:200]}...")


class TestOpenAIIntegration:
    """Integration tests for OpenAI API."""

    def test_create_client(self, openai_api_key):
        """Test creating a real client."""
        provider = OpenAIProvider()
        client = provider.create_client(openai_api_key)
        assert client is not None

    def test_send_message_text_only(self, openai_api_key):
        """Test sending a simple text message."""
        provider = OpenAIProvider()
        client = provider.create_client(openai_api_key)

        response = provider.send_message(
            client=client,
            model="gpt-4o",
            system="You are a helpful assistant.",
            content=[{"type": "text", "text": "Say 'hello' and nothing else."}],
            max_tokens=100,
        )

        assert "hello" in response.lower()

    def test_send_message_with_image(self, openai_api_key, real_image):
        """Test sending a message with an image."""
        provider = OpenAIProvider()
        client = provider.create_client(openai_api_key)

        response = provider.send_message(
            client=client,
            model="gpt-4o",
            system="You are a UI analyst. Describe UI elements you see.",
            content=[
                {"type": "text", "text": "What is in this image? Be brief."},
                provider.encode_image(real_image),
            ],
            max_tokens=200,
        )

        assert len(response) > 10
        print(f"\nOpenAI response: {response[:200]}...")


class TestGoogleIntegration:
    """Integration tests for Google Gemini API."""

    def test_create_client(self, google_api_key):
        """Test creating a real client."""
        provider = GoogleProvider()
        client = provider.create_client(google_api_key)
        assert client is not None
        # The new google-genai SDK returns a Client object, not a dict
        # Verify it has the expected models attribute
        assert hasattr(client, "models")

    def test_send_message_text_only(self, google_api_key):
        """Test sending a simple text message."""
        provider = GoogleProvider()
        client = provider.create_client(google_api_key)

        response = provider.send_message(
            client=client,
            model="gemini-2.0-flash",
            system="You are a helpful assistant.",
            content=[{"type": "text", "text": "Say 'hello' and nothing else."}],
            max_tokens=100,
        )

        assert "hello" in response.lower()

    def test_send_message_with_image(self, google_api_key, real_image):
        """Test sending a message with an image."""
        provider = GoogleProvider()
        client = provider.create_client(google_api_key)

        response = provider.send_message(
            client=client,
            model="gemini-2.0-flash",
            system="You are a UI analyst. Describe UI elements you see.",
            content=[
                {"type": "text", "text": "What is in this image? Be brief."},
                provider.encode_image(real_image),
            ],
            max_tokens=200,
        )

        assert len(response) > 10
        print(f"\nGoogle response: {response[:200]}...")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in providers."""

    def test_anthropic_invalid_api_key(self):
        """Test error with invalid API key."""
        provider = AnthropicProvider()
        client = provider.create_client("invalid-key-12345")

        with pytest.raises(RuntimeError, match="Anthropic API call failed"):
            provider.send_message(
                client=client,
                model="claude-sonnet-4-5-20250929",
                system="Test",
                content=[{"type": "text", "text": "Test"}],
            )

    def test_openai_invalid_api_key(self):
        """Test error with invalid API key."""
        provider = OpenAIProvider()
        client = provider.create_client("invalid-key-12345")

        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            provider.send_message(
                client=client,
                model="gpt-4o",
                system="Test",
                content=[{"type": "text", "text": "Test"}],
            )

    def test_google_invalid_api_key(self):
        """Test error with invalid API key."""
        provider = GoogleProvider()
        client = provider.create_client("invalid-key-12345")

        with pytest.raises(RuntimeError, match="Google AI API call failed"):
            provider.send_message(
                client=client,
                model="gemini-2.5-flash",
                system="Test",
                content=[{"type": "text", "text": "Test"}],
            )

    def test_unsupported_model_anthropic(self, anthropic_api_key):
        """Test error with unsupported model."""
        provider = AnthropicProvider()
        client = provider.create_client(anthropic_api_key)

        with pytest.raises(ValueError, match="not supported"):
            provider.send_message(
                client=client,
                model="unsupported-model",
                system="Test",
                content=[{"type": "text", "text": "Test"}],
            )
