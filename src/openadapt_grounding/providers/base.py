"""Base protocol for VLM API providers."""

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class BaseAPIProvider(ABC):
    """Abstract base class for VLM API providers.

    Provides a unified interface for interacting with different VLM APIs
    (Anthropic, OpenAI, Google/Gemini) for vision-language tasks.

    API keys should be passed as parameters to create_client(), NOT read
    from environment variables. This allows the calling application to
    manage API key configuration centrally.

    Example:
        >>> provider = AnthropicProvider()
        >>> client = provider.create_client(api_key="sk-...")
        >>> response = provider.send_message(
        ...     client=client,
        ...     model="claude-opus-4-5-20251101",
        ...     system="You are a helpful assistant.",
        ...     content=[{"type": "text", "text": "Describe this image."},
        ...              provider.encode_image(image)],
        ...     max_tokens=1024,
        ... )
    """

    # Supported models for this provider
    SUPPORTED_MODELS: List[str] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai', 'google')."""
        ...

    @abstractmethod
    def create_client(self, api_key: str, **kwargs: Any) -> Any:
        """Create and return an API client instance.

        Args:
            api_key: The API key for authentication. MUST be passed explicitly,
                    not read from environment variables.
            **kwargs: Additional provider-specific configuration options
                     (e.g., base_url, timeout, organization).

        Returns:
            A client instance specific to the provider.

        Raises:
            ImportError: If the provider's SDK is not installed.
            ValueError: If the API key is empty or invalid.
        """
        ...

    @abstractmethod
    def send_message(
        self,
        client: Any,
        model: str,
        system: str,
        content: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a message to the VLM and get a response.

        Args:
            client: The client instance from create_client().
            model: Model identifier (e.g., "claude-opus-4-5-20251101").
            system: System prompt to set context/behavior.
            content: List of content blocks (text, images, etc.).
                    Use encode_image() to create image content blocks.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Additional provider-specific parameters.

        Returns:
            The text content of the model's response.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API call fails.
        """
        ...

    @abstractmethod
    def encode_image(
        self,
        image: Image.Image,
        media_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Encode a PIL Image for the API.

        Args:
            image: PIL Image to encode.
            media_type: MIME type for the image (default: "image/png").

        Returns:
            A dict formatted for the provider's content array.
            The exact structure varies by provider.
        """
        ...

    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by this provider.

        Args:
            model: Model identifier to check.

        Returns:
            True if the model is in SUPPORTED_MODELS.
        """
        return model in self.SUPPORTED_MODELS

    def validate_model(self, model: str) -> None:
        """Validate that a model is supported, raising if not.

        Args:
            model: Model identifier to validate.

        Raises:
            ValueError: If the model is not supported.
        """
        if not self.is_model_supported(model):
            raise ValueError(
                f"Model '{model}' is not supported by {self.name}. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL Image to a base64-encoded string.

        Args:
            image: PIL Image to encode.
            format: Image format for encoding (default: "PNG").

        Returns:
            Base64-encoded string of the image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_supported_models(self) -> List[str]:
        """Return the list of supported model identifiers.

        Returns:
            List of model identifier strings.
        """
        return self.SUPPORTED_MODELS.copy()
