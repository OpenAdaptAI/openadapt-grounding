"""Anthropic API provider for Claude models."""

from typing import Any, Dict, List, Optional

from PIL import Image

from openadapt_grounding.providers.base import BaseAPIProvider


class AnthropicProvider(BaseAPIProvider):
    """Provider for Anthropic's Claude API.

    Supports Claude models for vision-language tasks.

    Example:
        >>> provider = AnthropicProvider()
        >>> client = provider.create_client(api_key="sk-ant-...")
        >>> image = Image.open("screenshot.png")
        >>> response = provider.send_message(
        ...     client=client,
        ...     model="claude-opus-4-5-20251101",
        ...     system="Describe UI elements in the image.",
        ...     content=[
        ...         {"type": "text", "text": "What buttons are visible?"},
        ...         provider.encode_image(image),
        ...     ],
        ... )
    """

    SUPPORTED_MODELS: List[str] = [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
    ]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    def create_client(self, api_key: str, **kwargs: Any) -> Any:
        """Create an Anthropic API client.

        Args:
            api_key: Anthropic API key. MUST be passed explicitly.
            **kwargs: Additional options:
                - base_url: Custom API base URL (optional).
                - timeout: Request timeout in seconds (default: 60.0).
                - max_retries: Maximum retry attempts (default: 2).

        Returns:
            anthropic.Anthropic client instance.

        Raises:
            ImportError: If anthropic package is not installed.
            ValueError: If api_key is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key must be provided and non-empty")

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is not installed. "
                "Install with: pip install anthropic"
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}

        if "base_url" in kwargs:
            client_kwargs["base_url"] = kwargs["base_url"]
        if "timeout" in kwargs:
            client_kwargs["timeout"] = kwargs["timeout"]
        if "max_retries" in kwargs:
            client_kwargs["max_retries"] = kwargs["max_retries"]

        return anthropic.Anthropic(**client_kwargs)

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
        """Send a message to Claude and get a response.

        Args:
            client: Anthropic client from create_client().
            model: Model identifier (e.g., "claude-opus-4-5-20251101").
            system: System prompt to set context/behavior.
            content: List of content blocks. Supported types:
                - {"type": "text", "text": "..."}
                - Image blocks from encode_image()
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Additional parameters for the API call.

        Returns:
            The text content of Claude's response.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API call fails.
        """
        self.validate_model(model)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                **kwargs,
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return ""

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    def encode_image(
        self,
        image: Image.Image,
        media_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Encode a PIL Image for the Anthropic API.

        Args:
            image: PIL Image to encode.
            media_type: MIME type (default: "image/png").
                       Supported: image/png, image/jpeg, image/gif, image/webp.

        Returns:
            A dict formatted for Anthropic's content array:
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "<base64-data>"
                }
            }
        """
        # Determine format from media_type
        format_map = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/gif": "GIF",
            "image/webp": "WEBP",
        }
        img_format = format_map.get(media_type, "PNG")

        base64_data = self.image_to_base64(image, format=img_format)

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }
