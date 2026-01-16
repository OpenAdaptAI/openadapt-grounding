"""OpenAI API provider for GPT models."""

from typing import Any, Dict, List, Optional

from PIL import Image

from openadapt_grounding.providers.base import BaseAPIProvider


class OpenAIProvider(BaseAPIProvider):
    """Provider for OpenAI's GPT API.

    Supports GPT models with vision capabilities for vision-language tasks.

    Example:
        >>> provider = OpenAIProvider()
        >>> client = provider.create_client(api_key="sk-...")
        >>> image = Image.open("screenshot.png")
        >>> response = provider.send_message(
        ...     client=client,
        ...     model="gpt-5.2",
        ...     system="Describe UI elements in the image.",
        ...     content=[
        ...         {"type": "text", "text": "What buttons are visible?"},
        ...         provider.encode_image(image),
        ...     ],
        ... )
    """

    SUPPORTED_MODELS: List[str] = [
        "gpt-5.2",
        "gpt-4o",
    ]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "openai"

    def create_client(self, api_key: str, **kwargs: Any) -> Any:
        """Create an OpenAI API client.

        Args:
            api_key: OpenAI API key. MUST be passed explicitly.
            **kwargs: Additional options:
                - base_url: Custom API base URL (for Azure, proxies, etc.).
                - organization: OpenAI organization ID (optional).
                - timeout: Request timeout in seconds (default: 60.0).
                - max_retries: Maximum retry attempts (default: 2).

        Returns:
            openai.OpenAI client instance.

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If api_key is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key must be provided and non-empty")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is not installed. "
                "Install with: pip install openai"
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}

        if "base_url" in kwargs:
            client_kwargs["base_url"] = kwargs["base_url"]
        if "organization" in kwargs:
            client_kwargs["organization"] = kwargs["organization"]
        if "timeout" in kwargs:
            client_kwargs["timeout"] = kwargs["timeout"]
        if "max_retries" in kwargs:
            client_kwargs["max_retries"] = kwargs["max_retries"]

        return OpenAI(**client_kwargs)

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
        """Send a message to GPT and get a response.

        Args:
            client: OpenAI client from create_client().
            model: Model identifier (e.g., "gpt-5.2", "gpt-4o").
            system: System prompt to set context/behavior.
            content: List of content blocks. Supported types:
                - {"type": "text", "text": "..."}
                - Image blocks from encode_image()
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Additional parameters for the API call.

        Returns:
            The text content of GPT's response.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API call fails.
        """
        self.validate_model(model)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def encode_image(
        self,
        image: Image.Image,
        media_type: str = "image/png",
        detail: str = "auto",
    ) -> Dict[str, Any]:
        """Encode a PIL Image for the OpenAI API.

        Args:
            image: PIL Image to encode.
            media_type: MIME type (default: "image/png").
                       Supported: image/png, image/jpeg, image/gif, image/webp.
            detail: Image detail level - "auto", "low", or "high".
                   "auto" lets the model decide based on image size.
                   "low" uses fewer tokens, "high" provides more detail.

        Returns:
            A dict formatted for OpenAI's content array:
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,<base64-data>",
                    "detail": "auto"
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
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{base64_data}",
                "detail": detail,
            },
        }
