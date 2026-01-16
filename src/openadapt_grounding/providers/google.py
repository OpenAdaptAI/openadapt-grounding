"""Google AI API provider for Gemini models."""

from typing import Any, Dict, List, Optional

from PIL import Image

from openadapt_grounding.providers.base import BaseAPIProvider


class GoogleProvider(BaseAPIProvider):
    """Provider for Google's Gemini API.

    Supports Gemini models for vision-language tasks.

    Example:
        >>> provider = GoogleProvider()
        >>> client = provider.create_client(api_key="...")
        >>> image = Image.open("screenshot.png")
        >>> response = provider.send_message(
        ...     client=client,
        ...     model="gemini-2.5-pro",
        ...     system="Describe UI elements in the image.",
        ...     content=[
        ...         {"type": "text", "text": "What buttons are visible?"},
        ...         provider.encode_image(image),
        ...     ],
        ... )
    """

    SUPPORTED_MODELS: List[str] = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "google"

    def create_client(self, api_key: str, **kwargs: Any) -> Any:
        """Create a Google Generative AI client.

        Args:
            api_key: Google AI API key. MUST be passed explicitly.
            **kwargs: Additional options (reserved for future use).

        Returns:
            A dict containing the configured API key and settings.
            Note: google.generativeai uses a configure() pattern rather
            than a client object, so we wrap the configuration.

        Raises:
            ImportError: If google-generativeai package is not installed.
            ValueError: If api_key is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key must be provided and non-empty")

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Install with: pip install google-generativeai"
            )

        # Configure the library with the API key
        genai.configure(api_key=api_key)

        # Return a wrapper dict that holds the module reference and config
        # This allows multiple clients with different keys if needed
        return {
            "genai": genai,
            "api_key": api_key,
            **kwargs,
        }

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
        """Send a message to Gemini and get a response.

        Args:
            client: Client dict from create_client().
            model: Model identifier (e.g., "gemini-3-pro").
            system: System prompt to set context/behavior.
            content: List of content blocks. Supported types:
                - {"type": "text", "text": "..."}
                - Image blocks from encode_image()
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 = deterministic).
            **kwargs: Additional parameters for the API call.

        Returns:
            The text content of Gemini's response.

        Raises:
            ValueError: If the model is not supported.
            RuntimeError: If the API call fails.
        """
        self.validate_model(model)

        try:
            genai = client["genai"]

            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Create the model with system instruction
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system,
                generation_config=generation_config,
            )

            # Convert content to Gemini format
            gemini_content = self._convert_content(content)

            # Generate response
            response = gemini_model.generate_content(gemini_content)

            # Extract text from response
            if response.text:
                return response.text
            return ""

        except Exception as e:
            raise RuntimeError(f"Google AI API call failed: {e}") from e

    def encode_image(
        self,
        image: Image.Image,
        media_type: str = "image/png",
    ) -> Dict[str, Any]:
        """Encode a PIL Image for the Gemini API.

        Args:
            image: PIL Image to encode.
            media_type: MIME type (default: "image/png").
                       Supported: image/png, image/jpeg, image/gif, image/webp.

        Returns:
            A dict formatted for our content array that will be
            converted to Gemini format in _convert_content():
            {
                "type": "image",
                "image": <PIL.Image>,
                "media_type": "image/png"
            }
        """
        return {
            "type": "image",
            "image": image,
            "media_type": media_type,
        }

    def _convert_content(self, content: List[Dict[str, Any]]) -> List[Any]:
        """Convert our generic content format to Gemini format.

        Args:
            content: List of content blocks in our format.

        Returns:
            List of content items in Gemini's expected format.
        """
        gemini_content = []

        for item in content:
            if item.get("type") == "text":
                gemini_content.append(item["text"])
            elif item.get("type") == "image":
                # Gemini accepts PIL Images directly
                gemini_content.append(item["image"])

        return gemini_content
