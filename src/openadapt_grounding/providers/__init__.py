"""VLM API providers for vision-language model integrations.

This module provides a unified interface for interacting with different
VLM APIs (Anthropic, OpenAI, Google/Gemini) for vision-language tasks.

API keys should be passed as parameters to create_client(), NOT read
from environment variables. This allows the calling application (e.g.,
openadapt-ml) to manage API key configuration centrally.

Example:
    >>> from openadapt_grounding.providers import get_provider
    >>> from PIL import Image
    >>>
    >>> # Get a provider by name
    >>> provider = get_provider("anthropic")
    >>>
    >>> # Create client with explicit API key
    >>> client = provider.create_client(api_key="sk-ant-...")
    >>>
    >>> # Prepare content with image
    >>> image = Image.open("screenshot.png")
    >>> content = [
    ...     {"type": "text", "text": "What buttons are visible?"},
    ...     provider.encode_image(image),
    ... ]
    >>>
    >>> # Send message
    >>> response = provider.send_message(
    ...     client=client,
    ...     model="claude-opus-4-5-20251101",
    ...     system="Describe UI elements in the image.",
    ...     content=content,
    ... )

Supported Providers:
    - anthropic: Claude models (claude-opus-4-5-20251101, claude-sonnet-4-5-20250929)
    - openai: GPT models (gpt-5.2, gpt-4o)
    - google: Gemini models (gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash)
"""

from typing import Dict, List, Optional, Type

from openadapt_grounding.providers.base import BaseAPIProvider
from openadapt_grounding.providers.anthropic import AnthropicProvider
from openadapt_grounding.providers.openai import OpenAIProvider
from openadapt_grounding.providers.google import GoogleProvider


# Registry of available providers
_PROVIDERS: Dict[str, Type[BaseAPIProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

# Aliases for convenience
_PROVIDER_ALIASES: Dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
}


def get_provider(name: str) -> BaseAPIProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name or alias. Supported values:
            - "anthropic" or "claude" - Anthropic Claude models
            - "openai" or "gpt" - OpenAI GPT models
            - "google" or "gemini" - Google Gemini models

    Returns:
        An instance of the requested provider.

    Raises:
        ValueError: If the provider name is not recognized.

    Example:
        >>> provider = get_provider("anthropic")
        >>> client = provider.create_client(api_key="sk-ant-...")
    """
    # Resolve alias if provided
    resolved_name = _PROVIDER_ALIASES.get(name.lower(), name.lower())

    if resolved_name not in _PROVIDERS:
        available = list(_PROVIDERS.keys()) + list(_PROVIDER_ALIASES.keys())
        raise ValueError(
            f"Unknown provider: '{name}'. "
            f"Available providers: {sorted(set(available))}"
        )

    return _PROVIDERS[resolved_name]()


def list_providers() -> List[str]:
    """Return a list of available provider names.

    Returns:
        List of provider name strings.

    Example:
        >>> providers = list_providers()
        >>> print(providers)
        ['anthropic', 'google', 'openai']
    """
    return sorted(_PROVIDERS.keys())


def list_all_models() -> Dict[str, List[str]]:
    """Return all supported models grouped by provider.

    Returns:
        Dict mapping provider names to lists of model identifiers.

    Example:
        >>> models = list_all_models()
        >>> print(models["anthropic"])
        ['claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929']
    """
    return {
        name: provider_class.SUPPORTED_MODELS
        for name, provider_class in _PROVIDERS.items()
    }


def get_provider_for_model(model: str) -> Optional[BaseAPIProvider]:
    """Get the provider instance for a given model.

    Args:
        model: Model identifier (e.g., "claude-opus-4-5-20251101").

    Returns:
        Provider instance that supports the model, or None if not found.

    Example:
        >>> provider = get_provider_for_model("gpt-5.2")
        >>> print(provider.name)
        'openai'
    """
    for provider_class in _PROVIDERS.values():
        if model in provider_class.SUPPORTED_MODELS:
            return provider_class()
    return None


__all__ = [
    # Base class
    "BaseAPIProvider",
    # Provider implementations
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    # Factory functions
    "get_provider",
    "get_provider_for_model",
    "list_providers",
    "list_all_models",
]
