"""
OpenAI LLM Provider
Supports GPT-4, GPT-4o, GPT-4o-mini, etc.
"""

import os
from typing import Optional

from ..base import LLMProvider, LLMResponse


class OpenAILLM(LLMProvider):
    """
    OpenAI API LLM provider

    Models available:
    - gpt-4o (latest, multimodal)
    - gpt-4o-mini (fast, cheap)
    - gpt-4-turbo (128K context)
    - gpt-4 (original)
    - gpt-3.5-turbo (legacy, cheap)

    Supports Azure OpenAI via base_url parameter.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048
    ):
        """
        Initialize OpenAI LLM

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL (for Azure or proxies)
            organization: OpenAI organization ID
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens for response
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

        self._model_name = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_API_URL")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize client
        client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if organization:
            client_kwargs["organization"] = organization

        self._client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text response from OpenAI

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens or self._default_max_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self._model_name,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0
        )

    @property
    def model_name(self) -> str:
        """Model name"""
        return self._model_name
