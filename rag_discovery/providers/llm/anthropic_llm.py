"""
Anthropic Claude LLM Provider
"""

import os
from typing import Optional

from ..base import LLMProvider, LLMResponse


class AnthropicLLM(LLMProvider):
    """
    Cliente para modelos Claude via SDK oficial da Anthropic.

    Modelos recomendados:
    - claude-3-5-sonnet-20240620 (equilíbrio custo/desempenho)
    - claude-3-opus-20240229 (qualidade superior)
    - claude-3-haiku-20240307 (baixo custo)
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        api_key: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depende de instalação opcional
            raise ImportError(
                "anthropic not installed. Install with: pip install anthropic"
            ) from exc

        self._model_name = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature if temperature is not None else self._default_temperature,
            messages=messages,
        )

        content_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                content_parts.append(block.get("text", ""))

        return LLMResponse(
            content="".join(content_parts),
            model=self._model_name,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

    @property
    def model_name(self) -> str:
        return self._model_name
