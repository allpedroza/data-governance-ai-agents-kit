"""
DeepSeek LLM Provider
Usa a API compatível com OpenAI disponibilizada pela DeepSeek.
"""

import os
from typing import Optional

from ..base import LLMProvider, LLMResponse


class DeepSeekLLM(LLMProvider):
    """
    Cliente para a API da DeepSeek usando o SDK do OpenAI.

    Modelos disponíveis:
    - deepseek-chat (geral)
    - deepseek-reasoner (raciocínio avançado)
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depende de instalação opcional
            raise ImportError("openai not installed. Install with: pip install openai") from exc

        self._model_name = model
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._base_url = base_url
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        if not self._api_key:
            raise ValueError(
                "DeepSeek API key not provided. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

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

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens or self._default_max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self._model_name,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    @property
    def model_name(self) -> str:
        return self._model_name
