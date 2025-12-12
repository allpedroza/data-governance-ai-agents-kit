"""
Vertex AI LLM Provider
Supports Gemini models on Google Cloud
"""

import os
from typing import Optional

from ..base import LLMProvider, LLMResponse


class VertexAILLM(LLMProvider):
    """
    Google Vertex AI LLM provider (Gemini)

    Models available:
    - gemini-1.5-pro (latest, 1M context)
    - gemini-1.5-flash (fast, cheap)
    - gemini-1.0-pro (stable)

    Requires Google Cloud credentials:
    - Set GOOGLE_APPLICATION_CREDENTIALS env var, or
    - Run: gcloud auth application-default login
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048
    ):
        """
        Initialize Vertex AI LLM

        Args:
            model: Gemini model name
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            location: GCP region
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens for response
        """
        self._model_name = model
        self._project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
        self._location = location
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = None

        if not self._project_id:
            raise ValueError(
                "GCP project ID not provided. "
                "Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter."
            )

    def _get_client(self):
        """Lazy initialization of Vertex AI client"""
        if self._client is None:
            try:
                from langchain_google_vertexai import VertexAI
            except ImportError:
                raise ImportError(
                    "langchain-google-vertexai not installed. "
                    "Install with: pip install langchain-google-vertexai"
                )

            self._client = VertexAI(
                model_name=self._model_name,
                project=self._project_id,
                location=self._location,
                temperature=self._default_temperature,
                max_output_tokens=self._default_max_tokens
            )

        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text response from Vertex AI

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        client = self._get_client()

        # Build full prompt with system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # For now, use simple invoke
        # Note: LangChain VertexAI doesn't expose token counts easily
        response = client.invoke(full_prompt)

        return LLMResponse(
            content=response if isinstance(response, str) else str(response),
            model=self._model_name,
            input_tokens=0,  # VertexAI via LangChain doesn't easily expose this
            output_tokens=0
        )

    @property
    def model_name(self) -> str:
        """Model name"""
        return self._model_name


class GeminiDirectLLM(LLMProvider):
    """
    Direct Gemini API provider (without LangChain)
    Uses google-generativeai package directly
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048
    ):
        """
        Initialize direct Gemini API

        Args:
            model: Gemini model name
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens for response
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

        self._model_name = model
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        if self._api_key:
            genai.configure(api_key=self._api_key)

        self._model = genai.GenerativeModel(model)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Generate text response from Gemini"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = {
            "temperature": temperature if temperature is not None else self._default_temperature,
            "max_output_tokens": max_tokens or self._default_max_tokens
        }

        response = self._model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        # Extract token counts if available
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        return LLMResponse(
            content=response.text if hasattr(response, 'text') else str(response),
            model=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    @property
    def model_name(self) -> str:
        """Model name"""
        return self._model_name
