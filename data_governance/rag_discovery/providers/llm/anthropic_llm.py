# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
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
