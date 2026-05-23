"""LLM call #1 — synthesize a candidate taxonomy YAML from a warehouse inventory."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..models import TaxonomyDocument
from .collector import SchemaInventory
from .prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE

if TYPE_CHECKING:
    from data_governance.rag_discovery.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_YAML_FENCE_RE = re.compile(r"```yaml\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


class TaxonomySynthesisError(RuntimeError):
    """Raised when the LLM response cannot be parsed into a taxonomy."""


@dataclass
class SynthesisResult:
    taxonomy: TaxonomyDocument
    yaml_text: str
    raw_response: str
    model: str
    tokens_used: Optional[int] = None


class TaxonomySynthesizer:
    """Use an injected ``LLMProvider`` to produce the candidate taxonomy."""

    def __init__(
        self,
        llm: "LLMProvider",
        max_tokens: int = 4000,
        max_tables: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_tables = max_tables
        self.temperature = temperature

    def synthesize(self, inventory: SchemaInventory) -> SynthesisResult:
        inventory_json = json.dumps(
            inventory.to_llm_dict(max_tables=self.max_tables),
            ensure_ascii=False,
            default=str,
        )
        prompt = SYNTHESIS_USER_TEMPLATE.format(inventory_json=inventory_json)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = getattr(response, "content", "") or ""
        yaml_text = self._extract_yaml(content)

        try:
            taxonomy = TaxonomyDocument.from_yaml_string(yaml_text)
        except Exception as exc:  # noqa: BLE001 — wrap parser errors
            raise TaxonomySynthesisError(
                f"LLM returned a YAML that failed to parse: {exc}\n\n"
                f"--- raw response ---\n{content[:2000]}"
            ) from exc

        if not taxonomy.concept_groups:
            raise TaxonomySynthesisError(
                "LLM produced YAML with no concept_groups — refusing to "
                "treat as a valid taxonomy."
            )

        return SynthesisResult(
            taxonomy=taxonomy,
            yaml_text=yaml_text,
            raw_response=content,
            model=getattr(self.llm, "model_name", "unknown") or "unknown",
            tokens_used=getattr(response, "tokens_used", None),
        )

    @staticmethod
    def _extract_yaml(content: str) -> str:
        """Extract the YAML block from the LLM response.

        Accepts ```yaml ... ``` fences or, as a fallback, treats the whole
        content as YAML — but never trusts surrounding prose to be valid.
        """
        if not content:
            raise TaxonomySynthesisError("Empty LLM response.")
        match = _YAML_FENCE_RE.search(content)
        if match:
            return match.group("body").strip()
        # Strip a leading ``` even when language tag is missing
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.lstrip("`").strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
            return stripped
        return stripped
