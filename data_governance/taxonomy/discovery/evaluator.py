"""LLM call #2 — evaluate AS-IS taxonomy + render improvement plan as HTML.

The LLM is asked for **JSON** (not HTML). Python is responsible for rendering
that JSON into an HTML report so the LLM cannot inject markup or scripts.
"""
from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..models import TaxonomyDocument
from ..scorer import TaxonomyScore, TaxonomyScorer
from .prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_TEMPLATE

if TYPE_CHECKING:
    from data_governance.rag_discovery.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class TaxonomyEvaluationError(RuntimeError):
    """Raised when the LLM response cannot be parsed as the expected JSON."""


@dataclass
class EvaluationResult:
    plan: Dict[str, Any]
    plan_html: str
    raw_response: str
    model: str
    tokens_used: Optional[int] = None


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


class TaxonomyEvaluator:
    """Use an injected ``LLMProvider`` to produce the improvement plan."""

    SEVERITY_COLOR = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}

    def __init__(
        self,
        llm: "LLMProvider",
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def evaluate(
        self,
        taxonomy: TaxonomyDocument,
        score: TaxonomyScore,
        framework: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        framework = framework or {
            "weights": TaxonomyScorer.WEIGHTS,
            "maturity_levels": {
                "1": "Initial (0-20)", "2": "Managed (21-40)",
                "3": "Defined (41-60)", "4": "Measured (61-80)",
                "5": "Optimized (81-100)",
            },
        }
        prompt = EVALUATION_USER_TEMPLATE.format(
            framework_json=json.dumps(framework, ensure_ascii=False),
            score_json=json.dumps(score.to_dict(), ensure_ascii=False),
            taxonomy_yaml=taxonomy.to_yaml(),
        )
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = getattr(response, "content", "") or ""
        plan = self._parse_json(content)
        self._validate(plan)
        plan_html = self.render_html(plan, score)
        return EvaluationResult(
            plan=plan,
            plan_html=plan_html,
            raw_response=content,
            model=getattr(self.llm, "model_name", "unknown") or "unknown",
            tokens_used=getattr(response, "tokens_used", None),
        )

    # ------------------------------------------------------------------
    # Parsing / validation
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        if not content:
            raise TaxonomyEvaluationError("Empty LLM response.")
        # Strip ```json fences if present
        cleaned = content.strip()
        m = re.search(r"```(?:json)?\s*(?P<body>.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if m:
            cleaned = m.group("body").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise TaxonomyEvaluationError(
                f"LLM response is not valid JSON: {exc}\n"
                f"--- raw response ---\n{content[:1500]}"
            ) from exc

    @staticmethod
    def _validate(plan: Dict[str, Any]) -> None:
        required = {"executive_summary", "dimensions"}
        missing = required - set(plan or {})
        if missing:
            raise TaxonomyEvaluationError(
                f"Plan JSON is missing required fields: {sorted(missing)}"
            )
        if not isinstance(plan["dimensions"], list) or not plan["dimensions"]:
            raise TaxonomyEvaluationError(
                "Plan JSON has no `dimensions` entries."
            )

    # ------------------------------------------------------------------
    # HTML rendering — all interpolation is HTML-escaped
    # ------------------------------------------------------------------
    def render_html(self, plan: Dict[str, Any], score: TaxonomyScore) -> str:
        parts: List[str] = []
        parts.append(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'self' 'unsafe-inline'">
<title>Plano de Melhoria de Taxonomia</title>
<style>
 :root {{ --dark:#0B2545; --mid:#134074; --accent:#3B82F6; --green:#10B981;
         --amber:#F59E0B; --red:#EF4444; --bg:#F8FAFC; --card:#FFFFFF;
         --text:#0F172A; --muted:#475569; --border:#E2E8F0; }}
 * {{ box-sizing:border-box; font-family: 'Inter', system-ui, sans-serif; }}
 body {{ margin:0; background:var(--bg); color:var(--text); }}
 header {{ background:var(--dark); color:white; padding:1.75rem 2.5rem; }}
 header h1 {{ margin:0; font-size:1.6rem; font-weight:600; }}
 header .sub {{ color:#94A3B8; font-size:0.9rem; margin-top:0.25rem; }}
 main {{ max-width:1100px; margin:2rem auto; padding:0 2rem; }}
 .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px;
         padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
 .pill {{ display:inline-block; padding:0.2rem 0.6rem; border-radius:999px;
         font-size:0.75rem; font-weight:600; margin-right:0.4rem; }}
 .pill-effort-low    {{ background:#DCFCE7; color:#065F46; }}
 .pill-effort-medium {{ background:#FEF3C7; color:#92400E; }}
 .pill-effort-high   {{ background:#FEE2E2; color:#991B1B; }}
 .pill-impact-low    {{ background:#E5E7EB; color:#374151; }}
 .pill-impact-medium {{ background:#DBEAFE; color:#1E3A8A; }}
 .pill-impact-high   {{ background:#FDE68A; color:#854D0E; }}
 h2 {{ font-size:1.2rem; color:var(--dark); margin:0 0 1rem 0; }}
 h3 {{ font-size:1.05rem; color:var(--dark); margin:0 0 0.5rem 0; }}
 .action {{ border-left:3px solid var(--accent); padding:0.75rem 1rem;
            margin-bottom:0.75rem; background:#F8FAFC; border-radius:0 8px 8px 0; }}
 .action-title {{ font-weight:600; margin-bottom:0.25rem; }}
 .action-detail {{ font-size:0.9rem; color:var(--muted); margin-bottom:0.5rem; }}
 .score-bar {{ display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; }}
 .score-bar .bar {{ flex:1; height:8px; background:var(--border); border-radius:999px; overflow:hidden; }}
 .score-bar .fill {{ height:100%; background:var(--accent); }}
 ul {{ margin:0; padding-left:1.25rem; }}
 ul li {{ margin-bottom:0.25rem; }}
 .summary {{ font-size:1rem; color:var(--text); line-height:1.5; }}
 .badge-quickwin {{ background:var(--green); color:white; padding:0.15rem 0.5rem;
                    border-radius:6px; font-size:0.7rem; margin-left:0.5rem; }}
 @media print {{ body {{ background:white; }} .card {{ break-inside: avoid; }} }}
</style>
</head>
<body>
<header>
  <h1>Plano de Melhoria de Taxonomia</h1>
  <div class="sub">Score atual: {_e(score.overall_score)}/100 · Nível {_e(score.maturity_level)} — {_e(score.maturity_label)}</div>
</header>
<main>
""")

        parts.append(f"""<section class="card">
  <h2>Resumo executivo</h2>
  <p class="summary">{_e(plan.get('executive_summary', ''))}</p>
</section>""")

        strengths = plan.get("strengths") or []
        risks = plan.get("risks") or []
        if strengths or risks:
            parts.append('<section class="card"><div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">')
            if strengths:
                parts.append('<div><h3 style="color:var(--green)">Pontos fortes</h3><ul>')
                for s in strengths:
                    parts.append(f"<li>{_e(s)}</li>")
                parts.append("</ul></div>")
            if risks:
                parts.append('<div><h3 style="color:var(--red)">Riscos</h3><ul>')
                for r in risks:
                    parts.append(f"<li>{_e(r)}</li>")
                parts.append("</ul></div>")
            parts.append("</div></section>")

        quick_wins = {q.strip().lower() for q in (plan.get("quick_wins") or []) if isinstance(q, str)}

        for dim in plan.get("dimensions", []):
            if not isinstance(dim, dict):
                continue
            name = str(dim.get("dimension", "")).replace("_", " ").title()
            current = dim.get("current_score", 0)
            target = dim.get("target_score", current)
            try:
                pct = max(0.0, min(100.0, float(current)))
            except (TypeError, ValueError):
                pct = 0.0
            parts.append(f"""<section class="card">
  <h2>{_e(name)}</h2>
  <div class="score-bar">
    <span style="min-width:140px; font-size:0.85rem; color:var(--muted)">{_e(current)}/100 → meta {_e(target)}/100</span>
    <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
  </div>
  <p class="summary" style="margin-bottom:1rem;">{_e(dim.get('diagnosis', ''))}</p>
""")
            for action in dim.get("actions") or []:
                if not isinstance(action, dict):
                    continue
                title = action.get("title", "")
                quickwin = title.strip().lower() in quick_wins
                effort = str(action.get("effort", "medium")).lower()
                impact = str(action.get("expected_impact", "medium")).lower()
                parts.append(f"""<div class="action">
  <div class="action-title">{_e(title)}{'<span class="badge-quickwin">Quick Win</span>' if quickwin else ''}</div>
  <div class="action-detail">{_e(action.get('detail', ''))}</div>
  <span class="pill pill-effort-{_e(effort)}">Esforço: {_e(effort)}</span>
  <span class="pill pill-impact-{_e(impact)}">Impacto: {_e(impact)}</span>
""")
                if action.get("owner_role"):
                    parts.append(f'<span class="pill" style="background:#E0F2FE;color:#0369A1">{_e(action["owner_role"])}</span>')
                deps = action.get("depends_on") or []
                if deps:
                    parts.append(
                        f'<div style="font-size:0.75rem; color:var(--muted); margin-top:0.4rem;">Depende de: {_e(", ".join(deps))}</div>'
                    )
                parts.append("</div>")
            parts.append("</section>")

        parts.append("</main></body></html>")
        return "".join(parts)
