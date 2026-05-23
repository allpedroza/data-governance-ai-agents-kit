"""Premium standalone HTML artifact for taxonomy AS-IS + scoring.

All user-supplied taxonomy content is HTML-escaped before interpolation to
prevent XSS payloads from a malicious YAML document leaking into the
rendered artifact. JSON injected into the in-page ``<script>`` block is
escaped against ``</script>`` break-outs.
"""
from __future__ import annotations

import html
import json
from typing import Any, Dict, Iterable


def _e(value: Any) -> str:
    """HTML-escape a value, including quotes for safe attribute use."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _safe_json(value: Any) -> str:
    """JSON-serialize for inline ``<script>`` injection."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _join_pills(items: Iterable[Any], cls: str = "pill") -> str:
    return "".join(f"<span class='{_e(cls)}'>{_e(i)}</span>" for i in items)


def generate_taxonomy_html(taxonomy_data: dict, score_data: dict) -> str:
    """Generate the premium HTML artifact with AS-IS taxonomy and scoring pages."""
    meta = taxonomy_data.get("metadata", {}) or {}
    title = meta.get("title", "Data Dictionary Taxonomy")
    version = meta.get("version", "1.0")
    domain = meta.get("domain", "Enterprise")
    platform = meta.get("platform", "Data Lake")

    overall_score = score_data.get("overall_score", 0)
    mat_level = score_data.get("maturity_level", 1)
    mat_label = score_data.get("maturity_label", "Initial")

    groups = taxonomy_data.get("concept_groups", []) or []
    computed_concepts = sum(len(g.get("concepts", []) or []) for g in groups)
    computed_aliases = sum(
        len(c.get("aliases", []) or [])
        for g in groups
        for c in (g.get("concepts", []) or [])
    )
    total_concepts = meta.get("total_concepts") or computed_concepts
    total_aliases = meta.get("total_aliases") or computed_aliases
    total_groups = len(groups)

    score_color = "score-red"
    if overall_score >= 70:
        score_color = "score-green"
    elif overall_score >= 40:
        score_color = "score-yellow"

    _ai_desc = (taxonomy_data.get("ai_agent_instructions") or {}).get("description", "")

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net">
    <title>{_e(title)} v{_e(version)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-dark: #0B2545; --primary-mid: #134074;
            --accent-blue: #3B82F6; --accent-green: #10B981;
            --accent-amber: #F59E0B; --accent-red: #EF4444;
            --surface: #F8FAFC; --card: #FFFFFF;
            --text-primary: #0F172A; --text-secondary: #475569;
            --border: #E2E8F0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}
        body {{ background-color: var(--surface); color: var(--text-primary); }}
        .app-container {{ display: flex; min-height: 100vh; flex-direction: column; }}
        header {{ background: var(--primary-dark); color: white; padding: 1.5rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); z-index: 10; }}
        .header-left h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; }}
        .breadcrumb {{ color: #94A3B8; font-size: 0.875rem; }}
        .version-badge {{ background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.875rem; font-weight: 500; }}
        .main-content {{ display: flex; flex: 1; overflow: hidden; }}
        .tab-nav {{ background: white; padding: 0 2rem; border-bottom: 1px solid var(--border); display: flex; gap: 2rem; }}
        .tab-btn {{ background: none; border: none; padding: 1rem 0; font-size: 1rem; font-weight: 500; color: var(--text-secondary); cursor: pointer; border-bottom: 3px solid transparent; }}
        .tab-btn.active {{ color: var(--accent-blue); border-bottom-color: var(--accent-blue); }}
        .page {{ display: none; flex: 1; overflow-y: auto; }}
        .page.active {{ display: flex; }}
        .sidebar {{ width: 280px; background: white; border-right: 1px solid var(--border); padding: 1.5rem 0; overflow-y: auto; flex-shrink: 0; }}
        .nav-section-title {{ font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; padding: 0 1.5rem; margin-bottom: 0.75rem; margin-top: 1.5rem; }}
        .nav-item {{ padding: 0.5rem 1.5rem; cursor: pointer; display: flex; align-items: center; justify-content: space-between; font-size: 0.9rem; color: var(--text-primary); transition: background 0.2s; }}
        .nav-item:hover {{ background: var(--surface); }}
        .nav-item.active {{ background: #EFF6FF; color: var(--accent-blue); font-weight: 500; border-right: 3px solid var(--accent-blue); }}
        .nav-item-count {{ background: var(--surface); padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; color: var(--text-secondary); }}
        .content-area {{ flex: 1; padding: 2rem; background: var(--surface); overflow-y: auto; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        .metric-card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .metric-value {{ font-size: 2rem; font-weight: 700; color: var(--primary-dark); margin-bottom: 0.25rem; }}
        .metric-label {{ font-size: 0.875rem; color: var(--text-secondary); font-weight: 500; }}
        .section-block {{ display: none; animation: fadeIn 0.3s ease; }}
        .section-block.active {{ display: block; }}
        h2 {{ font-size: 1.5rem; font-weight: 600; color: var(--primary-dark); margin-bottom: 1.5rem; }}
        h3 {{ font-size: 1.125rem; font-weight: 600; color: var(--text-primary); margin-bottom: 1rem; margin-top: 1.5rem; }}
        .concept-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px,1fr)); gap: 1.5rem; }}
        .concept-card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .concept-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }}
        .concept-name {{ font-size: 1.25rem; font-weight: 600; color: var(--primary-dark); }}
        .data-type {{ background: #F1F5F9; color: #475569; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; font-family: monospace; }}
        .concept-def {{ font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem; line-height: 1.5; }}
        .attr-group {{ margin-bottom: 0.75rem; }}
        .attr-label {{ font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.25rem; display: block; }}
        .pill {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; background: #EFF6FF; color: var(--accent-blue); border: 1px solid #BFDBFE; margin: 0.1rem 0.25rem 0.1rem 0; }}
        .pill.alias {{ background: #F8FAFC; color: var(--text-secondary); border-color: var(--border); }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid var(--border); margin-bottom: 1.5rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
        th {{ background: #F8FAFC; font-weight: 600; color: var(--text-secondary); }}
        .score-container {{ max-width: 1200px; margin: 0 auto; width: 100%; }}
        .dashboard-grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 2rem; margin-bottom: 2rem; }}
        .score-card {{ background: white; border-radius: 16px; padding: 2rem; border: 1px solid var(--border); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .score-number {{ font-size: 4rem; font-weight: 700; line-height: 1; margin: 1rem 0; }}
        .score-red {{ color: var(--accent-red); }} .score-yellow {{ color: var(--accent-amber); }} .score-green {{ color: var(--accent-green); }}
        .mat-badge {{ background: var(--primary-dark); color: white; padding: 0.5rem 1.5rem; border-radius: 999px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.875rem; margin-top: 1rem; }}
        .chart-card {{ background: white; border-radius: 16px; padding: 1.5rem; border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .gap-card {{ background: white; border-radius: 8px; border-left: 4px solid var(--border); padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
        .gap-critical {{ border-left-color: var(--accent-red); }}
        .gap-major {{ border-left-color: var(--accent-amber); }}
        .gap-minor {{ border-left-color: var(--accent-green); }}
        .rec-item {{ background: white; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; border: 1px solid var(--border); display: flex; gap: 1rem; }}
        .rec-number {{ width: 32px; height: 32px; background: #EFF6FF; color: var(--accent-blue); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @media print {{
            .sidebar, .tab-nav {{ display: none !important; }}
            .page {{ display: block !important; overflow: visible !important; }}
            .section-block {{ display: block !important; margin-bottom: 2rem; page-break-inside: avoid; }}
            body {{ background: white; }}
        }}
    </style>
</head>
<body>
    <div class="app-container">
        <header>
            <div class="header-left">
                <h1>{_e(title)}</h1>
                <div class="breadcrumb">{_e(platform)} &gt; {_e(domain)}</div>
            </div>
            <div class="header-right">
                <span class="version-badge">v{_e(version)}</span>
            </div>
        </header>
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('p1', this)">AS IS Taxonomy</button>
            <button class="tab-btn" onclick="switchTab('p2', this)">Score &amp; Delta</button>
        </div>
        <div class="main-content">
            <div id="p1" class="page active">
                <div class="sidebar">
                    <div class="nav-section-title">Visão Geral</div>
                    <div class="nav-item active" onclick="showSection('overview', this)"><span>📊 Dashboard</span></div>
                    <div class="nav-item" onclick="showSection('naming', this)"><span>📝 Naming Conventions</span></div>
                    <div class="nav-section-title">Grupos de Conceitos</div>
""")

    for idx, group in enumerate(groups):
        g_name = group.get("name", "") if isinstance(group, dict) else ""
        g_icon = group.get("icon", "📄") if isinstance(group, dict) else "📄"
        g_count = len(group.get("concepts", []) or []) if isinstance(group, dict) else 0
        parts.append(
            f"""                    <div class="nav-item" onclick="showSection('group-{idx}', this)">
                        <span>{_e(g_icon)} {_e(g_name)}</span>
                        <span class="nav-item-count">{g_count}</span>
                    </div>
"""
        )

    parts.append("""
                    <div class="nav-section-title">Regras &amp; Padrões</div>
                    <div class="nav-item" onclick="showSection('context', this)"><span>⚖️ Context-Based Rules</span></div>
                    <div class="nav-item" onclick="showSection('aliases', this)"><span>⚠️ Aliases Ambíguos</span></div>
                    <div class="nav-item" onclick="showSection('lake', this)"><span>🌊 Lake Standards</span></div>
                </div>
                <div class="content-area">
                    <div id="overview" class="section-block active">
                        <h2>Dashboard Taxonômico</h2>
                        <div class="metrics-grid">
""")
    parts.append(
        f"""                            <div class="metric-card"><div class="metric-value">{_e(total_concepts)}</div><div class="metric-label">Conceitos Mapeados</div></div>
                            <div class="metric-card"><div class="metric-value">{_e(total_aliases)}</div><div class="metric-label">Aliases Tratados</div></div>
                            <div class="metric-card"><div class="metric-value">{_e(total_groups)}</div><div class="metric-label">Domínios de Dados</div></div>
                        </div>
                        <h3>Instruções para Agentes de IA</h3>
                        <div style="background:#F8FAFC; border:1px solid var(--border); padding:1.5rem; border-radius:8px;">
                            <p style="margin-bottom:1rem; color:var(--text-secondary);">{_e(_ai_desc)}</p>
                            <ul style="margin-left:1.5rem; color:var(--primary-dark); font-weight:500;">
"""
    )

    for qr in (taxonomy_data.get("ai_agent_instructions") or {}).get("quick_rules", []) or []:
        parts.append(f"                                <li style='margin-bottom:0.5rem;'>{_e(qr)}</li>\n")

    parts.append("""                            </ul>
                        </div>
                    </div>
                    <div id="naming" class="section-block">
                        <h2>Naming Conventions &amp; Princípios</h2>
""")

    nc = taxonomy_data.get("naming_conventions", {}) or {}
    if nc:
        casing = nc.get("casing_rules", {}) or {}
        parts.append("<h3>Casing Rules</h3><table><tr><th>Object Type</th><th>Casing Rule</th><th>Examples</th></tr>")
        examples = casing.get("examples", {}) or {}
        for key, label in (("columns", "Columns"), ("entities", "Entities/Tables"), ("ctes", "CTEs")):
            if key in casing:
                ex_key = {"columns": "column", "entities": "entity", "ctes": "cte"}[key]
                ex_list = examples.get(ex_key, []) or []
                parts.append(
                    f"<tr><td>{_e(label)}</td><td><span class='data-type'>{_e(casing[key])}</span></td>"
                    f"<td>{_e(', '.join(str(e) for e in ex_list))}</td></tr>"
                )
        parts.append("</table>")

        acronyms = nc.get("canonical_acronyms", {}) or {}
        parts.append(
            f"<h3>Canonical Acronyms</h3><p style='margin-bottom:1rem;color:var(--text-secondary);'>"
            f"{_e(acronyms.get('rule', ''))}</p><div>"
        )
        parts.append(_join_pills(acronyms.get("items", []) or []))
        parts.append("</div>")

        forbidden = nc.get("forbidden_forms", {}) or {}
        parts.append("<h3 style='margin-top:2rem;'>Forbidden Forms</h3><table><tr><th>Pattern</th><th>Reason</th></tr>")
        for item in forbidden.get("items", []) or []:
            if isinstance(item, dict):
                parts.append(
                    f"<tr><td><span style='color:var(--accent-red);font-weight:600;'>✗ "
                    f"{_e(item.get('pattern'))}</span></td><td>{_e(item.get('reason'))}</td></tr>"
                )
        parts.append("</table>")

    parts.append("</div>")

    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        g_name = group.get("name", "")
        g_desc = group.get("description", "")
        concepts = group.get("concepts", []) or []
        parts.append(
            f"""<div id="group-{idx}" class="section-block">
                <h2>{_e(g_name)}</h2>
                <p style="color:var(--text-secondary); margin-bottom:2rem;">{_e(g_desc)}</p>
                <div class="concept-grid">
"""
        )
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            c_name = concept.get("name", "")
            c_type = concept.get("data_type", "")
            c_def = concept.get("definition", "")
            c_accepted = concept.get("accepted_types", []) or []
            c_aliases = concept.get("aliases", []) or []
            c_eqf = concept.get("entity_qualified_forms", {}) or {}

            acc_html = _join_pills(c_accepted)
            if c_aliases:
                alias_html = _join_pills(c_aliases, cls="pill alias")
            else:
                alias_html = "<span style='font-size:0.8rem;color:#94A3B8;'>Nenhum mapeado</span>"

            eqf_rows = []
            for k, v in c_eqf.items():
                label = str(k).replace("_", " ").title()
                eqf_rows.append(
                    f"<tr><td style='padding:0.25rem 0;font-size:0.8rem;'>{_e(label)}</td>"
                    f"<td style='padding:0.25rem 0;font-weight:600;font-size:0.8rem;'>{_e(v)}</td></tr>"
                )
            eqf_html = "<table>" + "".join(eqf_rows) + "</table>"

            parts.append(
                f"""<div class="concept-card">
                    <div class="concept-header">
                        <div class="concept-name">{_e(c_name)}</div>
                        <div class="data-type">{_e(c_type)}</div>
                    </div>
                    <div class="concept-def">{_e(c_def)}</div>
                    <div class="attr-group">
                        <span class="attr-label">Tipos Aceitos</span>
                        {acc_html}
                    </div>
                    <div class="attr-group" style="margin-top:1rem;">
                        <span class="attr-label">Entity Qualified Forms</span>
                        {eqf_html}
                    </div>
                    <div class="attr-group" style="margin-top:1rem; border-top:1px dashed var(--border); padding-top:1rem;">
                        <span class="attr-label">Source Mappings / Aliases</span>
                        {alias_html}
                    </div>
                </div>"""
            )
        parts.append("</div></div>")

    parts.append('<div id="context" class="section-block"><h2>Context-Based Rules</h2>')
    for rule in taxonomy_data.get("context_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        parts.append(
            f"""<div style="background:white; border:1px solid var(--border); border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; border-left:4px solid var(--accent-blue);">
                <h3>{_e(rule.get('title', ''))}</h3>
                <p style="font-weight:500; margin-bottom:0.5rem;">{_e(rule.get('subtitle', ''))}</p>
                <p style="color:var(--text-secondary); margin-bottom:1rem;">{_e(rule.get('definition', ''))}</p>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
                    <div>
                        <h4 style="font-size:0.875rem; color:var(--accent-green); margin-bottom:0.5rem;">✓ Examples</h4>
                        <ul style="font-size:0.875rem; margin-left:1.5rem;">
"""
        )
        for k, v in (rule.get("examples", {}) or {}).items():
            v_list = v if isinstance(v, list) else [v]
            parts.append(f"<li><strong>{_e(k)}:</strong> {_e(', '.join(str(x) for x in v_list))}</li>")
        parts.append('</ul></div><div><h4 style="font-size:0.875rem; color:var(--accent-red); margin-bottom:0.5rem;">✗ Anti-patterns</h4><ul style="font-size:0.875rem; margin-left:1.5rem;">')
        for k, v in (rule.get("anti_patterns", {}) or {}).items():
            v_list = v if isinstance(v, list) else [v]
            parts.append(f"<li><strong>{_e(k)}:</strong> {_e(', '.join(str(x) for x in v_list))}</li>")
        parts.append(
            f'</ul></div></div><div style="margin-top:1rem; background:#F8FAFC; padding:1rem; border-radius:8px; font-size:0.875rem;">'
            f'<strong>Rationale:</strong> {_e(rule.get("rationale", ""))}</div></div>'
        )
    parts.append("</div>")

    parts.append('<div id="aliases" class="section-block"><h2>Aliases Ambíguos</h2><div style="display:grid; gap:1rem;">')
    for alias in taxonomy_data.get("ambiguous_aliases", []) or []:
        if not isinstance(alias, dict):
            continue
        parts.append(
            f"""<div style="background:white; border:1px solid var(--border); border-radius:8px; padding:1.25rem;">
                <h3 style="margin-top:0; color:var(--accent-amber);">{_e(alias.get('name'))}</h3>
                <p style="color:var(--text-secondary); margin-bottom:1rem;">{_e(alias.get('description'))}</p>
                <table><tr><th style="width:150px;">Contexto</th><th>Resolução</th></tr>
"""
        )
        for k, v in (alias.get("resolution_rules", {}) or {}).items():
            parts.append(f"<tr><td><span class='data-type'>{_e(k)}</span></td><td>{_e(v)}</td></tr>")
        parts.append("</table></div>")
    parts.append("</div></div>")

    lake = taxonomy_data.get("lake_standards", {}) or {}
    zones = lake.get("zones", []) or []
    parts.append(
        '<div id="lake" class="section-block"><h2>Lake / Platform Standards</h2>'
        '<div style="display:flex; gap:1.5rem; margin-bottom:2rem;">'
    )
    zone_colors = ["#FCD34D", "#94A3B8", "#FBBF24"]
    for i, z in enumerate(zones):
        if not isinstance(z, dict):
            continue
        color = zone_colors[i % len(zone_colors)]
        parts.append(
            f"""<div style="flex:1; background:white; border:1px solid var(--border); border-top:4px solid {color}; border-radius:8px; padding:1.5rem;">
                <h3>{_e(str(z.get('name','')).title())} <span style="font-weight:400; color:var(--text-secondary);">({_e(z.get('alias',''))})</span></h3>
                <p style="font-size:0.875rem; margin-bottom:1rem;">{_e(z.get('description',''))}</p>
                <div style="font-family:monospace; background:#F8FAFC; padding:0.5rem; border-radius:4px; font-size:0.875rem; text-align:center;">{_e(z.get('naming_pattern',''))}</div>
            </div>"""
        )
    parts.append("</div></div></div></div>")

    # Page 2 — Score & Delta
    parts.append(
        f"""<div id="p2" class="page">
            <div class="content-area">
                <div class="score-container">
                    <h2>Taxonomy Maturity Evaluation</h2>
                    <div class="dashboard-grid">
                        <div class="score-card">
                            <h3 style="margin-top:0;">Overall Score</h3>
                            <div class="score-number {score_color}">{_e(overall_score)}</div>
                            <div style="color:var(--text-secondary); font-weight:500;">/ 100 points</div>
                            <div class="mat-badge">Level {_e(mat_level)} : {_e(mat_label)}</div>
                        </div>
                        <div class="chart-card">
                            <h3 style="margin-top:0;">Dimension Analysis</h3>
                            <div style="position:relative; height:300px; width:100%; display:flex; justify-content:center;">
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="dashboard-grid">
                        <div class="chart-card">
                            <h3 style="margin-top:0;">Industry Benchmark Comparison</h3>
                            <div style="position:relative; height:300px; width:100%;">
                                <canvas id="barChart"></canvas>
                            </div>
                        </div>
                        <div class="chart-card" style="overflow-y:auto; max-height:400px;">
                            <h3 style="margin-top:0;">Dimension Findings</h3>
                            <table>"""
    )
    for dim, finds in (score_data.get("dimension_findings", {}) or {}).items():
        parts.append(
            f"<tr><td style='font-weight:600;width:30%;'>{_e(str(dim).replace('_', ' ').title())}</td><td>"
        )
        for f_msg in finds or []:
            parts.append(f"<div style='margin-bottom:0.25rem;'>• {_e(f_msg)}</div>")
        parts.append("</td></tr>")
    parts.append("</table></div></div>")

    parts.append('<h2>Gap Analysis</h2><div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); gap:1.5rem; margin-bottom:2rem;">')
    for gap in score_data.get("gaps", []) or []:
        sev = gap.get("severity", "minor")
        parts.append(
            f"""<div class="gap-card gap-{_e(sev)}">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                    <span style="font-weight:600; font-size:0.875rem;">{_e(str(gap.get('dimension', '')).replace('_', ' ').title())}</span>
                    <span style="text-transform:uppercase; font-size:0.75rem; font-weight:700; color:var(--text-secondary);">{_e(sev)}</span>
                </div>
                <p style="font-size:0.875rem; color:var(--text-secondary); margin-bottom:0.5rem;"><strong>Current:</strong> {_e(gap.get('current_state', ''))}</p>
                <p style="font-size:0.875rem; color:var(--primary-dark);"><strong>Target:</strong> {_e(gap.get('target_state', ''))}</p>
            </div>"""
        )
    parts.append("</div>")

    parts.append('<h2>Recommendations (Delta TO-BE)</h2><div style="margin-bottom:3rem;">')
    for rec in score_data.get("recommendations", []) or []:
        parts.append(
            f"""<div class="rec-item">
                <div class="rec-number">{_e(rec.get('priority', ''))}</div>
                <div style="flex:1;">
                    <h3 style="margin:0 0 0.5rem 0; font-size:1.125rem;">{_e(rec.get('title', ''))}</h3>
                    <p style="color:var(--text-secondary); margin-bottom:1rem; font-size:0.95rem;">{_e(rec.get('description', ''))}</p>
                    <div style="display:flex; gap:1rem;">
                        <span class="pill" style="margin:0;">+{_e(rec.get('expected_score_impact', ''))} pts</span>
                        <span class="pill alias" style="margin:0;">Effort: {_e(rec.get('effort', ''))}</span>
                        <span class="pill alias" style="margin:0;">Timeline: {_e(rec.get('timeline', ''))}</span>
                    </div>
                </div>
            </div>"""
        )
    parts.append("</div></div></div></div></div></div>")

    dim_labels = list((score_data.get("dimension_scores", {}) or {}).keys())
    chart_labels = _safe_json([str(l).replace("_", " ").title() for l in dim_labels])
    chart_data = _safe_json(list((score_data.get("dimension_scores", {}) or {}).values()))
    bench_data = _safe_json(
        [(score_data.get("benchmark_comparison", {}) or {}).get(d, 50) for d in dim_labels]
    )

    parts.append(
        f"""<script>
        function switchTab(tabId, btn) {{
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
            if (tabId === 'p2' && !window.chartsRendered) {{
                renderCharts();
                window.chartsRendered = true;
            }}
        }}
        function showSection(sectionId, navItem) {{
            document.querySelectorAll('.section-block').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            navItem.classList.add('active');
            document.querySelector('.content-area').scrollTop = 0;
        }}
        function renderCharts() {{
            const labels = {chart_labels};
            const scores = {chart_data};
            const bench = {bench_data};
            new Chart(document.getElementById('radarChart'), {{
                type: 'radar',
                data: {{ labels: labels, datasets: [{{
                    label: 'Current Score', data: scores,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: '#3B82F6',
                    pointBackgroundColor: '#3B82F6',
                    borderWidth: 2
                }}]}},
                options: {{ responsive: true, maintainAspectRatio: false,
                    scales: {{ r: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
            new Chart(document.getElementById('barChart'), {{
                type: 'bar',
                data: {{ labels: labels, datasets: [
                    {{ label: 'Your Score', data: scores, backgroundColor: '#3B82F6', borderRadius: 4 }},
                    {{ label: 'Industry Benchmark', data: bench, backgroundColor: '#94A3B8', borderRadius: 4 }}
                ] }},
                options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    scales: {{ x: {{ min: 0, max: 100 }} }}
                }}
            }});
        }}
    </script>
</body>
</html>"""
    )
    return "".join(parts)
