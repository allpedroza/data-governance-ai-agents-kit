"""
Gerador de grafo interativo de arquitetura (Cytoscape.js).

Adaptado de: https://github.com/madrade1472/cadeia_de_graficos_arquitetura
Autor original: Marcus Andrade (github.com/madrade1472)

Converte uma definição de arquitetura em camadas (layers/components)
para o formato JSON compatível com Cytoscape.js (nodes + edges).

Este módulo é usado exclusivamente como biblioteca — sem CLI.
"""

import re
from typing import Dict, List, Set, Tuple

DEFAULT_COLORS = [
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#dc2626",
    "#0891b2",
    "#854d0e",
    "#475569",
]

_OUTPUT_TYPES = {"source", "api", "process"}
_INPUT_TYPES = {"process", "store", "api"}


def _safe_id(text: str, prefix: str, index: int) -> str:
    """Gera ID seguro a partir de texto."""
    slug = re.sub(r"[^a-z0-9]", "_", text.lower())[:14].strip("_")
    return f"{prefix}_{index}_{slug}"


def generate(arch: dict, custom_colors: list | None = None) -> dict:
    """
    Recebe o dict de arquitetura e retorna o dict Cytoscape.js
    com nodes e edges prontos para o viewer.

    Args:
        arch: dict no formato {project_name, layers: [{id, name, color, components, connections_to}]}
        custom_colors: lista opcional de cores hex para substituir DEFAULT_COLORS

    Returns:
        dict com {nodes, edges, project_name}
    """
    colors = custom_colors or DEFAULT_COLORS
    layers = arch.get("layers", [])
    project_name = arch.get("project_name", "Arquitetura")

    nodes: list[dict] = []
    edges: list[dict] = []
    edge_ids: set[str] = set()

    def _add_edge(src: str, tgt: str, etype: str, color: str) -> None:
        eid = f"e_{src}__{tgt}"
        if eid in edge_ids or src == tgt:
            return
        edge_ids.add(eid)
        edges.append({
            "data": {"id": eid, "source": src, "target": tgt,
                     "type": etype, "color": color},
            "classes": f"{etype}-edge",
        })

    # Passo 1: montar nodes + lookup nome -> id
    name_to_id: dict[str, str] = {}
    layer_color: dict[str, str] = {}
    comp_records: list[tuple] = []  # (lid, cid, comp, color)

    for i, layer in enumerate(layers):
        color = layer.get("color") or colors[i % len(colors)]
        lid = layer.get("id") or f"layer_{i+1}"
        layer_color[lid] = color

        nodes.append({
            "data": {
                "id": lid,
                "label": layer.get("name", lid),
                "type": "layer",
                "color": color,
                "description": layer.get("description", ""),
            },
            "classes": "layer-node",
        })

        for j, comp in enumerate(layer.get("components", [])):
            cid = _safe_id(comp.get("name", f"comp_{j}"), lid, j)
            name_to_id[comp.get("name", "").lower().strip()] = cid
            comp_records.append((lid, cid, comp, color))

            nodes.append({
                "data": {
                    "id": cid,
                    "label": comp.get("name", ""),
                    "tech": comp.get("tech", ""),
                    "comp_type": comp.get("type", "process"),
                    "type": "component",
                    "color": color,
                    "parent_layer": lid,
                    "description": comp.get("description", ""),
                },
                "classes": "comp-node",
            })

    # Passo 2: edges de pertencimento (layer -> comp, dashed)
    for lid, cid, comp, color in comp_records:
        _add_edge(lid, cid, "member", color)

    # Passo 3: edges de fluxo entre camadas
    layer_ids = [layer.get("id") or f"layer_{i+1}" for i, layer in enumerate(layers)]
    connected_layers: set[tuple] = set()

    for i, layer in enumerate(layers):
        lid = layer_ids[i]
        targets = layer.get("connections_to", [])
        if targets:
            for tid in targets:
                if tid in layer_color:
                    _add_edge(lid, tid, "flow", layer_color[lid])
                    connected_layers.add((lid, tid))
        else:
            if i > 0:
                prev = layer_ids[i - 1]
                _add_edge(prev, lid, "flow", layer_color[lid])
                connected_layers.add((prev, lid))

    # Passo 4: edges comp -> comp vindos da definicao
    has_comp_connections = False
    for lid, cid, comp, color in comp_records:
        for target_name in comp.get("connections_to", []):
            target_id = name_to_id.get(target_name.lower().strip())
            if target_id and target_id != cid:
                _add_edge(cid, target_id, "comp-flow", color)
                has_comp_connections = True

    # Passo 5: fallback por tipo quando nao ha conexoes explicitas
    if not has_comp_connections:
        by_layer: dict[str, list] = {}
        for lid, cid, comp, color in comp_records:
            by_layer.setdefault(lid, []).append((cid, comp, color))

        for src_lid, tgt_lid in connected_layers:
            src_comps = by_layer.get(src_lid, [])
            tgt_comps = by_layer.get(tgt_lid, [])
            if not src_comps or not tgt_comps:
                continue

            src_cands = [(c, co, cl) for c, co, cl in src_comps
                         if co.get("type", "process") in _OUTPUT_TYPES] or src_comps[:1]
            tgt_cands = [(c, co, cl) for c, co, cl in tgt_comps
                         if co.get("type", "process") in _INPUT_TYPES] or tgt_comps[:1]

            for s_cid, _, s_col in src_cands[:2]:
                for t_cid, _, _ in tgt_cands[:2]:
                    _add_edge(s_cid, t_cid, "comp-flow", s_col)

    return {"nodes": nodes, "edges": edges, "project_name": project_name}
