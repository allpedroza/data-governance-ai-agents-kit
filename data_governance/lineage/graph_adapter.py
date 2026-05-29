"""
Graph Adapter — Ponte entre o DataLineageAgent e o gerador de grafos Cytoscape.js.

Converte os resultados da análise de linhagem (assets + transformações)
para o formato de definição de arquitetura em camadas, gera o JSON
Cytoscape.js via arch_generator.generate(), e renderiza HTML inline
para embedding no Streamlit.

Baseado em: https://github.com/madrade1472/cadeia_de_graficos_arquitetura
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from . import arch_generator

# ---------------------------------------------------------------------------
# Mapeamento tipo de asset → (layer_id, layer_name, component_type, cor)
# ---------------------------------------------------------------------------
_LAYER_MAP: Dict[str, tuple] = {
    # Fontes & Ingestão
    "file":            ("layer_source", "Fontes & Ingestão",     "source", "#2563eb"),
    "stream":          ("layer_source", "Fontes & Ingestão",     "source", "#2563eb"),
    "data_source":     ("layer_source", "Fontes & Ingestão",     "source", "#2563eb"),
    # Processamento & Orquestração
    "airflow_task":    ("layer_process", "Processamento & Orquestração", "process", "#9333ea"),
    # Dados & Persistência
    "table":           ("layer_data", "Dados & Persistência",    "store",  "#16a34a"),
    "view":            ("layer_data", "Dados & Persistência",    "store",  "#16a34a"),
    "databricks_table":("layer_data", "Dados & Persistência",    "store",  "#16a34a"),
    "delta_table":     ("layer_data", "Dados & Persistência",    "store",  "#16a34a"),
    "data_sink":       ("layer_data", "Dados & Persistência",    "store",  "#16a34a"),
    # Infra & Recursos
    "terraform_resource": ("layer_infra", "Infra & Recursos",    "infra",  "#ea580c"),
    "terraform_data":     ("layer_infra", "Infra & Recursos",    "infra",  "#ea580c"),
}

_DEFAULT_LAYER = ("layer_other", "Outros Componentes", "process", "#475569")

# Caminho do template viewer
_TEMPLATE_PATH = Path(__file__).parent / "viewer_template.html"


def _asset_to_layer_info(asset_type: str) -> tuple:
    """Retorna (layer_id, layer_name, comp_type, color) para um tipo de asset."""
    return _LAYER_MAP.get(asset_type, _DEFAULT_LAYER)


def lineage_to_arch_definition(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte os resultados do DataLineageAgent.analyze_pipeline()
    para o formato de definição de arquitetura (layers + components).

    Args:
        results: dict retornado por analyze_pipeline() com chaves:
                 assets, transformations, graph, metrics, etc.

    Returns:
        dict no formato arch_definition compatível com arch_generator.generate()
    """
    assets = results.get("assets", [])
    transformations = results.get("transformations", [])

    # Agrupa assets por layer
    layers_dict: Dict[str, Dict[str, Any]] = {}
    asset_name_to_layer: Dict[str, str] = {}

    for asset in assets:
        asset_name = asset.name if hasattr(asset, "name") else str(asset)
        asset_type = asset.type if hasattr(asset, "type") else "table"

        layer_id, layer_name, comp_type, color = _asset_to_layer_info(asset_type)
        asset_name_to_layer[asset_name] = layer_id

        if layer_id not in layers_dict:
            layers_dict[layer_id] = {
                "id": layer_id,
                "name": layer_name,
                "description": f"Camada: {layer_name}",
                "color": color,
                "components": [],
                "connections_to": [],
            }

        # Monta componente
        tech = ""
        description = ""
        if hasattr(asset, "metadata") and isinstance(asset.metadata, dict):
            tech = asset.metadata.get("operator", "")
            description = asset.metadata.get("description", "")
        if hasattr(asset, "source_file") and asset.source_file:
            if not description:
                description = f"Definido em: {Path(asset.source_file).name}"

        layers_dict[layer_id]["components"].append({
            "name": asset_name,
            "tech": tech or asset_type,
            "type": comp_type,
            "description": description,
            "connections_to": [],  # preenchido abaixo
        })

    # Mapeia transformações como connections_to entre componentes
    #  + connections_to entre layers
    connected_layer_pairs: set = set()

    for trans in transformations:
        src_name = trans.source.name if hasattr(trans, "source") else ""
        tgt_name = trans.target.name if hasattr(trans, "target") else ""

        if not src_name or not tgt_name or src_name == tgt_name:
            continue

        src_layer = asset_name_to_layer.get(src_name)
        tgt_layer = asset_name_to_layer.get(tgt_name)

        # Adiciona connections_to no componente fonte
        for comp in layers_dict.get(src_layer, {}).get("components", []):
            if comp["name"] == src_name:
                if tgt_name not in comp["connections_to"]:
                    comp["connections_to"].append(tgt_name)
                break

        # Registra conexão entre layers
        if src_layer and tgt_layer and src_layer != tgt_layer:
            connected_layer_pairs.add((src_layer, tgt_layer))

    # Preenche connections_to entre layers
    for src_lid, tgt_lid in connected_layer_pairs:
        if src_lid in layers_dict and tgt_lid not in layers_dict[src_lid]["connections_to"]:
            layers_dict[src_lid]["connections_to"].append(tgt_lid)

    # Ordena layers por prioridade lógica
    layer_order = ["layer_source", "layer_process", "layer_data", "layer_infra", "layer_other"]
    sorted_layers = []
    for lid in layer_order:
        if lid in layers_dict:
            sorted_layers.append(layers_dict[lid])
    # Adiciona quaisquer layers extras não previstos
    for lid, layer in layers_dict.items():
        if lid not in layer_order:
            sorted_layers.append(layer)

    return {
        "project_name": "Data Lineage",
        "layers": sorted_layers,
    }


def generate_cytoscape_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline completo: lineage results → arch_definition → Cytoscape.js JSON.

    Args:
        results: dict retornado por DataLineageAgent.analyze_pipeline()

    Returns:
        dict com {nodes, edges, project_name} pronto para o viewer
    """
    arch_def = lineage_to_arch_definition(results)
    return arch_generator.generate(arch_def)


def render_interactive_html(
    cytoscape_json: Dict[str, Any],
    height: int = 700,
) -> str:
    """
    Injeta o JSON Cytoscape.js no template viewer e retorna HTML completo.

    Args:
        cytoscape_json: dict com {nodes, edges, project_name}
        height: altura do iframe em pixels

    Returns:
        string HTML pronta para st.components.v1.html()
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    project_name = cytoscape_json.get("project_name", "Data Lineage")
    json_str = json.dumps(cytoscape_json, ensure_ascii=False)

    html = template.replace("__GRAPH_JSON__", json_str)
    html = html.replace("__PROJECT_NAME__", project_name)

    return html


def cytoscape_json_download(cytoscape_json: Dict[str, Any]) -> str:
    """Serializa o JSON Cytoscape.js para download."""
    return json.dumps(cytoscape_json, ensure_ascii=False, indent=2)
