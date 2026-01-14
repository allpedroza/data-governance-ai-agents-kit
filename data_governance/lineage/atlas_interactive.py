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
Visualização Atlas Interativa com Expansão de Níveis
Permite explorar o grafo de linhagem de forma hierárquica
"""

import networkx as nx
from typing import Dict, Set, List, Optional
import plotly.graph_objects as go
import json
from i18n import get_i18n


class AtlasInteractiveVisualization:
    """Visualização interativa estilo Apache Atlas com expansão de níveis"""

    def __init__(self, graph: nx.DiGraph, language: str = 'pt'):
        """
        Inicializa a visualização

        Args:
            graph: Grafo NetworkX
            language: Idioma para a interface ('pt' ou 'en')
        """
        self.graph = graph
        self.i18n = get_i18n()
        self.i18n.set_language(language)

        # Cores por tipo
        self.type_colors = {
            'table': '#1f77b4',
            'view': '#ff7f0e',
            'file': '#2ca02c',
            'stream': '#d62728',
            'dataset': '#9467bd',
            'terraform_resource': '#8c564b',
            'databricks_table': '#e377c2',
            'airflow_task': '#7f7f7f',
            'delta_table': '#bcbd22',
            'unknown': '#17becf'
        }

    def get_nodes_at_level(self, start_node: str, levels: int = 1,
                          direction: str = 'both') -> Set[str]:
        """
        Obtém nós a N níveis de distância de um nó inicial

        Args:
            start_node: Nó inicial
            levels: Número de níveis para expandir
            direction: 'upstream', 'downstream' ou 'both'

        Returns:
            Set de nós encontrados
        """
        if start_node not in self.graph:
            return set()

        nodes = {start_node}

        for _ in range(levels):
            new_nodes = set()
            for node in nodes:
                if direction in ['upstream', 'both']:
                    # Adiciona predecessores
                    new_nodes.update(self.graph.predecessors(node))
                if direction in ['downstream', 'both']:
                    # Adiciona sucessores
                    new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)

        return nodes

    def create_subgraph_view(self, nodes: Set[str]) -> nx.DiGraph:
        """Cria um subgrafo contendo apenas os nós especificados"""
        return self.graph.subgraph(nodes).copy()

    def generate_html(self, output_file: str, initial_nodes: Optional[List[str]] = None,
                     initial_levels: int = 2, title: str = None):
        """
        Gera HTML interativo com controles de expansão

        Args:
            output_file: Caminho do arquivo HTML de saída
            initial_nodes: Nós iniciais (None = todos os nós mais importantes)
            initial_levels: Número inicial de níveis a mostrar
            title: Título da visualização
        """
        if title is None:
            title = self.i18n.t('atlas_view')

        # Se não especificado, seleciona nós mais importantes
        if initial_nodes is None:
            # Pega nós com mais conexões
            degrees = dict(self.graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            initial_nodes = [node for node, _ in sorted_nodes[:min(10, len(sorted_nodes))]]

        # Prepara dados do grafo completo para JavaScript
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            nodes_data.append({
                'id': node,
                'type': node_type,
                'color': self.type_colors.get(node_type, self.type_colors['unknown']),
                'in_degree': self.graph.in_degree(node),
                'out_degree': self.graph.out_degree(node),
                'source_file': data.get('source_file', 'N/A')
            })

        edges_data = []
        for source, target, data in self.graph.edges(data=True):
            edges_data.append({
                'source': source,
                'target': target,
                'operation': data.get('operation', 'N/A')
            })

        # Cria HTML com controles interativos
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        #header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
        }}
        #controls {{
            background: white;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        button {{
            padding: 10px 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #2980b9;
        }}
        button.secondary {{
            background: #95a5a6;
        }}
        button.secondary:hover {{
            background: #7f8c8d;
        }}
        select, input[type="number"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        #graph {{
            width: 100%;
            height: calc(100vh - 200px);
        }}
        #info {{
            background: white;
            padding: 15px;
            margin: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info-item {{
            display: inline-block;
            margin-right: 20px;
            color: #7f8c8d;
        }}
        .info-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🗺️ {title}</h1>
    </div>

    <div id="controls">
        <div class="control-group">
            <label>{self.i18n.t('expand_level')}:</label>
            <input type="number" id="levels" value="{initial_levels}" min="1" max="10" />
            <button onclick="updateLevels()">{self.i18n.t('view_visualization')}</button>
        </div>

        <div class="control-group">
            <label>Foco:</label>
            <select id="focusNode" onchange="updateGraph()">
                <option value="">-- Todos os principais --</option>
"""

        # Adiciona opções de nós importantes
        for node in initial_nodes[:20]:
            html_content += f'                <option value="{node}">{node[:50]}</option>\n'

        html_content += f"""
            </select>
        </div>

        <div class="control-group">
            <button onclick="expandUpstream()" class="secondary">⬅️ {self.i18n.t('show_upstream')}</button>
            <button onclick="expandDownstream()" class="secondary">➡️ {self.i18n.t('show_downstream')}</button>
        </div>

        <div class="control-group">
            <button onclick="resetView()" class="secondary">🔄 {self.i18n.t('reset_view')}</button>
        </div>
    </div>

    <div id="info">
        <span class="info-item">{self.i18n.t('total_assets')}: <span class="info-value" id="nodeCount">0</span></span>
        <span class="info-item">{self.i18n.t('transformations')}: <span class="info-value" id="edgeCount">0</span></span>
        <span class="info-item">Níveis: <span class="info-value" id="currentLevels">{initial_levels}</span></span>
    </div>

    <div id="graph"></div>

    <script>
        // Dados do grafo
        const allNodes = {json.dumps(nodes_data)};
        const allEdges = {json.dumps(edges_data)};
        const initialNodes = {json.dumps(initial_nodes)};

        let currentFocusNode = null;
        let currentLevels = {initial_levels};
        let visibleNodes = new Set();

        function getNodesAtLevel(startNodes, levels, direction = 'both') {{
            const nodes = new Set(startNodes);
            const edges = new Map();

            // Cria mapa de adjacência
            allEdges.forEach(edge => {{
                if (!edges.has(edge.source)) edges.set(edge.source, {{ upstream: [], downstream: [] }});
                if (!edges.has(edge.target)) edges.set(edge.target, {{ upstream: [], downstream: [] }});
                edges.get(edge.target).upstream.push(edge.source);
                edges.get(edge.source).downstream.push(edge.target);
            }});

            for (let i = 0; i < levels; i++) {{
                const newNodes = new Set();
                nodes.forEach(node => {{
                    const adj = edges.get(node);
                    if (adj) {{
                        if (direction === 'upstream' || direction === 'both') {{
                            adj.upstream.forEach(n => newNodes.add(n));
                        }}
                        if (direction === 'downstream' || direction === 'both') {{
                            adj.downstream.forEach(n => newNodes.add(n));
                        }}
                    }}
                }});
                newNodes.forEach(n => nodes.add(n));
            }}

            return Array.from(nodes);
        }}

        function updateGraph() {{
            const focusSelect = document.getElementById('focusNode');
            currentFocusNode = focusSelect.value;

            let nodesToShow;
            if (currentFocusNode) {{
                nodesToShow = getNodesAtLevel([currentFocusNode], currentLevels, 'both');
            }} else {{
                nodesToShow = getNodesAtLevel(initialNodes, currentLevels, 'both');
            }}

            visibleNodes = new Set(nodesToShow);
            renderGraph(nodesToShow);
        }}

        function renderGraph(nodesToShow) {{
            const nodeSet = new Set(nodesToShow);
            const visibleEdges = allEdges.filter(e =>
                nodeSet.has(e.source) && nodeSet.has(e.target)
            );

            // Atualiza info
            document.getElementById('nodeCount').textContent = nodesToShow.length;
            document.getElementById('edgeCount').textContent = visibleEdges.length;
            document.getElementById('currentLevels').textContent = currentLevels;

            // Prepara layout hierárquico simples
            const nodeMap = new Map();
            nodesToShow.forEach((nodeId, index) => {{
                const nodeData = allNodes.find(n => n.id === nodeId);
                if (nodeData) {{
                    nodeMap.set(nodeId, nodeData);
                }}
            }});

            // Calcula posições (layout hierárquico simples)
            const levels = assignLevels(nodesToShow, visibleEdges);
            const positions = calculatePositions(levels);

            // Prepara traces por tipo
            const tracesByType = {{}};

            nodesToShow.forEach(nodeId => {{
                const nodeData = nodeMap.get(nodeId);
                if (!nodeData) return;

                const nodeType = nodeData.type;
                if (!tracesByType[nodeType]) {{
                    tracesByType[nodeType] = {{
                        x: [],
                        y: [],
                        text: [],
                        hovertext: [],
                        mode: 'markers+text',
                        type: 'scatter',
                        name: nodeType,
                        marker: {{
                            size: 25,
                            color: nodeData.color,
                            line: {{ width: 2, color: 'white' }}
                        }},
                        textposition: 'top center',
                        textfont: {{ size: 10, family: 'Arial Black' }},
                        hoverinfo: 'text'
                    }};
                }}

                const pos = positions.get(nodeId) || {{ x: 0, y: 0 }};
                const label = nodeId.length > 30 ? nodeId.substring(0, 30) + '...' : nodeId;
                const hoverText = `<b>${{nodeId}}</b><br><b>Type:</b> ${{nodeType}}<br><b>In:</b> ${{nodeData.in_degree}} | <b>Out:</b> ${{nodeData.out_degree}}`;

                tracesByType[nodeType].x.push(pos.x);
                tracesByType[nodeType].y.push(pos.y);
                tracesByType[nodeType].text.push(label);
                tracesByType[nodeType].hovertext.push(hoverText);
            }});

            // Prepara edges
            const edgeTrace = {{
                x: [],
                y: [],
                mode: 'lines',
                line: {{ width: 2, color: '#888' }},
                hoverinfo: 'none',
                showlegend: false
            }};

            visibleEdges.forEach(edge => {{
                const sourcePos = positions.get(edge.source);
                const targetPos = positions.get(edge.target);
                if (sourcePos && targetPos) {{
                    edgeTrace.x.push(sourcePos.x, targetPos.x, null);
                    edgeTrace.y.push(sourcePos.y, targetPos.y, null);
                }}
            }});

            // Combina traces
            const data = [edgeTrace, ...Object.values(tracesByType)];

            const layout = {{
                title: '{title}',
                showlegend: true,
                hovermode: 'closest',
                margin: {{ b: 40, l: 40, r: 200, t: 100 }},
                xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
                yaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
                paper_bgcolor: 'white',
                plot_bgcolor: '#fafafa'
            }};

            Plotly.newPlot('graph', data, layout, {{ responsive: true }});
        }}

        function assignLevels(nodes, edges) {{
            // Atribui níveis hierárquicos aos nós
            const levels = new Map();
            const inDegree = new Map();

            nodes.forEach(node => {{
                inDegree.set(node, 0);
                levels.set(node, 0);
            }});

            edges.forEach(edge => {{
                if (inDegree.has(edge.target)) {{
                    inDegree.set(edge.target, inDegree.get(edge.target) + 1);
                }}
            }});

            // Topological sort
            const queue = nodes.filter(node => inDegree.get(node) === 0);
            queue.forEach(node => levels.set(node, 0));

            const processed = new Set();
            while (queue.length > 0) {{
                const node = queue.shift();
                processed.add(node);
                const nodeLevel = levels.get(node);

                edges.forEach(edge => {{
                    if (edge.source === node && nodes.includes(edge.target)) {{
                        levels.set(edge.target, Math.max(levels.get(edge.target), nodeLevel + 1));
                        const newDegree = inDegree.get(edge.target) - 1;
                        inDegree.set(edge.target, newDegree);
                        if (newDegree === 0 && !processed.has(edge.target)) {{
                            queue.push(edge.target);
                        }}
                    }}
                }});
            }}

            return levels;
        }}

        function calculatePositions(levels) {{
            const positions = new Map();
            const levelNodes = new Map();

            // Agrupa nós por nível
            levels.forEach((level, node) => {{
                if (!levelNodes.has(level)) levelNodes.set(level, []);
                levelNodes.get(level).push(node);
            }});

            // Calcula posições
            const maxLevel = Math.max(...levels.values());
            const spacing = 1000 / Math.max(maxLevel, 1);
            const verticalSpacing = 100;

            levelNodes.forEach((nodes, level) => {{
                const x = level * spacing;
                const totalHeight = (nodes.length - 1) * verticalSpacing;
                const startY = -totalHeight / 2;

                nodes.forEach((node, index) => {{
                    positions.set(node, {{
                        x: x,
                        y: startY + (index * verticalSpacing)
                    }});
                }});
            }});

            return positions;
        }}

        function updateLevels() {{
            currentLevels = parseInt(document.getElementById('levels').value);
            updateGraph();
        }}

        function expandUpstream() {{
            if (currentFocusNode) {{
                const nodesToShow = getNodesAtLevel([currentFocusNode], currentLevels + 1, 'upstream');
                visibleNodes = new Set(nodesToShow);
                renderGraph(nodesToShow);
            }} else {{
                alert('Selecione um nó de foco primeiro');
            }}
        }}

        function expandDownstream() {{
            if (currentFocusNode) {{
                const nodesToShow = getNodesAtLevel([currentFocusNode], currentLevels + 1, 'downstream');
                visibleNodes = new Set(nodesToShow);
                renderGraph(nodesToShow);
            }} else {{
                alert('Selecione um nó de foco primeiro');
            }}
        }}

        function resetView() {{
            currentFocusNode = null;
            currentLevels = {initial_levels};
            document.getElementById('focusNode').value = '';
            document.getElementById('levels').value = {initial_levels};
            updateGraph();
        }}

        // Renderiza grafo inicial
        updateGraph();
    </script>
</body>
</html>
"""

        # Salva arquivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_file
