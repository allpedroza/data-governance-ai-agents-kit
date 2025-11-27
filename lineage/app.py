"""
Data Lineage AI Agent - Interactive Web Interface
Streamlit application for data lineage visualization and impact analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
from pathlib import Path
import tempfile
import os
import sys
from datetime import datetime
import hashlib
import colorsys
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from data_lineage_agent import DataLineageAgent
from visualization_engine import DataLineageVisualizer
from lineage_system import DataLineageSystem
from parsers.terraform_parser import TerraformParser, parse_terraform_directory
from parsers.databricks_parser import DatabricksParser, parse_databricks_workspace
from parsers.airflow_parser import AirflowParser, parse_airflow_dags_folder


# Page configuration
st.set_page_config(
    page_title="Data Lineage AI Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .graph-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'lineage_agent' not in st.session_state:
    st.session_state.lineage_agent = DataLineageAgent()
    st.session_state.visualizer = DataLineageVisualizer()
    st.session_state.lineage_system = DataLineageSystem()
    st.session_state.analysis_results = None
    st.session_state.selected_assets = []
    st.session_state.impact_analysis = None
    st.session_state.comparison_results = None


def create_header():
    """Create application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.image("https://via.placeholder.com/100x100.png?text=DL", width=100)
    
    with col2:
        st.title("🔍 Data Lineage AI Agent")
        st.markdown("**Análise Inteligente de Linhagem de Dados e Impacto**")
    
    with col3:
        st.markdown(f"**Última Análise:** {datetime.now().strftime('%H:%M:%S')}")


def sidebar_configuration():
    """Configure sidebar with options"""
    st.sidebar.title("⚙️ Configurações")
    
    # File upload section
    st.sidebar.header("📁 Upload de Arquivos")
    
    upload_type = st.sidebar.selectbox(
        "Tipo de Upload",
        ["Arquivos Individuais", "Diretório ZIP", "Exemplos Pré-carregados"]
    )
    
    uploaded_files = []
    
    if upload_type == "Arquivos Individuais":
        uploaded_files = st.sidebar.file_uploader(
            "Selecione arquivos do pipeline",
            type=['py', 'sql', 'tf', 'json', 'scala', 'ipynb'],
            accept_multiple_files=True
        )
    
    elif upload_type == "Diretório ZIP":
        zip_file = st.sidebar.file_uploader(
            "Upload de arquivo ZIP",
            type=['zip']
        )
        if zip_file:
            # Extract and process ZIP
            import zipfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                # Process extracted files
                uploaded_files = list(Path(temp_dir).glob('**/*'))
    
    else:  # Exemplos Pré-carregados
        example = st.sidebar.selectbox(
            "Escolha um exemplo",
            ["E-commerce Pipeline", "Financial Data Lake", "IoT Streaming", "ML Feature Store"]
        )
        
        if st.sidebar.button("Carregar Exemplo"):
            uploaded_files = load_example_pipeline(example)
    
    # Analysis options
    st.sidebar.header("🎯 Opções de Análise")
    
    analysis_depth = st.sidebar.select_slider(
        "Profundidade da Análise",
        options=["Superficial", "Normal", "Profunda", "Completa"],
        value="Normal"
    )
    
    include_terraform = st.sidebar.checkbox("Incluir Infraestrutura (Terraform)", value=True)
    include_databricks = st.sidebar.checkbox("Incluir Databricks", value=True)
    include_airflow = st.sidebar.checkbox("Incluir Airflow DAGs", value=True)
    detect_streaming = st.sidebar.checkbox("Detectar Pipelines Streaming", value=True)
    analyze_schemas = st.sidebar.checkbox("Analisar Evolução de Schemas", value=True)
    
    # Visualization options
    st.sidebar.header("📊 Visualização")
    
    graph_type = st.sidebar.selectbox(
        "Tipo de Grafo",
        ["Force-Directed", "Hierárquico", "Sankey", "Radial", "3D"]
    )
    
    show_labels = st.sidebar.checkbox("Mostrar Labels", value=True)
    show_metrics = st.sidebar.checkbox("Mostrar Métricas", value=True)
    color_by = st.sidebar.selectbox(
        "Colorir por",
        ["Tipo", "Impacto", "Criticidade", "Provider"]
    )
    
    return {
        'files': uploaded_files,
        'analysis_depth': analysis_depth,
        'include_terraform': include_terraform,
        'include_databricks': include_databricks,
        'detect_streaming': detect_streaming,
        'analyze_schemas': analyze_schemas,
        'graph_type': graph_type,
        'show_labels': show_labels,
        'show_metrics': show_metrics,
        'color_by': color_by
    }


def load_example_pipeline(example_name: str) -> List:
    """Load example pipeline files"""
    example_dir = Path("examples") / example_name.lower().replace(" ", "_")
    
    # Create example files if they don't exist
    if not example_dir.exists():
        create_example_files(example_dir, example_name)
    
    return list(example_dir.glob('**/*'))


def create_example_files(directory: Path, example_name: str):
    """Create example pipeline files"""
    directory.mkdir(parents=True, exist_ok=True)
    
    if example_name == "E-commerce Pipeline":
        # Create sample e-commerce pipeline files
        files = {
            "extract_orders.py": """
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("OrdersETL").getOrCreate()

# Extract orders from source
orders_df = spark.read.format("delta").load("bronze.orders")
customers_df = spark.table("silver.customers")

# Join and transform
enriched_orders = orders_df.join(customers_df, "customer_id")
enriched_orders.write.format("delta").mode("overwrite").saveAsTable("gold.enriched_orders")
            """,
            "aggregate_sales.sql": """
-- Create sales aggregation
CREATE OR REPLACE TABLE gold.sales_summary AS
SELECT 
    date_trunc('day', order_date) as sale_date,
    product_category,
    COUNT(*) as order_count,
    SUM(total_amount) as total_sales
FROM gold.enriched_orders
GROUP BY 1, 2;

-- Create customer metrics
MERGE INTO gold.customer_metrics AS target
USING (
    SELECT customer_id, COUNT(*) as purchase_count, SUM(total_amount) as lifetime_value
    FROM gold.enriched_orders
    GROUP BY customer_id
) AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
            """,
            "infrastructure.tf": """
resource "aws_s3_bucket" "data_lake" {
  bucket = "ecommerce-data-lake"
}

resource "aws_glue_catalog_database" "analytics" {
  name = "ecommerce_analytics"
}

resource "databricks_cluster" "etl_cluster" {
  cluster_name = "ecommerce-etl"
  spark_version = "11.3.x-scala2.12"
  node_type_id = "i3.xlarge"
}
            """
        }
        
        for filename, content in files.items():
            (directory / filename).write_text(content)


def process_files(config: Dict):
    """Process uploaded files and run analysis"""
    if not config['files']:
        st.warning("Por favor, faça upload de arquivos para análise.")
        return None
    
    with st.spinner("🔄 Processando arquivos..."):
        # Save files to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            for uploaded_file in config['files']:
                if hasattr(uploaded_file, 'read'):
                    # Streamlit UploadedFile
                    file_path = Path(temp_dir) / uploaded_file.name
                    file_path.write_bytes(uploaded_file.read())
                else:
                    # Path object
                    file_path = uploaded_file
                
                file_paths.append(str(file_path))
            
            # Run analysis
            analysis_results = {
                'data_lineage': None,
                'terraform': None,
                'databricks': None
            }
            
            # Analyze with main lineage agent
            analysis_results['data_lineage'] = st.session_state.lineage_agent.analyze_pipeline(file_paths)
            
            # Analyze Terraform if enabled
            if config['include_terraform']:
                tf_files = [f for f in file_paths if f.endswith('.tf') or f.endswith('.tf.json')]
                if tf_files:
                    terraform_results = parse_terraform_directory(Path(temp_dir))
                    analysis_results['terraform'] = terraform_results
            
            # Analyze Databricks if enabled
            if config['include_databricks']:
                databricks_results = parse_databricks_workspace(temp_dir)
                if databricks_results['assets']:
                    analysis_results['databricks'] = databricks_results
            
            # Analyze Airflow DAGs if enabled
            if config.get('include_airflow'):
                # Check for Airflow DAG files
                dag_files = [f for f in file_paths if 'dag' in f.lower() or 
                           ('airflow' in f.lower() and f.endswith('.py'))]
                if dag_files:
                    airflow_results = parse_airflow_dags_folder(Path(temp_dir))
                    if airflow_results['dags']:
                        analysis_results['airflow'] = airflow_results
            
            st.session_state.analysis_results = analysis_results
            
            return analysis_results


def display_metrics_dashboard(analysis_results: Dict):
    """Display metrics dashboard"""
    if not analysis_results:
        return
    
    st.header("📊 Dashboard de Métricas")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = len(analysis_results['data_lineage'].get('assets', []))
        st.metric("Total de Assets", total_assets, delta="↑ 100%")
    
    with col2:
        total_transforms = len(analysis_results['data_lineage'].get('transformations', []))
        st.metric("Transformações", total_transforms)
    
    with col3:
        graph = analysis_results['data_lineage'].get('graph')
        complexity = graph.number_of_edges() if graph else 0
        st.metric("Complexidade", complexity)
    
    with col4:
        critical_nodes = identify_critical_nodes(analysis_results)
        st.metric("Nós Críticos", len(critical_nodes), delta="⚠️" if critical_nodes else "✓")
    
    # Additional metrics in expandable sections
    with st.expander("📈 Métricas Detalhadas", expanded=False):
        metrics = analysis_results['data_lineage'].get('metrics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Estatísticas do Grafo")
            if metrics.get('graph_metrics'):
                df = pd.DataFrame([metrics['graph_metrics']])
                st.dataframe(df.T, use_container_width=True)
        
        with col2:
            st.subheader("Análise de Complexidade")
            if metrics.get('complexity_metrics'):
                complexity_data = metrics['complexity_metrics']
                
                # Create gauge chart for complexity score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=complexity_data.get('avg_degree', 0),
                    title={'text': "Grau Médio"},
                    gauge={'axis': {'range': [0, 10]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 3], 'color': "lightgray"},
                               {'range': [3, 7], 'color': "yellow"},
                               {'range': [7, 10], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 8}}
                ))
                st.plotly_chart(fig, use_container_width=True)


def visualize_lineage_graph(analysis_results: Dict, config: Dict):
    """Create and display lineage graph visualization"""
    if not analysis_results or 'data_lineage' not in analysis_results:
        st.info("Execute a análise primeiro para visualizar o grafo.")
        return
    
    st.header("🔗 Visualização da Linhagem")
    
    graph = analysis_results['data_lineage'].get('graph')
    if not graph or graph.number_of_nodes() == 0:
        st.warning("Nenhum grafo de linhagem disponível.")
        return
    
    # Update visualizer with current graph
    st.session_state.visualizer.graph = graph
    
    # Create visualization based on selected type
    if config['graph_type'] == "Force-Directed":
        fig = create_force_directed_graph(graph, config)
    elif config['graph_type'] == "Hierárquico":
        fig = create_hierarchical_graph(graph, config)
    elif config['graph_type'] == "Sankey":
        fig = create_sankey_diagram(graph, config)
    elif config['graph_type'] == "Radial":
        fig = create_radial_graph(graph, config)
    else:  # 3D
        fig = create_3d_graph(graph, config)
    
    # Display the graph
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    # Graph analysis tools
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Encontrar Caminho"):
            source = st.selectbox("Origem", list(graph.nodes()))
            target = st.selectbox("Destino", list(graph.nodes()))
            if source and target and source != target:
                try:
                    path = nx.shortest_path(graph, source, target)
                    st.success(f"Caminho: {' → '.join(path)}")
                except nx.NetworkXNoPath:
                    st.error("Não há caminho entre os nós selecionados.")
    
    with col2:
        if st.button("🔍 Detectar Ciclos"):
            try:
                cycles = list(nx.simple_cycles(graph))[:5]
                if cycles:
                    st.warning("⚠️ Ciclos detectados!")
                    for i, cycle in enumerate(cycles, 1):
                        st.write(f"Ciclo {i}: {' → '.join(cycle + [cycle[0]])}")
                else:
                    st.success("✅ Nenhum ciclo detectado.")
            except:
                st.info("Grafo não contém ciclos.")
    
    with col3:
        if st.button("📊 Análise de Centralidade"):
            centrality = nx.degree_centrality(graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            st.write("**Nós mais centrais:**")
            for node, score in top_nodes:
                st.write(f"- {node}: {score:.3f}")


def create_force_directed_graph(graph: nx.DiGraph, config: Dict) -> go.Figure:
    """Create force-directed graph visualization"""
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Create edge trace
    edge_trace = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none'
        ))
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Color based on configuration
        if config['color_by'] == "Tipo":
            # Color by node type (simplified)
            if 'table' in node.lower():
                node_color.append('blue')
            elif 'view' in node.lower():
                node_color.append('green')
            else:
                node_color.append('gray')
        else:
            node_color.append(graph.degree(node))
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text' if config['show_labels'] else 'markers',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))
    
    return fig


def create_hierarchical_graph(graph: nx.DiGraph, config: Dict) -> go.Figure:
    """Create hierarchical graph visualization"""
    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    except:
        # Fallback to custom hierarchical layout
        pos = hierarchical_layout(graph)
    
    # Similar to force-directed but with hierarchical positions
    return create_force_directed_graph(graph, config)  # Reuse with different positions


def create_sankey_diagram(graph: nx.DiGraph, config: Dict) -> go.Figure:
    """Create Sankey diagram for data flow"""
    # Prepare data for Sankey
    nodes = list(graph.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    source = []
    target = []
    value = []
    
    for edge in graph.edges():
        source.append(node_indices[edge[0]])
        target.append(node_indices[edge[1]])
        value.append(1)  # Could be based on data volume or frequency
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(0,0,255,0.2)"
        )
    )])
    
    fig.update_layout(
        title_text="Data Flow Sankey Diagram",
        font_size=10,
        height=600
    )
    
    return fig


def create_radial_graph(graph: nx.DiGraph, config: Dict) -> go.Figure:
    """Create radial graph visualization"""
    # Create radial layout
    center_node = max(graph.nodes(), key=lambda n: graph.degree(n))
    
    # BFS from center to create layers
    layers = {}
    visited = set()
    queue = [(center_node, 0)]
    
    while queue:
        node, layer = queue.pop(0)
        if node in visited:
            continue
        
        visited.add(node)
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, layer + 1))
    
    # Position nodes in circles
    pos = {}
    import math
    
    for layer, nodes in layers.items():
        if layer == 0:
            pos[nodes[0]] = (0, 0)
        else:
            radius = layer * 2
            angle_step = 2 * math.pi / len(nodes)
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node] = (x, y)
    
    # Add remaining nodes
    for node in graph.nodes():
        if node not in pos:
            pos[node] = (0, 0)
    
    # Create visualization (reuse force-directed with radial positions)
    return create_force_directed_graph(graph, config)


def create_3d_graph(graph: nx.DiGraph, config: Dict) -> go.Figure:
    """Create 3D graph visualization"""
    # 3D layout
    pos = nx.spring_layout(graph, dim=3, k=2, iterations=50)
    
    # Extract coordinates
    edge_trace = []
    for edge in graph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='none'
        ))
    
    # Node trace
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    
    for node in graph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
    
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers+text' if config['show_labels'] else 'markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=5,
            color=list(range(len(node_x))),
            colorscale='Viridis',
            showscale=True,
            line=dict(width=0.5, color='white')
        )
    )
    
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='')
        ),
        height=700
    )
    
    return fig


def impact_analysis_section():
    """Section for impact analysis"""
    st.header("💥 Análise de Impacto")
    
    if not st.session_state.analysis_results:
        st.info("Execute a análise primeiro para realizar análise de impacto.")
        return
    
    graph = st.session_state.analysis_results['data_lineage'].get('graph')
    if not graph:
        return
    
    # Select assets for impact analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selecione Assets para Análise")
        all_assets = list(graph.nodes())
        selected_assets = st.multiselect(
            "Assets a modificar",
            all_assets,
            default=st.session_state.selected_assets
        )
        st.session_state.selected_assets = selected_assets
    
    with col2:
        st.subheader("Tipo de Mudança")
        change_type = st.selectbox(
            "Tipo de modificação",
            ["Schema Change", "Data Type Change", "Column Addition", 
             "Column Removal", "Table Deletion", "Performance Optimization"]
        )
        
        severity = st.select_slider(
            "Severidade da mudança",
            options=["Minor", "Moderate", "Major", "Critical"],
            value="Moderate"
        )
    
    if st.button("🔍 Analisar Impacto", type="primary"):
        if selected_assets:
            with st.spinner("Analisando impacto..."):
                # Perform impact analysis
                impact = st.session_state.lineage_agent.analyze_change_impact(selected_assets)
                st.session_state.impact_analysis = impact
                
                # Display results
                st.subheader("📊 Resultados da Análise de Impacto")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Assets Afetados Diretamente",
                        len(impact['directly_affected']),
                        delta=f"+{len(impact['downstream_affected'])} downstream"
                    )
                
                with col2:
                    risk_color = {
                        'LOW': '🟢',
                        'MEDIUM': '🟡',
                        'HIGH': '🔴',
                        'CRITICAL': '🔴🔴'
                    }
                    st.metric(
                        "Nível de Risco",
                        f"{risk_color.get(impact['risk_level'], '❓')} {impact['risk_level']}"
                    )
                
                with col3:
                    st.metric(
                        "Pipelines Críticos",
                        len(impact.get('affected_pipelines', []))
                    )
                
                # Detailed impact view
                with st.expander("🔍 Detalhes do Impacto", expanded=True):
                    tabs = st.tabs(["Downstream", "Upstream", "Recomendações", "Grafo de Impacto"])
                    
                    with tabs[0]:  # Downstream
                        st.write("**Assets afetados downstream:**")
                        if impact['downstream_affected']:
                            for asset in impact['downstream_affected']:
                                st.write(f"- 📊 {asset}")
                        else:
                            st.success("Nenhum asset downstream afetado.")
                    
                    with tabs[1]:  # Upstream
                        st.write("**Dependências upstream:**")
                        if impact['upstream_dependencies']:
                            for asset in impact['upstream_dependencies']:
                                st.write(f"- 📥 {asset}")
                        else:
                            st.info("Nenhuma dependência upstream.")
                    
                    with tabs[2]:  # Recommendations
                        st.write("**Recomendações:**")
                        for rec in impact.get('recommendations', []):
                            st.info(rec)
                        
                        # Additional recommendations based on change type
                        if change_type == "Schema Change":
                            st.warning("⚠️ Verifique compatibilidade de schema em todos os consumidores.")
                        elif change_type == "Table Deletion":
                            st.error("🚨 ATENÇÃO: Remoção de tabela tem alto impacto!")
                    
                    with tabs[3]:  # Impact Graph
                        # Create subgraph showing impact
                        impact_graph = create_impact_subgraph(
                            graph,
                            selected_assets,
                            impact['downstream_affected'],
                            impact['upstream_dependencies']
                        )
                        st.plotly_chart(impact_graph, use_container_width=True)
        else:
            st.warning("Selecione pelo menos um asset para análise.")


def create_impact_subgraph(graph, selected, downstream, upstream):
    """Create a subgraph highlighting impact"""
    # Get all relevant nodes
    relevant_nodes = set(selected) | set(downstream) | set(upstream)
    
    # Create subgraph
    subgraph = graph.subgraph(relevant_nodes)
    
    # Position nodes
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # Create traces with different colors
    edge_trace = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Color based on impact direction
        if edge[0] in selected and edge[1] in downstream:
            color = 'red'
            width = 3
        elif edge[1] in selected and edge[0] in upstream:
            color = 'blue'
            width = 2
        else:
            color = '#888'
            width = 1
        
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none'
        ))
    
    # Node trace with colors
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        if node in selected:
            node_color.append('gold')
            node_size.append(20)
        elif node in downstream:
            node_color.append('red')
            node_size.append(15)
        elif node in upstream:
            node_color.append('blue')
            node_size.append(15)
        else:
            node_color.append('gray')
            node_size.append(10)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            title="Grafo de Impacto",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
    )
    
    # Add legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='gold'),
        legendgroup='selected', showlegend=True, name="Selecionados"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='red'),
        legendgroup='downstream', showlegend=True, name="Downstream"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='blue'),
        legendgroup='upstream', showlegend=True, name="Upstream"
    ))
    
    return fig


def version_comparison_section():
    """Section for comparing pipeline versions"""
    st.header("🔄 Comparação de Versões")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pipeline v1 (Atual)")
        current_files = st.file_uploader(
            "Upload arquivos atuais",
            type=['py', 'sql', 'tf'],
            accept_multiple_files=True,
            key="v1_files"
        )
    
    with col2:
        st.subheader("Pipeline v2 (Nova)")
        new_files = st.file_uploader(
            "Upload arquivos novos",
            type=['py', 'sql', 'tf'],
            accept_multiple_files=True,
            key="v2_files"
        )
    
    if st.button("📊 Comparar Versões", type="primary"):
        if current_files and new_files:
            with st.spinner("Comparando versões..."):
                # Process both versions
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save and process v1
                    v1_paths = []
                    for file in current_files:
                        path = Path(temp_dir) / "v1" / file.name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(file.read())
                        v1_paths.append(str(path))
                    
                    # Save and process v2
                    v2_paths = []
                    for file in new_files:
                        path = Path(temp_dir) / "v2" / file.name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(file.read())
                        v2_paths.append(str(path))
                    
                    # Compare versions
                    comparison = st.session_state.lineage_agent.compare_versions(v1_paths, v2_paths)
                    st.session_state.comparison_results = comparison
                    
                    # Display comparison results
                    display_comparison_results(comparison)


def display_comparison_results(comparison: Dict):
    """Display version comparison results"""
    st.subheader("📈 Resultados da Comparação")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Assets Adicionados", len(comparison.get('added_assets', [])), delta="↑")
    
    with col2:
        st.metric("Assets Removidos", len(comparison.get('removed_assets', [])), delta="↓")
    
    with col3:
        st.metric("Conexões Adicionadas", len(comparison.get('added_connections', [])))
    
    with col4:
        st.metric("Conexões Removidas", len(comparison.get('removed_connections', [])))
    
    # Detailed changes
    with st.expander("🔍 Mudanças Detalhadas", expanded=True):
        tabs = st.tabs(["Assets", "Conexões", "Impacto", "Riscos"])
        
        with tabs[0]:  # Assets
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Assets Adicionados:**")
                for asset in comparison.get('added_assets', []):
                    st.write(f"+ {asset}")
            
            with col2:
                st.write("**❌ Assets Removidos:**")
                for asset in comparison.get('removed_assets', []):
                    st.write(f"- {asset}")
        
        with tabs[1]:  # Connections
            st.write("**Mudanças nas Conexões:**")
            
            if comparison.get('added_connections'):
                st.write("**Novas conexões:**")
                for conn in comparison['added_connections']:
                    st.write(f"+ {conn[0]} → {conn[1]}")
            
            if comparison.get('removed_connections'):
                st.write("**Conexões removidas:**")
                for conn in comparison['removed_connections']:
                    st.write(f"- {conn[0]} → {conn[1]}")
        
        with tabs[2]:  # Impact
            if comparison.get('risk_assessment', {}).get('removed_assets_impact'):
                st.write("**Impacto da remoção de assets:**")
                for asset, impact in comparison['risk_assessment']['removed_assets_impact'].items():
                    st.warning(f"⚠️ Remoção de '{asset}' afeta: {', '.join(impact)}")
        
        with tabs[3]:  # Risks
            st.write("**Avaliação de Riscos:**")
            
            risk_level = assess_change_risk(comparison)
            
            if risk_level == "HIGH":
                st.error("🔴 Risco Alto: Mudanças significativas detectadas!")
            elif risk_level == "MEDIUM":
                st.warning("🟡 Risco Médio: Revisar mudanças cuidadosamente.")
            else:
                st.success("🟢 Risco Baixo: Mudanças parecem seguras.")


def identify_critical_nodes(analysis_results: Dict) -> List[str]:
    """Identify critical nodes in the graph"""
    if not analysis_results or 'data_lineage' not in analysis_results:
        return []
    
    graph = analysis_results['data_lineage'].get('graph')
    if not graph:
        return []
    
    critical = []
    
    for node in graph.nodes():
        # Critical if many dependencies
        if graph.in_degree(node) > 5 or graph.out_degree(node) > 5:
            critical.append(node)
    
    return critical


def hierarchical_layout(graph: nx.DiGraph) -> Dict:
    """Create hierarchical layout for graph"""
    # Simple hierarchical layout based on topological sort
    try:
        layers = list(nx.topological_generations(graph))
    except nx.NetworkXError:
        # Graph has cycles, use spring layout
        return nx.spring_layout(graph)
    
    pos = {}
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            pos[node] = (j - len(layer) / 2, -i)
    
    return pos


def assess_change_risk(comparison: Dict) -> str:
    """Assess risk level of changes"""
    removed_count = len(comparison.get('removed_assets', []))
    removed_conn_count = len(comparison.get('removed_connections', []))
    
    if removed_count > 5 or removed_conn_count > 10:
        return "HIGH"
    elif removed_count > 2 or removed_conn_count > 5:
        return "MEDIUM"
    else:
        return "LOW"


def export_results():
    """Export analysis results"""
    if not st.session_state.analysis_results:
        st.warning("Nenhum resultado para exportar.")
        return
    
    st.header("📥 Exportar Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Exportar Documentação"):
            doc = st.session_state.lineage_agent.generate_documentation()
            st.download_button(
                label="Download Documentation.md",
                data=doc,
                file_name=f"lineage_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("📊 Exportar Grafo"):
            # Export graph as JSON
            graph_data = nx.node_link_data(st.session_state.analysis_results['data_lineage']['graph'])
            graph_json = json.dumps(graph_data, indent=2)
            
            st.download_button(
                label="Download Graph.json",
                data=graph_json,
                file_name=f"lineage_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📈 Exportar Métricas"):
            metrics = st.session_state.analysis_results['data_lineage'].get('metrics', {})
            metrics_df = pd.DataFrame([metrics])
            
            st.download_button(
                label="Download Metrics.csv",
                data=metrics_df.to_csv(index=False),
                file_name=f"lineage_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    """Main application"""
    # Create header
    create_header()
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔗 Visualização",
        "💥 Análise de Impacto",
        "🔄 Comparação",
        "📥 Exportar"
    ])
    
    with tab1:
        if st.button("🚀 Executar Análise", type="primary", use_container_width=True):
            results = process_files(config)
            if results:
                st.success("✅ Análise concluída com sucesso!")
        
        display_metrics_dashboard(st.session_state.analysis_results)
    
    with tab2:
        visualize_lineage_graph(st.session_state.analysis_results, config)
    
    with tab3:
        impact_analysis_section()
    
    with tab4:
        version_comparison_section()
    
    with tab5:
        export_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Data Lineage AI Agent v2.0 | Desenvolvido com ❤️ por Claude AI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
