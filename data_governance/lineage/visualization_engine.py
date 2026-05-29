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
Visualization Engine - Sistema de Visualização Interativa para Data Lineage
Suporta: Force-directed, Hierárquico, Sankey, Radial, 3D
"""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import colorsys
import numpy as np
from datetime import datetime


class DataLineageVisualizer:
    """
    Engine de visualização interativa para grafos de linhagem de dados
    """
    
    def __init__(self, graph: nx.DiGraph = None):
        self.graph = graph if graph else nx.DiGraph()
        self.color_schemes = {
            'default': self._get_default_colors(),
            'impact': self._get_impact_colors(),
            'type_based': self._get_type_colors()
        }
        self.layout_cache = {}
        self.metrics = None  # Store calculated metrics
        self.llm_analysis = None  # Store LLM analysis
        
        # Calculate metrics and LLM analysis on initialization if graph provided
        if self.graph.number_of_nodes() > 0:
            self._calculate_metrics_and_analysis()
        
    def _get_default_colors(self) -> Dict:
        """Esquema de cores padrão"""
        return {
            'node_color': '#3498db',
            'edge_color': '#95a5a6',
            'highlight_color': '#e74c3c',
            'background': '#ecf0f1'
        }
    
    def _get_impact_colors(self) -> Dict:
        """Esquema de cores para análise de impacto"""
        return {
            'source': '#2ecc71',
            'affected': '#e74c3c',
            'indirect': '#f39c12',
            'normal': '#3498db'
        }
    
    def _get_type_colors(self) -> Dict:
        """Cores baseadas no tipo de asset"""
        return {
            'table': '#3498db',
            'view': '#9b59b6',
            'file': '#2ecc71',
            'stream': '#e67e22',
            'terraform_resource': '#34495e',
            'databricks_table': '#e74c3c',
            'delta_table': '#16a085'
        }
    
    def _calculate_metrics_and_analysis(self):
        """
        Calcula métricas do grafo e gera análise LLM
        """
        # Calculate basic metrics
        self.metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'is_dag': nx.is_directed_acyclic_graph(self.graph),
        }
        
        if self.graph.number_of_nodes() > 0:
            try:
                self.metrics['avg_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
                self.metrics['connected_components'] = nx.number_weakly_connected_components(self.graph)
                
                # Find sources and sinks
                self.metrics['sources'] = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
                self.metrics['sinks'] = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
                
                # Calculate longest path if DAG
                if self.metrics['is_dag']:
                    try:
                        longest_path = nx.dag_longest_path(self.graph)
                        self.metrics['longest_path_length'] = len(longest_path)
                        self.metrics['longest_path'] = longest_path
                    except:
                        pass
                
                # Centrality measures for small graphs
                if self.graph.number_of_nodes() < 1000:
                    self.metrics['betweenness'] = nx.betweenness_centrality(self.graph)
                    self.metrics['pagerank'] = nx.pagerank(self.graph, max_iter=100)
            except:
                pass
        
        # Generate LLM analysis
        try:
            from llm_graph_analyzer import GraphLLMAnalyzer
            analyzer = GraphLLMAnalyzer()
            self.llm_analysis = analyzer.analyze_graph(self.graph, self.metrics)
            
            # Merge key insights into metrics
            if self.llm_analysis:
                self.metrics['llm_summary'] = self.llm_analysis.get('overall_summary', '')
                self.metrics['insights'] = self.llm_analysis.get('insights', [])
                self.metrics['recommendations'] = self.llm_analysis.get('recommendations', [])
                self.metrics['natural_language_report'] = self.llm_analysis.get('natural_language_report', '')
        except Exception as e:
            print(f"LLM analysis not available: {e}")
            self.llm_analysis = None
    
    def get_llm_summary(self) -> str:
        """
        Retorna o resumo em linguagem natural do grafo
        """
        if not self.llm_analysis:
            self._calculate_metrics_and_analysis()
        
        if self.llm_analysis:
            return self.llm_analysis.get('overall_summary', 'No LLM summary available')
        return "LLM analysis not available"
    
    def get_insights(self) -> List[Dict]:
        """
        Retorna insights detectados pelo LLM
        """
        if not self.llm_analysis:
            self._calculate_metrics_and_analysis()
        
        if self.llm_analysis:
            return self.llm_analysis.get('insights', [])
        return []
    
    def get_recommendations(self) -> List[Dict]:
        """
        Retorna recomendações de melhorias
        """
        if not self.llm_analysis:
            self._calculate_metrics_and_analysis()
        
        if self.llm_analysis:
            return self.llm_analysis.get('recommendations', [])
        return []
    
    def get_natural_language_report(self) -> str:
        """
        Retorna relatório completo em linguagem natural
        """
        if not self.llm_analysis:
            self._calculate_metrics_and_analysis()
        
        if self.llm_analysis:
            return self.llm_analysis.get('natural_language_report', '')
        return "No natural language report available"
    
    def visualize_force_directed(self, 
                                highlight_nodes: List[str] = None,
                                title: str = "Data Lineage - Force Directed Graph",
                                show_labels: bool = True) -> go.Figure:
        """
        Cria visualização force-directed interativa
        """
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_figure("No data to visualize")
        
        # Layout usando spring layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Prepara dados dos nós
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            marker=dict(
                size=[],
                color=[],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left'
                ),
                line=dict(width=2)
            ),
            text=[],
            textposition="top center",
            hovertext=()
        )
        
        # Prepara dados das arestas
        edge_traces = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                hovertext=f"{edge[0]} → {edge[1]}"
            )
            edge_traces.append(edge_trace)
        
        # Adiciona nós
        for node in self.graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Informações do nó
            node_info = self.graph.nodes[node]
            connections = self.graph.degree(node)
            
            # Cor baseada no tipo ou destaque
            if highlight_nodes and node in highlight_nodes:
                color = 10
            else:
                color = connections
                
            node_trace['marker']['size'] += tuple([10 + connections * 2])
            node_trace['marker']['color'] += tuple([color])
            
            # Texto do hover
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {node_info.get('type', 'unknown')}<br>"
            hover_text += f"Connections: {connections}<br>"
            hover_text += f"In: {self.graph.in_degree(node)} | Out: {self.graph.out_degree(node)}"
            
            node_trace['hovertext'] += tuple([hover_text])
            if show_labels:
                node_trace['text'] += tuple([node[:20] + '...' if len(node) > 20 else node])
        
        # Cria a figura
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=dict(text=title, font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           paper_bgcolor='white',
                           plot_bgcolor='white'
                       ))
        
        # Adiciona interatividade
        fig.update_layout(
            dragmode='pan',
            clickmode='event+select',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    def visualize_hierarchical(self,
                              root_node: str = None,
                              orientation: str = 'vertical',
                              title: str = "Data Lineage - Hierarchical View") -> go.Figure:
        """
        Cria visualização hierárquica (árvore)
        """
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_figure("No data to visualize")
        
        # Se não há root especificado, encontra nós sem predecessores
        if not root_node:
            roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
            if not roots:
                # Se não há raízes claras, usa o nó com mais conexões
                root_node = max(self.graph.nodes(), key=lambda n: self.graph.degree(n))
            else:
                root_node = roots[0]
        
        # Gera layout hierárquico
        if orientation == 'vertical':
            pos = self._hierarchical_layout_vertical(root_node)
        else:
            pos = self._hierarchical_layout_horizontal(root_node)
        
        # Cria traces para visualização
        edge_traces = []
        for edge in self.graph.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
        
        # Trace dos nós
        node_trace = go.Scatter(
            x=[pos[node][0] for node in pos],
            y=[pos[node][1] for node in pos],
            mode='markers+text',
            text=[str(node)[:15] + '...' if len(str(node)) > 15 else str(node) for node in pos],
            textposition="bottom center",
            hoverinfo='text',
            hovertext=[self._get_node_hover_text(node) for node in pos],
            marker=dict(
                size=15,
                color=[self._get_node_color(node) for node in pos],
                line=dict(width=2, color='white')
            )
        )
        
        # Cria figura
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def visualize_sankey(self,
                        title: str = "Data Flow - Sankey Diagram",
                        filter_threshold: int = 0) -> go.Figure:
        """
        Cria diagrama Sankey para visualizar fluxo de dados
        """
        if self.graph.number_of_edges() == 0:
            return self._create_empty_figure("No data flows to visualize")
        
        # Prepara dados para Sankey
        nodes = list(self.graph.nodes())
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        sources = []
        targets = []
        values = []
        labels = []
        
        for edge in self.graph.edges(data=True):
            source_idx = node_indices[edge[0]]
            target_idx = node_indices[edge[1]]
            
            # Peso da conexão (pode ser personalizado)
            weight = edge[2].get('weight', 1)
            
            if weight > filter_threshold:
                sources.append(source_idx)
                targets.append(target_idx)
                values.append(weight)
        
        # Labels dos nós
        node_labels = [str(node)[:20] + '...' if len(str(node)) > 20 else str(node) 
                      for node in nodes]
        
        # Cores dos nós baseadas no tipo
        node_colors = [self._get_node_color_hex(node) for node in nodes]
        
        # Cria o diagrama Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                hovertemplate='%{label}<br>Total connections: %{value}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(100, 100, 100, 0.2)',
                hovertemplate='%{source.label} → %{target.label}<br>Flow: %{value}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            font=dict(size=10),
            height=600
        )
        
        return fig
    
    def visualize_impact_analysis(self,
                                 changed_nodes: List[str],
                                 title: str = "Impact Analysis Visualization") -> go.Figure:
        """
        Visualiza análise de impacto com destaque nas áreas afetadas
        """
        if not changed_nodes:
            return self._create_empty_figure("No nodes selected for impact analysis")

        changed_nodes = [n for n in changed_nodes if n in self.graph]
        if not changed_nodes:
            return self._create_empty_figure("Selected nodes are not present in the graph")
        
        # Calcula impacto
        directly_affected = set(changed_nodes)
        upstream_affected = set()
        downstream_affected = set()
        
        for node in changed_nodes:
            # Upstream
            for pred in nx.ancestors(self.graph, node):
                upstream_affected.add(pred)
            
            # Downstream
            for succ in nx.descendants(self.graph, node):
                downstream_affected.add(succ)
        
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Traces para diferentes categorias
        traces = []
        
        # Arestas
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Cor da aresta baseada no impacto
            if edge[0] in directly_affected or edge[1] in directly_affected:
                color = 'red'
                width = 2
            elif edge[0] in downstream_affected or edge[1] in downstream_affected:
                color = 'orange'
                width = 1.5
            else:
                color = '#ddd'
                width = 0.5
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='none'
            )
            traces.append(edge_trace)
        
        # Nós por categoria
        categories = [
            ('Directly Changed', directly_affected, 'red', 20),
            ('Upstream Dependencies', upstream_affected - directly_affected, 'blue', 15),
            ('Downstream Impact', downstream_affected - directly_affected, 'orange', 15),
            ('Unaffected', set(self.graph.nodes()) - directly_affected - upstream_affected - downstream_affected, '#ddd', 10)
        ]
        
        for cat_name, nodes, color, size in categories:
            if nodes:
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in nodes if node in pos],
                    y=[pos[node][1] for node in nodes if node in pos],
                    mode='markers+text',
                    name=cat_name,
                    text=[str(node)[:10] for node in nodes if node in pos],
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=[f"{node}<br>Category: {cat_name}" for node in nodes if node in pos],
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=2, color='white')
                    )
                )
                traces.append(node_trace)
        
        # Cria figura
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        )
        
        # Adiciona anotações com estatísticas
        stats_text = f"Directly affected: {len(directly_affected)}<br>"
        stats_text += f"Upstream dependencies: {len(upstream_affected - directly_affected)}<br>"
        stats_text += f"Downstream impact: {len(downstream_affected - directly_affected)}<br>"
        stats_text += f"Total affected: {len(directly_affected | upstream_affected | downstream_affected)}"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            align="left"
        )
        
        return fig
    
    def visualize_3d_graph(self,
                          title: str = "Data Lineage - 3D Visualization") -> go.Figure:
        """
        Cria visualização 3D interativa do grafo
        """
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_figure("No data to visualize")
        
        # Layout 3D
        pos = nx.spring_layout(self.graph, dim=3, k=2, iterations=50, seed=42)
        
        # Extrai coordenadas
        x_nodes = [pos[node][0] for node in self.graph.nodes()]
        y_nodes = [pos[node][1] for node in self.graph.nodes()]
        z_nodes = [pos[node][2] for node in self.graph.nodes()]
        
        # Arestas
        x_edges = []
        y_edges = []
        z_edges = []
        
        for edge in self.graph.edges():
            x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])
            z_edges.extend([pos[edge[0]][2], pos[edge[1]][2], None])
        
        # Trace das arestas
        edge_trace = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode='lines',
            line=dict(color='rgb(125,125,125)', width=1),
            hoverinfo='none'
        )
        
        # Trace dos nós
        node_trace = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=[5 + self.graph.degree(node) * 2 for node in self.graph.nodes()],
                color=[self.graph.degree(node) for node in self.graph.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Connections",
                    thickness=10,
                    x=1.1
                ),
                line=dict(width=1, color='white')
            ),
            text=[str(node)[:15] for node in self.graph.nodes()],
            hoverinfo='text',
            hovertext=[self._get_node_hover_text(node) for node in self.graph.nodes()]
        )
        
        # Cria figura 3D
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                scene=dict(
                    xaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
                    yaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
                    zaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                paper_bgcolor='white'
            )
        )
        
        return fig
    
    def visualize_radial(self,
                        center_node: str = None,
                        max_depth: int = 3,
                        title: str = "Data Lineage - Radial Layout") -> go.Figure:
        """
        Cria visualização radial centrada em um nó específico
        """
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_figure("No data to visualize")
        
        # Se não há centro especificado, usa o nó com mais conexões
        if not center_node:
            center_node = max(self.graph.nodes(), key=lambda n: self.graph.degree(n))
        
        if center_node not in self.graph:
            return self._create_empty_figure(f"Node {center_node} not found in graph")
        
        # Calcula layout radial
        pos = self._radial_layout(center_node, max_depth)
        
        # Traces
        traces = []
        
        # Arestas
        for edge in self.graph.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none'
                )
                traces.append(edge_trace)
        
        # Nós por distância do centro
        distances = self._calculate_distances(center_node)
        
        for dist in range(max_depth + 1):
            nodes_at_dist = [n for n, d in distances.items() if d == dist and n in pos]
            
            if nodes_at_dist:
                # Cor baseada na distância
                color = self._get_color_by_distance(dist, max_depth)
                size = 20 - dist * 3 if dist > 0 else 25
                
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in nodes_at_dist],
                    y=[pos[node][1] for node in nodes_at_dist],
                    mode='markers+text',
                    name=f"Distance {dist}",
                    text=[str(node)[:10] for node in nodes_at_dist],
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=[f"{node}<br>Distance from center: {dist}" for node in nodes_at_dist],
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=2, color='white')
                    )
                )
                traces.append(node_trace)
        
        # Cria figura
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(text=f"{title}<br>Center: {center_node}", font=dict(size=16)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def create_dashboard(self,
                        changed_nodes: List[str] = None,
                        title: str = "Data Lineage Dashboard") -> go.Figure:
        """
        Cria um dashboard completo com múltiplas visualizações
        """
        # Cria subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Force-Directed Graph', 'Impact Analysis', 
                          'Node Statistics', 'Data Flow Distribution'),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'pie'}]
            ]
        )
        
        # 1. Force-directed graph (simplificado)
        pos = nx.spring_layout(self.graph, k=1, iterations=30, seed=42)
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(
                go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                          mode='lines', line=dict(width=0.5, color='#888'),
                          showlegend=False, hoverinfo='none'),
                row=1, col=1
            )
        
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        
        fig.add_trace(
            go.Scatter(x=node_x, y=node_y,
                      mode='markers',
                      marker=dict(size=8, color=[self.graph.degree(n) for n in self.graph.nodes()],
                                colorscale='Viridis'),
                      showlegend=False),
            row=1, col=1
        )
        
        # 2. Impact Analysis (se houver nós mudados)
        if changed_nodes:
            downstream = set()
            for node in changed_nodes:
                try:
                    downstream.update(nx.descendants(self.graph, node))
                except:
                    pass
            
            impact_x = ['Changed', 'Downstream Impact', 'Unaffected']
            impact_y = [len(changed_nodes), len(downstream), 
                       self.graph.number_of_nodes() - len(changed_nodes) - len(downstream)]
            
            fig.add_trace(
                go.Bar(x=impact_x, y=impact_y,
                      marker_color=['red', 'orange', 'green'],
                      showlegend=False),
                row=1, col=2
            )
        
        # 3. Estatísticas dos nós
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig.add_trace(
            go.Bar(x=[n[0][:10] for n in top_nodes],
                  y=[n[1] for n in top_nodes],
                  marker_color='lightblue',
                  showlegend=False),
            row=2, col=1
        )
        
        # 4. Distribuição de tipos
        type_counts = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        if type_counts:
            fig.add_trace(
                go.Pie(labels=list(type_counts.keys()),
                      values=list(type_counts.values()),
                      showlegend=True),
                row=2, col=2
            )
        
        # Atualiza layout
        fig.update_layout(
            title_text=title,
            height=800,
            showlegend=True
        )
        
        # Remove eixos desnecessários
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
        
        return fig
    
    def export_to_html(self, fig: go.Figure, filename: str = "lineage_viz.html"):
        """
        Exporta visualização para arquivo HTML interativo com resumos LLM
        """
        # Generate LLM analysis if not already done
        if not self.llm_analysis:
            self._calculate_metrics_and_analysis()
        
        # Create enhanced HTML with LLM insights
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Lineage Analysis - AI Enhanced</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #020617;
                    background-image: radial-gradient(circle at 20% 30%, rgba(59,130,246,0.06) 0%, transparent 50%),
                                      radial-gradient(circle at 80% 70%, rgba(139,92,246,0.04) 0%, transparent 40%);
                    min-height: 100vh;
                    color: #f1f5f9;
                    -webkit-font-smoothing: antialiased;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(15,23,42,0.80);
                    backdrop-filter: blur(12px);
                    border-radius: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
                    overflow: hidden;
                    border: 1px solid rgba(51,65,85,0.45);
                }}
                .header {{
                    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                .header::before {{
                    content: '';
                    position: absolute; top: -50%; left: -50%;
                    width: 200%; height: 200%;
                    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.08) 0%, transparent 50%);
                    pointer-events: none;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.4em;
                    font-weight: 800;
                    letter-spacing: -0.03em;
                    position: relative;
                }}
                .header p {{
                    opacity: 0.85;
                    position: relative;
                }}
                .content {{
                    padding: 40px;
                }}
                .summary-section {{
                    background: rgba(15,23,42,0.55);
                    backdrop-filter: blur(8px);
                    border-left: 4px solid #3b82f6;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 14px;
                    border: 1px solid rgba(51,65,85,0.45);
                    color: #94a3b8;
                }}
                .insights-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .insight-card {{
                    background: rgba(15,23,42,0.55);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(51,65,85,0.45);
                    border-radius: 14px;
                    padding: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    transition: all 0.25s cubic-bezier(.4,0,.2,1);
                }}
                .insight-card:hover {{
                    transform: translateY(-3px);
                    border-color: rgba(59,130,246,0.35);
                    box-shadow: 0 0 24px rgba(59,130,246,0.18);
                }}
                .severity-critical {{
                    border-top: 4px solid #ef4444;
                }}
                .severity-high {{
                    border-top: 4px solid #f59e0b;
                }}
                .severity-medium {{
                    border-top: 4px solid #3b82f6;
                }}
                .severity-low {{
                    border-top: 4px solid #64748b;
                }}
                .metric {{
                    display: inline-block;
                    background: linear-gradient(135deg,#3b82f6,#8b5cf6);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 999px;
                    margin: 5px;
                    font-weight: 600;
                    font-size: 0.9em;
                    box-shadow: 0 0 20px rgba(59,130,246,0.18);
                }}
                .recommendation {{
                    background: rgba(34,197,94,0.08);
                    border: 1px solid rgba(34,197,94,0.3);
                    border-radius: 14px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .recommendation h4 {{
                    color: #22c55e;
                    margin-top: 0;
                    font-weight: 600;
                }}
                pre {{
                    background: rgba(15,23,42,0.7);
                    color: #94a3b8;
                    padding: 15px;
                    border-radius: 10px;
                    overflow-x: auto;
                    white-space: pre-wrap;
                    border: 1px solid rgba(51,65,85,0.45);
                }}
                #plotly-div {{
                    margin: 30px 0;
                    border: 1px solid rgba(51,65,85,0.45);
                    border-radius: 14px;
                    overflow: hidden;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Data Lineage Intelligence Report</h1>
                    <p>AI-Powered Pipeline Analysis</p>
                </div>
                
                <div class="content">
                    <!-- Executive Summary -->
                    <div class="summary-section">
                        <h2>📊 Executive Summary</h2>
                        <p>{summary}</p>
                    </div>
                    
                    <!-- Key Metrics -->
                    <h2>📈 Key Metrics</h2>
                    <div>
                        <span class="metric">Assets: {nodes}</span>
                        <span class="metric">Connections: {edges}</span>
                        <span class="metric">Components: {components}</span>
                        <span class="metric">Complexity: {complexity}</span>
                        <span class="metric">Critical Path: {critical_path} steps</span>
                    </div>
                    
                    <!-- Interactive Visualization -->
                    <h2>🎨 Interactive Visualization</h2>
                    <div id="plotly-div">{plotly_graph}</div>
                    
                    <!-- Key Insights -->
                    <h2>💡 Key Insights</h2>
                    <div class="insights-grid">
                        {insights_html}
                    </div>
                    
                    <!-- Recommendations -->
                    <h2>🎯 Recommendations</h2>
                    {recommendations_html}
                    
                    <!-- Natural Language Report -->
                    <h2>📝 Detailed Analysis</h2>
                    <pre>{detailed_report}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Prepare data for template
        summary = self.llm_analysis.get('overall_summary', 'No summary available') if self.llm_analysis else 'Analysis pending'
        
        # Generate insights HTML
        insights_html = self._generate_insights_html()
        
        # Generate recommendations HTML
        recommendations_html = self._generate_recommendations_html()
        
        # Get detailed report
        detailed_report = self.llm_analysis.get('natural_language_report', '') if self.llm_analysis else ''
        
        # Calculate complexity description
        complexity = self._describe_complexity()
        
        # Convert figure to HTML
        plotly_html = fig.to_html(include_plotlyjs='cdn', div_id="plotly-div")
        
        # Fill template
        html_content = html_template.format(
            summary=summary,
            nodes=self.metrics.get('total_nodes', 0) if self.metrics else 0,
            edges=self.metrics.get('total_edges', 0) if self.metrics else 0,
            components=self.metrics.get('connected_components', 0) if self.metrics else 0,
            complexity=complexity,
            critical_path=self.metrics.get('longest_path_length', 'N/A') if self.metrics else 'N/A',
            plotly_graph=plotly_html,
            insights_html=insights_html,
            recommendations_html=recommendations_html,
            detailed_report=detailed_report
        )
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
        fig.write_html(filename, include_plotlyjs='cdn')
        return filename
    
    def export_to_json(self, filename: str = "lineage_data.json") -> str:
        """
        Exporta dados do grafo para JSON
        """
        data = {
            "nodes": [
                {
                    "id": node,
                    "type": self.graph.nodes[node].get('type', 'unknown'),
                    "degree": self.graph.degree(node),
                    "in_degree": self.graph.in_degree(node),
                    "out_degree": self.graph.out_degree(node),
                    "metadata": dict(self.graph.nodes[node])
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "metadata": dict(self.graph.edges[edge])
                }
                for edge in self.graph.edges()
            ],
            "statistics": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(self.graph),
                "connected_components": nx.number_weakly_connected_components(self.graph)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    # Métodos auxiliares privados
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Cria uma figura vazia com mensagem"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig
    
    def _get_node_hover_text(self, node: str) -> str:
        """Gera texto de hover para um nó"""
        info = self.graph.nodes[node]
        text = f"<b>{node}</b><br>"
        text += f"Type: {info.get('type', 'unknown')}<br>"
        text += f"In: {self.graph.in_degree(node)} | Out: {self.graph.out_degree(node)}<br>"
        text += f"Total connections: {self.graph.degree(node)}"
        return text
    
    def _get_node_color(self, node: str) -> str:
        """Retorna cor do nó baseada no tipo"""
        node_type = self.graph.nodes[node].get('type', 'unknown')
        colors = self.color_schemes['type_based']
        return colors.get(node_type, '#888888')
    
    def _get_node_color_hex(self, node: str) -> str:
        """Retorna cor do nó em formato hex"""
        color = self._get_node_color(node)
        return color if color.startswith('#') else '#888888'
    
    def _hierarchical_layout_vertical(self, root: str) -> Dict[str, Tuple[float, float]]:
        """Calcula layout hierárquico vertical"""
        pos = {}
        visited = set()
        levels = {root: 0}
        queue = [root]
        
        # BFS para definir níveis
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            for successor in self.graph.successors(node):
                if successor not in levels:
                    levels[successor] = levels[node] + 1
                    queue.append(successor)
        
        # Calcula posições
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        for level, nodes in level_nodes.items():
            y = -level * 2
            total_width = len(nodes) * 2
            start_x = -total_width / 2
            
            for i, node in enumerate(nodes):
                x = start_x + i * 2
                pos[node] = (x, y)
        
        return pos
    
    def _hierarchical_layout_horizontal(self, root: str) -> Dict[str, Tuple[float, float]]:
        """Calcula layout hierárquico horizontal"""
        pos = self._hierarchical_layout_vertical(root)
        # Inverte x e y para layout horizontal
        return {node: (y, x) for node, (x, y) in pos.items()}
    
    def _radial_layout(self, center: str, max_depth: int) -> Dict[str, Tuple[float, float]]:
        """Calcula layout radial"""
        pos = {center: (0, 0)}
        distances = self._calculate_distances(center)
        
        for dist in range(1, min(max_depth + 1, max(distances.values()) + 1)):
            nodes_at_dist = [n for n, d in distances.items() if d == dist]
            
            if nodes_at_dist:
                radius = dist * 2
                angle_step = 2 * np.pi / len(nodes_at_dist)
                
                for i, node in enumerate(nodes_at_dist):
                    angle = i * angle_step
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    pos[node] = (x, y)
        
        return pos
    
    def _calculate_distances(self, center: str) -> Dict[str, int]:
        """Calcula distâncias de todos os nós até o centro"""
        distances = {center: 0}
        visited = set()
        queue = [(center, 0)]
        
        while queue:
            node, dist = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            # Considera predecessores e sucessores (grafo não direcionado para distância)
            neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
            
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        # Nós não conectados
        for node in self.graph.nodes():
            if node not in distances:
                distances[node] = float('inf')
        
        return distances
    
    def _get_color_by_distance(self, distance: int, max_distance: int) -> str:
        """Gera cor baseada na distância"""
        if distance == 0:
            return 'red'
        
        # Gradiente de azul para verde
        hue = 0.6 - (distance / max_distance) * 0.3
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
    
    def _generate_insights_html(self) -> str:
        """Gera HTML para seção de insights"""
        if not self.llm_analysis or 'insights' not in self.llm_analysis:
            return "<p>No insights available</p>"
        
        html_parts = []
        for insight in self.llm_analysis['insights'][:6]:  # Top 6 insights
            severity_class = f"severity-{insight['severity'].lower()}"
            html_parts.append(f"""
            <div class="insight-card {severity_class}">
                <h3>{insight['title']}</h3>
                <p><strong>Severity:</strong> {insight['severity']}</p>
                <p>{insight['description']}</p>
                <p><strong>Affected:</strong> {len(insight.get('affected_nodes', []))} nodes</p>
                <p><strong>Action:</strong> {insight['recommendation']}</p>
            </div>
            """)
        
        return ''.join(html_parts)
    
    def _generate_recommendations_html(self) -> str:
        """Gera HTML para seção de recomendações"""
        if not self.llm_analysis or 'recommendations' not in self.llm_analysis:
            return "<p>No recommendations available</p>"
        
        html_parts = []
        for rec in self.llm_analysis['recommendations'][:5]:
            html_parts.append(f"""
            <div class="recommendation">
                <h4>{rec['title']} (Priority: {rec['priority']})</h4>
                <p>{rec['description']}</p>
                <ul>
                    {''.join(f"<li>{action}</li>" for action in rec.get('actions', [])[:3])}
                </ul>
                <p><strong>Impact:</strong> {rec.get('impact', 'N/A')}</p>
            </div>
            """)
        
        return ''.join(html_parts)
    
    def _describe_complexity(self) -> str:
        """Descreve complexidade do grafo"""
        if not self.metrics:
            return "Unknown"
        
        density = self.metrics.get('density', 0)
        nodes = self.metrics.get('total_nodes', 0)
        
        if nodes > 100 or density > 0.3:
            return "Very High"
        elif nodes > 50 or density > 0.2:
            return "High"
        elif nodes > 20 or density > 0.1:
            return "Medium"
        else:
            return "Low"

    def visualize_atlas_style(self,
                              title: str = "Data Lineage - Atlas View",
                              group_by_type: bool = True) -> go.Figure:
        """
        Visualização estilo Apache Atlas/Neo4j
        - Layout hierárquico limpo
        - Nodes grandes e coloridos por tipo
        - Labels sempre visíveis
        - Agrupamento por tipo de asset
        - Setas direcionais claras
        """
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_figure("No data to visualize")

        # Define cores por tipo de asset (estilo Atlas)
        type_colors = {
            'table': '#1f77b4',      # Azul
            'view': '#ff7f0e',       # Laranja
            'file': '#2ca02c',       # Verde
            'stream': '#d62728',     # Vermelho
            'dataset': '#9467bd',    # Roxo
            'terraform_resource': '#8c564b',  # Marrom
            'databricks_table': '#e377c2',    # Rosa
            'airflow_task': '#7f7f7f',        # Cinza
            'delta_table': '#bcbd22',         # Amarelo-verde
            'unknown': '#17becf'     # Ciano
        }

        # Layout hierárquico (tenta usar graphviz, fallback para kamada_kawai)
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(self.graph, prog='dot', args='-Grankdir=LR')
        except:
            try:
                pos = nx.kamada_kawai_layout(self.graph)
            except:
                pos = nx.spring_layout(self.graph, k=3, iterations=100, seed=42)

        # Normaliza posições para melhor visualização
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            x_range = max(xs) - min(xs) if len(set(xs)) > 1 else 1
            y_range = max(ys) - min(ys) if len(set(ys)) > 1 else 1
            pos = {
                node: ((x - min(xs)) / x_range * 1000, (y - min(ys)) / y_range * 600)
                for node, (x, y) in pos.items()
            }

        # Prepara edges com setas
        edge_traces = []
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Linha da aresta
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='text',
                hovertext=f"{source} → {target}<br>Operation: {data.get('operation', 'N/A')}",
                showlegend=False
            )
            edge_traces.append(edge_trace)

            # Seta (annotation)
            # Calcula ponto para a seta (80% do caminho)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)

        # Agrupa nodes por tipo
        nodes_by_type = {}
        for node in self.graph.nodes(data=True):
            node_name, node_data = node
            node_type = node_data.get('type', 'unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_name, node_data))

        # Cria trace para cada tipo de node
        node_traces = []
        for node_type, nodes in nodes_by_type.items():
            color = type_colors.get(node_type, type_colors['unknown'])

            # Dados do trace
            x_vals = []
            y_vals = []
            hover_texts = []
            labels = []

            for node_name, node_data in nodes:
                x, y = pos[node_name]
                x_vals.append(x)
                y_vals.append(y)

                # Label do nó (nome curto)
                label = node_name[:30] + '...' if len(node_name) > 30 else node_name
                labels.append(label)

                # Hover text rico
                in_degree = self.graph.in_degree(node_name)
                out_degree = self.graph.out_degree(node_name)
                source_file = node_data.get('source_file', 'N/A')

                hover_text = f"""<b>{node_name}</b><br>
<b>Type:</b> {node_type}<br>
<b>Source:</b> {source_file}<br>
<b>Connections:</b> {in_degree + out_degree}<br>
<b>Upstream:</b> {in_degree} | <b>Downstream:</b> {out_degree}"""
                hover_texts.append(hover_text)

            # Cria trace do tipo
            node_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                name=node_type,
                marker=dict(
                    size=25,
                    color=color,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                text=labels,
                textposition='top center',
                textfont=dict(size=10, color='black', family='Arial Black'),
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=True
            )
            node_traces.append(node_trace)

        # Cria figura
        fig = go.Figure(data=edge_traces + node_traces)

        # Layout estilo Atlas
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Arial', color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend=dict(
                title=dict(text="Asset Types", font=dict(size=14, family='Arial')),
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#ccc",
                borderwidth=1
            ),
            hovermode='closest',
            margin=dict(b=40, l=40, r=200, t=100),
            xaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            font=dict(family='Arial', size=12)
        )

        # Adiciona setas como annotations
        annotations = []
        for edge in self.graph.edges():
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            # Seta no final da linha
            annotations.append(
                dict(
                    ax=x0,
                    ay=y0,
                    x=x1,
                    y=y1,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=1.5,
                    arrowcolor='#888',
                    standoff=10
                )
            )

        fig.update_layout(annotations=annotations)

        # Configura interatividade
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

        return fig

    def visualize_atlas_interactive(self,
                                    output_file: str = "lineage_atlas_interactive.html",
                                    initial_nodes: list = None,
                                    initial_levels: int = 2,
                                    title: str = None,
                                    language: str = 'pt') -> str:
        """
        Visualização Atlas interativa com expansão de níveis

        Args:
            output_file: Arquivo HTML de saída
            initial_nodes: Nós iniciais (None = mais importantes)
            initial_levels: Número inicial de níveis
            title: Título da visualização
            language: Idioma da interface ('pt' ou 'en')

        Returns:
            Caminho do arquivo gerado
        """
        from atlas_interactive import AtlasInteractiveVisualization

        viz = AtlasInteractiveVisualization(self.graph, language=language)
        return viz.generate_html(output_file, initial_nodes, initial_levels, title)
