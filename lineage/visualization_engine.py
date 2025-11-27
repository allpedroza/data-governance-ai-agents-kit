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
        Exporta visualização para arquivo HTML interativo
        """
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
