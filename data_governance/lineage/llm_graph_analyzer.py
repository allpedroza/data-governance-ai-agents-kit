"""
LLM Graph Analysis and Natural Language Summarization Module
Generates automatic explanations, insights, and recommendations for data lineage graphs
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights that can be generated"""
    CRITICAL_PATH = "critical_path"
    BOTTLENECK = "bottleneck"
    SINGLE_POINT_FAILURE = "single_point_failure"
    DATA_SILO = "data_silo"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ORPHANED_NODE = "orphaned_node"
    HIGH_COMPLEXITY = "high_complexity"
    OPTIMIZATION_OPPORTUNITY = "optimization"
    COMPLIANCE_RISK = "compliance_risk"
    PERFORMANCE_ISSUE = "performance"


@dataclass
class GraphInsight:
    """Represents an insight about the graph"""
    type: InsightType
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    title: str
    description: str
    affected_nodes: List[str]
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'type': self.type.value,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'affected_nodes': self.affected_nodes,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


@dataclass
class SubgraphSummary:
    """Summary of a subgraph or component"""
    name: str
    domain: str
    node_count: int
    edge_count: int
    description: str
    purpose: str
    data_flow: str
    key_assets: List[str]
    insights: List[GraphInsight]
    metrics: Dict[str, Any]
    
    def to_dict(self):
        return {
            'name': self.name,
            'domain': self.domain,
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'description': self.description,
            'purpose': self.purpose,
            'data_flow': self.data_flow,
            'key_assets': self.key_assets,
            'insights': [i.to_dict() for i in self.insights],
            'metrics': self.metrics
        }


class GraphLLMAnalyzer:
    """
    Analyzes data lineage graphs and generates natural language summaries
    """
    
    def __init__(self, llm_client=None):
        """Initialize analyzer with optional LLM client"""
        self.llm_client = llm_client
        self.analysis_cache = {}
        
    def analyze_graph(
        self,
        graph: nx.DiGraph,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive graph analysis and generate summaries
        """
        if metadata is None:
            metadata = {}
        
        # Calculate base metrics
        metrics = self._calculate_graph_metrics(graph)
        
        # Identify subgraphs and components
        subgraphs = self._identify_subgraphs(graph, metadata)
        
        # Detect patterns and issues
        insights = self._detect_insights(graph, metrics)
        
        # Generate natural language summaries
        overall_summary = self._generate_overall_summary(graph, metrics, insights)
        
        # Generate subgraph summaries
        subgraph_summaries = []
        for subgraph in subgraphs:
            summary = self._analyze_subgraph(subgraph, graph, metadata)
            subgraph_summaries.append(summary)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights, metrics)
        
        # If LLM client available, enhance with AI
        if self.llm_client:
            overall_summary = self._enhance_with_llm(
                overall_summary, 
                graph, 
                metrics, 
                insights
            )
        
        return {
            'overall_summary': overall_summary,
            'metrics': metrics,
            'insights': [i.to_dict() for i in insights],
            'subgraph_summaries': [s.to_dict() for s in subgraph_summaries],
            'recommendations': recommendations,
            'natural_language_report': self._generate_full_report(
                overall_summary,
                metrics,
                insights,
                subgraph_summaries,
                recommendations
            )
        }
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics"""
        
        metrics = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'density': nx.density(graph) if graph.number_of_nodes() > 0 else 0,
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'connected_components': nx.number_weakly_connected_components(graph),
            'strongly_connected': nx.number_strongly_connected_components(graph)
        }
        
        # Node degree statistics
        if graph.number_of_nodes() > 0:
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            
            metrics['avg_in_degree'] = np.mean(list(in_degrees.values()))
            metrics['avg_out_degree'] = np.mean(list(out_degrees.values()))
            metrics['max_in_degree'] = max(in_degrees.values())
            metrics['max_out_degree'] = max(out_degrees.values())
            
            # Find hubs and authorities
            metrics['hubs'] = [
                node for node, degree in out_degrees.items() 
                if degree > metrics['avg_out_degree'] * 2
            ]
            metrics['sinks'] = [
                node for node, degree in out_degrees.items() 
                if degree == 0
            ]
            metrics['sources'] = [
                node for node, degree in in_degrees.items() 
                if degree == 0
            ]
            
            # Centrality measures
            try:
                betweenness = nx.betweenness_centrality(graph)
                metrics['high_betweenness_nodes'] = [
                    node for node, cent in betweenness.items()
                    if cent > np.percentile(list(betweenness.values()), 90)
                ]
            except:
                metrics['high_betweenness_nodes'] = []
            
            # Path analysis
            if metrics['is_dag']:
                try:
                    longest_path = nx.dag_longest_path(graph)
                    metrics['longest_path_length'] = len(longest_path)
                    metrics['longest_path'] = longest_path
                except:
                    metrics['longest_path_length'] = 0
                    metrics['longest_path'] = []
            
            # Detect cycles
            try:
                cycles = list(nx.simple_cycles(graph))
                metrics['has_cycles'] = len(cycles) > 0
                metrics['cycle_count'] = len(cycles)
                metrics['cycles'] = cycles[:5]  # Keep only first 5 cycles
            except:
                metrics['has_cycles'] = False
                metrics['cycle_count'] = 0
                metrics['cycles'] = []
        
        return metrics
    
    def _identify_subgraphs(
        self,
        graph: nx.DiGraph,
        metadata: Dict[str, Any]
    ) -> List[nx.DiGraph]:
        """Identify meaningful subgraphs (components, domains, clusters)"""
        
        subgraphs = []
        
        # 1. Weakly connected components
        components = list(nx.weakly_connected_components(graph))
        for comp in components:
            if len(comp) > 3:  # Only consider components with more than 3 nodes
                subgraph = graph.subgraph(comp).copy()
                subgraphs.append(subgraph)
        
        # 2. Domain-based subgraphs (if metadata available)
        if metadata and 'node_domains' in metadata:
            domains = defaultdict(set)
            for node, domain in metadata['node_domains'].items():
                if node in graph:
                    domains[domain].add(node)
            
            for domain, nodes in domains.items():
                if len(nodes) > 2:
                    subgraph = graph.subgraph(nodes).copy()
                    subgraph.graph['domain'] = domain
                    subgraphs.append(subgraph)
        
        # 3. Community detection (using modularity)
        try:
            if graph.number_of_edges() > 0:
                # Convert to undirected for community detection
                undirected = graph.to_undirected()
                communities = nx.community.greedy_modularity_communities(undirected)
                
                for community in communities:
                    if len(community) > 3:
                        subgraph = graph.subgraph(community).copy()
                        subgraphs.append(subgraph)
        except:
            pass
        
        return subgraphs[:10]  # Limit to 10 most significant subgraphs
    
    def _detect_insights(
        self,
        graph: nx.DiGraph,
        metrics: Dict[str, Any]
    ) -> List[GraphInsight]:
        """Detect patterns, issues, and insights in the graph"""
        
        insights = []
        
        # 1. Critical Path Analysis
        if metrics.get('longest_path_length', 0) > 10:
            insights.append(GraphInsight(
                type=InsightType.CRITICAL_PATH,
                severity="HIGH",
                title="Long Critical Path Detected",
                description=f"The pipeline has a critical path of {metrics['longest_path_length']} steps, "
                           f"which may impact processing time and reliability.",
                affected_nodes=metrics.get('longest_path', [])[:5],
                recommendation="Consider parallelizing independent transformations or breaking the pipeline "
                              "into smaller, more manageable stages.",
                metrics={'path_length': metrics['longest_path_length']}
            ))
        
        # 2. Single Points of Failure
        critical_nodes = self._find_single_points_of_failure(graph)
        if critical_nodes:
            insights.append(GraphInsight(
                type=InsightType.SINGLE_POINT_FAILURE,
                severity="CRITICAL",
                title=f"{len(critical_nodes)} Single Points of Failure",
                description=f"Found {len(critical_nodes)} nodes that, if failed, would break the pipeline. "
                           f"These nodes have high centrality and no alternative paths.",
                affected_nodes=critical_nodes[:5],
                recommendation="Add redundancy, implement retry logic, or create alternative data paths "
                              "for these critical nodes.",
                metrics={'critical_count': len(critical_nodes)}
            ))
        
        # 3. Bottlenecks
        bottlenecks = self._find_bottlenecks(graph, metrics)
        if bottlenecks:
            insights.append(GraphInsight(
                type=InsightType.BOTTLENECK,
                severity="HIGH",
                title=f"{len(bottlenecks)} Processing Bottlenecks",
                description=f"Identified {len(bottlenecks)} nodes with high fan-in/fan-out that may cause "
                           f"processing delays.",
                affected_nodes=bottlenecks[:5],
                recommendation="Consider scaling these nodes horizontally, optimizing their processing logic, "
                              "or implementing caching strategies.",
                metrics={'bottleneck_count': len(bottlenecks)}
            ))
        
        # 4. Circular Dependencies
        if metrics.get('has_cycles', False):
            insights.append(GraphInsight(
                type=InsightType.CIRCULAR_DEPENDENCY,
                severity="CRITICAL",
                title=f"{metrics['cycle_count']} Circular Dependencies",
                description=f"Found {metrics['cycle_count']} circular dependencies that could cause "
                           f"infinite loops or deadlocks.",
                affected_nodes=list(set(sum(metrics.get('cycles', [])[:3], [])))[:10],
                recommendation="Refactor the pipeline to eliminate cycles. Consider using staging tables "
                              "or breaking the circular logic.",
                metrics={'cycle_count': metrics['cycle_count']}
            ))
        
        # 5. Data Silos
        silos = self._find_data_silos(graph)
        if silos:
            insights.append(GraphInsight(
                type=InsightType.DATA_SILO,
                severity="MEDIUM",
                title=f"{len(silos)} Isolated Data Silos",
                description=f"Found {len(silos)} disconnected components that don't interact with "
                           f"the main pipeline.",
                affected_nodes=list(silos)[:10],
                recommendation="Consider integrating these silos with the main data flow or document "
                              "why they need to remain isolated.",
                metrics={'silo_count': len(silos)}
            ))
        
        # 6. Orphaned Nodes
        orphans = self._find_orphaned_nodes(graph)
        if orphans:
            insights.append(GraphInsight(
                type=InsightType.ORPHANED_NODE,
                severity="LOW",
                title=f"{len(orphans)} Orphaned Nodes",
                description=f"Found {len(orphans)} nodes with no connections, possibly unused or deprecated.",
                affected_nodes=orphans[:10],
                recommendation="Review and remove unused nodes or connect them properly to the pipeline.",
                metrics={'orphan_count': len(orphans)}
            ))
        
        # 7. High Complexity Areas
        complex_areas = self._find_high_complexity_areas(graph)
        if complex_areas:
            insights.append(GraphInsight(
                type=InsightType.HIGH_COMPLEXITY,
                severity="MEDIUM",
                title=f"{len(complex_areas)} High Complexity Areas",
                description=f"Found {len(complex_areas)} areas with dense connections that may be "
                           f"difficult to maintain.",
                affected_nodes=complex_areas[:10],
                recommendation="Consider refactoring complex areas into smaller, more modular components "
                              "with clear interfaces.",
                metrics={'complex_count': len(complex_areas)}
            ))
        
        # 8. Optimization Opportunities
        optimizations = self._find_optimization_opportunities(graph, metrics)
        for opt in optimizations:
            insights.append(opt)
        
        return sorted(insights, key=lambda x: self._severity_score(x.severity), reverse=True)
    
    def _find_single_points_of_failure(self, graph: nx.DiGraph) -> List[str]:
        """Find nodes that are critical single points of failure"""
        critical = []
        
        # Find articulation points (in undirected version)
        try:
            undirected = graph.to_undirected()
            articulation_points = list(nx.articulation_points(undirected))
            critical.extend(articulation_points)
        except:
            pass
        
        # Find nodes that are sole connection between components
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))
            successors = list(graph.successors(node))
            
            # Node with many dependencies and dependents
            if len(predecessors) > 3 and len(successors) > 3:
                # Check if removing this node disconnects the graph
                test_graph = graph.copy()
                test_graph.remove_node(node)
                
                if nx.number_weakly_connected_components(test_graph) > \
                   nx.number_weakly_connected_components(graph):
                    critical.append(node)
        
        return list(set(critical))
    
    def _find_bottlenecks(
        self,
        graph: nx.DiGraph,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Find processing bottlenecks in the graph"""
        bottlenecks = []
        
        avg_degree = (metrics.get('avg_in_degree', 1) + metrics.get('avg_out_degree', 1)) / 2
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            # High fan-in (many inputs)
            if in_degree > avg_degree * 3:
                bottlenecks.append(node)
            
            # High fan-out (many outputs)
            elif out_degree > avg_degree * 3:
                bottlenecks.append(node)
            
            # Hub node (both high in and out)
            elif in_degree > avg_degree * 2 and out_degree > avg_degree * 2:
                bottlenecks.append(node)
        
        return list(set(bottlenecks))
    
    def _find_data_silos(self, graph: nx.DiGraph) -> Set[str]:
        """Find isolated data silos"""
        silos = set()
        
        components = list(nx.weakly_connected_components(graph))
        
        if len(components) > 1:
            # Find the main component (largest)
            main_component = max(components, key=len)
            
            # All other components are potential silos
            for comp in components:
                if comp != main_component and len(comp) > 1:
                    silos.update(comp)
        
        return silos
    
    def _find_orphaned_nodes(self, graph: nx.DiGraph) -> List[str]:
        """Find nodes with no connections"""
        orphans = []
        
        for node in graph.nodes():
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                orphans.append(node)
        
        return orphans
    
    def _find_high_complexity_areas(self, graph: nx.DiGraph) -> List[str]:
        """Find areas of high complexity"""
        complex_nodes = []
        
        # Calculate local clustering coefficient
        undirected = graph.to_undirected()
        clustering = nx.clustering(undirected)
        
        # Find nodes with high clustering (dense local connections)
        threshold = np.percentile(list(clustering.values()), 80)
        
        for node, coeff in clustering.items():
            if coeff > threshold:
                # Also check degree
                total_degree = graph.in_degree(node) + graph.out_degree(node)
                if total_degree > 5:
                    complex_nodes.append(node)
        
        return complex_nodes
    
    def _find_optimization_opportunities(
        self,
        graph: nx.DiGraph,
        metrics: Dict[str, Any]
    ) -> List[GraphInsight]:
        """Find optimization opportunities"""
        opportunities = []
        
        # 1. Parallel Processing Opportunity
        independent_paths = self._find_independent_paths(graph)
        if len(independent_paths) > 3:
            opportunities.append(GraphInsight(
                type=InsightType.OPTIMIZATION_OPPORTUNITY,
                severity="MEDIUM",
                title="Parallel Processing Opportunity",
                description=f"Found {len(independent_paths)} independent paths that could be "
                           f"processed in parallel.",
                affected_nodes=list(set(sum(independent_paths[:3], [])))[:10],
                recommendation="Implement parallel processing for these independent paths to "
                              "reduce overall pipeline execution time.",
                metrics={'parallel_paths': len(independent_paths)}
            ))
        
        # 2. Caching Opportunity
        frequently_accessed = self._find_frequently_accessed_nodes(graph)
        if frequently_accessed:
            opportunities.append(GraphInsight(
                type=InsightType.OPTIMIZATION_OPPORTUNITY,
                severity="MEDIUM",
                title="Caching Opportunity",
                description=f"Found {len(frequently_accessed)} frequently accessed nodes that "
                           f"could benefit from caching.",
                affected_nodes=frequently_accessed[:5],
                recommendation="Implement caching for these frequently accessed data sources "
                              "to reduce redundant computations.",
                metrics={'cacheable_nodes': len(frequently_accessed)}
            ))
        
        # 3. Denormalization Opportunity
        long_join_chains = self._find_long_join_chains(graph)
        if long_join_chains:
            opportunities.append(GraphInsight(
                type=InsightType.OPTIMIZATION_OPPORTUNITY,
                severity="LOW",
                title="Denormalization Opportunity",
                description=f"Found {len(long_join_chains)} long join chains that might benefit "
                           f"from denormalization.",
                affected_nodes=long_join_chains[:5],
                recommendation="Consider denormalizing frequently joined tables to reduce "
                              "query complexity and improve performance.",
                metrics={'join_chains': len(long_join_chains)}
            ))
        
        return opportunities
    
    def _find_independent_paths(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find paths that can be executed independently"""
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        independent_paths = []
        
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(nx.all_simple_paths(graph, source, sink, cutoff=10))
                    if paths:
                        # Check if path is independent
                        path_nodes = set(paths[0])
                        is_independent = True
                        
                        for other_path in independent_paths:
                            if set(other_path) & path_nodes:
                                is_independent = False
                                break
                        
                        if is_independent and len(paths[0]) > 2:
                            independent_paths.append(paths[0])
                except:
                    pass
        
        return independent_paths[:10]
    
    def _find_frequently_accessed_nodes(self, graph: nx.DiGraph) -> List[str]:
        """Find nodes that are frequently accessed"""
        access_count = defaultdict(int)
        
        for node in graph.nodes():
            # Count incoming connections as accesses
            access_count[node] = graph.in_degree(node)
        
        # Find nodes with above-average access
        avg_access = np.mean(list(access_count.values())) if access_count else 0
        frequent = [
            node for node, count in access_count.items()
            if count > avg_access * 2
        ]
        
        return sorted(frequent, key=lambda x: access_count[x], reverse=True)
    
    def _find_long_join_chains(self, graph: nx.DiGraph) -> List[str]:
        """Find long chains of joins that might benefit from denormalization"""
        join_chains = []
        
        # Look for linear paths longer than 4 nodes
        for node in graph.nodes():
            if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
                # Part of a potential chain
                path = [node]
                current = node
                
                # Follow the chain forward
                while graph.out_degree(current) == 1:
                    successors = list(graph.successors(current))
                    if successors and graph.in_degree(successors[0]) == 1:
                        current = successors[0]
                        path.append(current)
                    else:
                        break
                
                if len(path) > 4:
                    join_chains.extend(path)
        
        return list(set(join_chains))
    
    def _analyze_subgraph(
        self,
        subgraph: nx.DiGraph,
        parent_graph: nx.DiGraph,
        metadata: Dict[str, Any]
    ) -> SubgraphSummary:
        """Analyze a specific subgraph and generate summary"""
        
        # Calculate subgraph metrics
        metrics = self._calculate_graph_metrics(subgraph)
        
        # Detect subgraph-specific insights
        insights = self._detect_insights(subgraph, metrics)
        
        # Determine domain/name
        domain = subgraph.graph.get('domain', 'Unknown')
        nodes = list(subgraph.nodes())
        
        # Find common prefix/pattern in node names
        name = self._infer_subgraph_name(nodes)
        
        # Analyze data flow pattern
        flow_pattern = self._analyze_flow_pattern(subgraph)
        
        # Generate descriptions
        description = self._generate_subgraph_description(
            subgraph, metrics, flow_pattern
        )
        
        purpose = self._infer_purpose(nodes, metadata)
        
        data_flow = self._describe_data_flow(subgraph, flow_pattern)
        
        # Identify key assets
        key_assets = self._identify_key_assets(subgraph)
        
        return SubgraphSummary(
            name=name,
            domain=domain,
            node_count=subgraph.number_of_nodes(),
            edge_count=subgraph.number_of_edges(),
            description=description,
            purpose=purpose,
            data_flow=data_flow,
            key_assets=key_assets,
            insights=insights[:3],  # Top 3 insights
            metrics=metrics
        )
    
    def _infer_subgraph_name(self, nodes: List[str]) -> str:
        """Infer a meaningful name for the subgraph"""
        if not nodes:
            return "Empty Subgraph"
        
        # Find common prefixes
        if len(nodes) > 1:
            common = self._longest_common_prefix(nodes)
            if common and len(common) > 3:
                return f"{common.strip('_.-')} Component"
        
        # Find common patterns
        patterns = {
            'etl': 'ETL Pipeline',
            'transform': 'Transformation Layer',
            'staging': 'Staging Area',
            'raw': 'Raw Data Ingestion',
            'aggregate': 'Aggregation Pipeline',
            'dim': 'Dimension Tables',
            'fact': 'Fact Tables',
            'report': 'Reporting Layer',
            'ml': 'ML Pipeline'
        }
        
        for pattern, name in patterns.items():
            if any(pattern in node.lower() for node in nodes):
                return name
        
        return f"Component ({len(nodes)} nodes)"
    
    def _longest_common_prefix(self, strings: List[str]) -> str:
        """Find longest common prefix of strings"""
        if not strings:
            return ""
        
        shortest = min(strings, key=len)
        
        for i, char in enumerate(shortest):
            for other in strings:
                if other[i] != char:
                    return shortest[:i]
        
        return shortest
    
    def _analyze_flow_pattern(self, graph: nx.DiGraph) -> str:
        """Analyze the flow pattern of the graph"""
        if graph.number_of_nodes() == 0:
            return "empty"
        
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        if len(sources) == 1 and len(sinks) == 1:
            return "linear"
        elif len(sources) > 1 and len(sinks) == 1:
            return "convergent"
        elif len(sources) == 1 and len(sinks) > 1:
            return "divergent"
        elif len(sources) > 1 and len(sinks) > 1:
            return "mesh"
        else:
            return "complex"
    
    def _generate_subgraph_description(
        self,
        graph: nx.DiGraph,
        metrics: Dict[str, Any],
        flow_pattern: str
    ) -> str:
        """Generate natural language description of subgraph"""
        
        descriptions = {
            'linear': f"A linear pipeline with {graph.number_of_nodes()} sequential steps, "
                     f"processing data from source to sink in a straightforward flow.",
            'convergent': f"A convergent pipeline that combines {len(metrics.get('sources', []))} "
                         f"data sources into a unified output through {graph.number_of_edges()} transformations.",
            'divergent': f"A divergent pipeline that distributes data from a single source to "
                        f"{len(metrics.get('sinks', []))} different outputs.",
            'mesh': f"A complex mesh topology with {graph.number_of_nodes()} nodes and "
                   f"{graph.number_of_edges()} connections, indicating sophisticated data processing.",
            'complex': f"A complex component with {graph.number_of_nodes()} nodes arranged in "
                      f"an intricate pattern with {metrics.get('connected_components', 1)} connected parts."
        }
        
        base_desc = descriptions.get(flow_pattern, descriptions['complex'])
        
        # Add specific characteristics
        if metrics.get('has_cycles', False):
            base_desc += " Contains circular dependencies that may indicate iterative processing."
        
        if metrics.get('high_betweenness_nodes'):
            base_desc += f" Has {len(metrics['high_betweenness_nodes'])} critical nodes that serve as key connectors."
        
        return base_desc
    
    def _infer_purpose(self, nodes: List[str], metadata: Dict[str, Any]) -> str:
        """Infer the purpose of a subgraph based on node names and metadata"""
        
        node_names = ' '.join(nodes).lower()
        
        purposes = {
            'extract': "Data extraction from source systems",
            'transform': "Data transformation and enrichment",
            'load': "Data loading into target systems",
            'validate': "Data quality validation and checks",
            'aggregate': "Data aggregation and summarization",
            'join': "Data integration and joining",
            'filter': "Data filtering and selection",
            'clean': "Data cleansing and standardization",
            'enrich': "Data enrichment with additional attributes",
            'stage': "Data staging for further processing",
            'archive': "Data archival and historical storage",
            'report': "Report generation and visualization",
            'ml': "Machine learning pipeline",
            'feature': "Feature engineering for ML",
            'model': "Model training or scoring"
        }
        
        for keyword, purpose in purposes.items():
            if keyword in node_names:
                return purpose
        
        # Check metadata
        if metadata and 'purpose' in metadata:
            return metadata['purpose']
        
        return "Data processing and transformation"
    
    def _describe_data_flow(self, graph: nx.DiGraph, pattern: str) -> str:
        """Describe how data flows through the subgraph"""
        
        if graph.number_of_nodes() == 0:
            return "No data flow detected."
        
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        # Get sample path
        if sources and sinks:
            try:
                sample_path = nx.shortest_path(graph, sources[0], sinks[0])
                path_desc = " → ".join(sample_path[:5])
                if len(sample_path) > 5:
                    path_desc += f" → ... ({len(sample_path) - 5} more steps)"
                
                return f"Data flows from {sources[0]} through {len(sample_path) - 2} intermediate " \
                       f"transformations to {sinks[0]}. Path: {path_desc}"
            except:
                pass
        
        # Fallback descriptions
        if pattern == "linear":
            return f"Sequential data flow through {graph.number_of_nodes()} processing stages."
        elif pattern == "convergent":
            return f"Multiple data streams ({len(sources)}) converge through transformations into a single output."
        elif pattern == "divergent":
            return f"Single data source branches into {len(sinks)} different processing paths."
        else:
            return f"Complex data flow with {graph.number_of_edges()} connections between {graph.number_of_nodes()} processing nodes."
    
    def _identify_key_assets(self, graph: nx.DiGraph) -> List[str]:
        """Identify the most important assets in the subgraph"""
        
        key_assets = []
        
        # Sources and sinks are usually important
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        
        key_assets.extend(sources[:2])
        key_assets.extend(sinks[:2])
        
        # High centrality nodes
        if graph.number_of_nodes() > 5:
            try:
                centrality = nx.degree_centrality(graph)
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                key_assets.extend([n for n, _ in top_central[:3]])
            except:
                pass
        
        # Remove duplicates and limit
        return list(dict.fromkeys(key_assets))[:5]
    
    def _generate_overall_summary(
        self,
        graph: nx.DiGraph,
        metrics: Dict[str, Any],
        insights: List[GraphInsight]
    ) -> str:
        """Generate overall summary of the entire graph"""
        
        summary_parts = []
        
        # Opening statement
        summary_parts.append(
            f"This data pipeline consists of {metrics['total_nodes']} data assets connected by "
            f"{metrics['total_edges']} transformations, organized into "
            f"{metrics['connected_components']} main component(s)."
        )
        
        # Topology description
        if metrics.get('is_dag', False):
            summary_parts.append(
                "The pipeline follows a directed acyclic graph (DAG) structure, ensuring no circular "
                "dependencies and predictable data flow."
            )
        else:
            summary_parts.append(
                f"The pipeline contains {metrics.get('cycle_count', 0)} circular dependencies that "
                f"require careful management to avoid infinite loops."
            )
        
        # Complexity assessment
        density = metrics.get('density', 0)
        if density < 0.1:
            complexity = "sparse"
        elif density < 0.3:
            complexity = "moderate"
        else:
            complexity = "dense"
        
        summary_parts.append(
            f"The graph has {complexity} connectivity (density: {density:.2f}), "
            f"with an average of {metrics.get('avg_in_degree', 0):.1f} inputs and "
            f"{metrics.get('avg_out_degree', 0):.1f} outputs per node."
        )
        
        # Key characteristics
        if metrics.get('sources'):
            summary_parts.append(
                f"Data enters the pipeline through {len(metrics['sources'])} source points and "
                f"exits through {len(metrics.get('sinks', []))} sink points."
            )
        
        # Critical insights
        if insights:
            critical_insights = [i for i in insights if i.severity in ['CRITICAL', 'HIGH']]
            if critical_insights:
                summary_parts.append(
                    f"Critical attention needed: {len(critical_insights)} high-priority issues detected, "
                    f"including {', '.join(set(i.type.value.replace('_', ' ') for i in critical_insights[:3]))}."
                )
        
        # Performance implications
        if metrics.get('longest_path_length', 0) > 0:
            summary_parts.append(
                f"The longest processing path contains {metrics['longest_path_length']} steps, "
                f"which determines the minimum processing time for end-to-end data flow."
            )
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(
        self,
        insights: List[GraphInsight],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on insights"""
        
        recommendations = []
        
        # Group insights by type for consolidated recommendations
        insight_groups = defaultdict(list)
        for insight in insights:
            insight_groups[insight.type].append(insight)
        
        # Priority 1: Critical Issues
        critical_insights = [i for i in insights if i.severity == "CRITICAL"]
        if critical_insights:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'title': 'Address Critical Issues',
                'description': f"Found {len(critical_insights)} critical issues that require immediate attention.",
                'actions': [i.recommendation for i in critical_insights[:3]],
                'impact': 'Preventing pipeline failures and data loss',
                'effort': 'High',
                'roi': 'Critical for system stability'
            })
        
        # Priority 2: Performance Optimizations
        perf_insights = insight_groups.get(InsightType.OPTIMIZATION_OPPORTUNITY, [])
        bottleneck_insights = insight_groups.get(InsightType.BOTTLENECK, [])
        
        if perf_insights or bottleneck_insights:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Optimize Performance',
                'description': f"Identified {len(perf_insights) + len(bottleneck_insights)} opportunities "
                              f"to improve pipeline performance.",
                'actions': [
                    "Implement parallel processing for independent paths",
                    "Add caching for frequently accessed data",
                    "Optimize bottleneck nodes with horizontal scaling"
                ],
                'impact': '30-50% reduction in processing time',
                'effort': 'Medium',
                'roi': 'High - immediate performance gains'
            })
        
        # Priority 3: Architecture Improvements
        if metrics.get('connected_components', 1) > 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Consolidate Pipeline Architecture',
                'description': f"The pipeline is fragmented into {metrics['connected_components']} separate components.",
                'actions': [
                    "Review and consolidate disconnected components",
                    "Implement a unified orchestration layer",
                    "Standardize data interfaces between components"
                ],
                'impact': 'Improved maintainability and monitoring',
                'effort': 'High',
                'roi': 'Long-term maintenance cost reduction'
            })
        
        # Priority 4: Data Quality
        quality_issues = insight_groups.get(InsightType.ORPHANED_NODE, [])
        if quality_issues or not metrics.get('is_dag', True):
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Improve Data Quality Controls',
                'description': "Enhance data quality and validation throughout the pipeline.",
                'actions': [
                    "Add data validation checkpoints at key stages",
                    "Implement data quality metrics and monitoring",
                    "Create data quality dashboards",
                    "Add automated testing for data transformations"
                ],
                'impact': 'Increased data reliability and trust',
                'effort': 'Medium',
                'roi': 'Reduces downstream data issues'
            })
        
        # Priority 5: Documentation and Governance
        if metrics['total_nodes'] > 50:
            recommendations.append({
                'priority': 'LOW',
                'title': 'Enhance Documentation and Governance',
                'description': "Large pipeline requires comprehensive documentation and governance.",
                'actions': [
                    "Create detailed data dictionary for all assets",
                    "Document business logic and transformation rules",
                    "Implement data lineage tracking",
                    "Establish data ownership and stewardship"
                ],
                'impact': 'Improved team efficiency and compliance',
                'effort': 'Low',
                'roi': 'Long-term team productivity gains'
            })
        
        return sorted(recommendations, key=lambda x: self._priority_score(x['priority']), reverse=True)
    
    def _generate_full_report(
        self,
        summary: str,
        metrics: Dict[str, Any],
        insights: List[GraphInsight],
        subgraphs: List[SubgraphSummary],
        recommendations: List[Dict[str, Any]]
    ) -> str:
        """Generate a complete natural language report"""
        
        report = []
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(summary)
        report.append("\n")
        
        # Key Metrics
        report.append("## Key Metrics\n")
        report.append(f"- **Total Assets**: {metrics['total_nodes']}")
        report.append(f"- **Total Transformations**: {metrics['total_edges']}")
        report.append(f"- **Pipeline Complexity**: {self._describe_complexity(metrics)}")
        report.append(f"- **Critical Path Length**: {metrics.get('longest_path_length', 'N/A')} steps")
        report.append(f"- **Parallelization Potential**: {self._estimate_parallelization(metrics)}%")
        report.append("\n")
        
        # Critical Findings
        critical_insights = [i for i in insights if i.severity in ['CRITICAL', 'HIGH']]
        if critical_insights:
            report.append("## Critical Findings\n")
            for i, insight in enumerate(critical_insights[:5], 1):
                report.append(f"### {i}. {insight.title}")
                report.append(f"**Severity**: {insight.severity}")
                report.append(f"**Description**: {insight.description}")
                report.append(f"**Affected Components**: {len(insight.affected_nodes)} assets")
                report.append(f"**Recommendation**: {insight.recommendation}")
                report.append("")
        
        # Component Analysis
        if subgraphs:
            report.append("## Component Analysis\n")
            for i, subgraph in enumerate(subgraphs[:5], 1):
                report.append(f"### Component {i}: {subgraph.name}")
                report.append(f"- **Size**: {subgraph.node_count} nodes, {subgraph.edge_count} edges")
                report.append(f"- **Purpose**: {subgraph.purpose}")
                report.append(f"- **Data Flow**: {subgraph.data_flow}")
                if subgraph.insights:
                    report.append(f"- **Key Issue**: {subgraph.insights[0].title}")
                report.append("")
        
        # Recommendations
        if recommendations:
            report.append("## Recommendations\n")
            for i, rec in enumerate(recommendations[:5], 1):
                report.append(f"### {i}. {rec['title']} (Priority: {rec['priority']})")
                report.append(f"**Rationale**: {rec['description']}")
                report.append("**Actions**:")
                for action in rec['actions'][:3]:
                    report.append(f"- {action}")
                report.append(f"**Expected Impact**: {rec['impact']}")
                report.append(f"**Effort Required**: {rec['effort']}")
                report.append("")
        
        # Next Steps
        report.append("## Next Steps\n")
        report.append("1. Review and prioritize critical issues identified in this report")
        report.append("2. Create implementation plan for high-priority recommendations")
        report.append("3. Establish monitoring for key pipeline metrics")
        report.append("4. Schedule regular pipeline health reviews")
        report.append("5. Document implemented changes and their impact")
        
        return "\n".join(report)
    
    def _enhance_with_llm(
        self,
        summary: str,
        graph: nx.DiGraph,
        metrics: Dict[str, Any],
        insights: List[GraphInsight]
    ) -> str:
        """Enhance summary with LLM if available"""
        
        if not self.llm_client:
            return summary
        
        # Prepare context for LLM
        context = {
            'current_summary': summary,
            'metrics': {
                'nodes': metrics['total_nodes'],
                'edges': metrics['total_edges'],
                'is_dag': metrics.get('is_dag', False),
                'cycles': metrics.get('cycle_count', 0)
            },
            'top_issues': [
                {'type': i.type.value, 'severity': i.severity}
                for i in insights[:5]
            ]
        }
        
        prompt = f"""
        Enhance this data pipeline summary with more insightful analysis:
        
        Current summary: {summary}
        
        Context: {json.dumps(context, indent=2)}
        
        Provide a more comprehensive summary that includes:
        1. Business impact assessment
        2. Risk evaluation
        3. Optimization potential
        4. Strategic recommendations
        
        Keep it concise (3-4 paragraphs).
        """
        
        try:
            # This would call your LLM service
            # enhanced = self.llm_client.generate(prompt)
            # return enhanced
            pass
        except:
            pass
        
        return summary
    
    # Helper methods
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score"""
        scores = {
            'CRITICAL': 4,
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        return scores.get(severity, 0)
    
    def _priority_score(self, priority: str) -> int:
        """Convert priority to numeric score"""
        scores = {
            'IMMEDIATE': 4,
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        return scores.get(priority, 0)
    
    def _describe_complexity(self, metrics: Dict[str, Any]) -> str:
        """Describe pipeline complexity in natural language"""
        
        score = 0
        
        # Factor in various complexity indicators
        if metrics['total_nodes'] > 100:
            score += 3
        elif metrics['total_nodes'] > 50:
            score += 2
        elif metrics['total_nodes'] > 20:
            score += 1
        
        if not metrics.get('is_dag', True):
            score += 2
        
        if metrics.get('density', 0) > 0.3:
            score += 2
        elif metrics.get('density', 0) > 0.15:
            score += 1
        
        if metrics.get('connected_components', 1) > 3:
            score += 1
        
        if score >= 6:
            return "Very High - Requires architectural review"
        elif score >= 4:
            return "High - Consider refactoring"
        elif score >= 2:
            return "Moderate - Manageable with good practices"
        else:
            return "Low - Well-structured pipeline"
    
    def _estimate_parallelization(self, metrics: Dict[str, Any]) -> int:
        """Estimate parallelization potential as percentage"""
        
        if metrics['total_nodes'] == 0:
            return 0
        
        # Estimate based on graph structure
        sources = len(metrics.get('sources', []))
        sinks = len(metrics.get('sinks', []))
        
        # Multiple sources or sinks indicate parallelization potential
        if sources > 1 or sinks > 1:
            base_parallel = 30
        else:
            base_parallel = 10
        
        # Low density means more independent paths
        density = metrics.get('density', 1)
        if density < 0.1:
            base_parallel += 40
        elif density < 0.2:
            base_parallel += 20
        
        # Multiple components can be parallelized
        if metrics.get('connected_components', 1) > 1:
            base_parallel += 20
        
        return min(base_parallel, 90)


# Integration function for existing system
def integrate_llm_summaries(graph, metrics, llm_client=None):
    """
    Easy integration function for existing data lineage system
    
    Args:
        graph: NetworkX DiGraph of data lineage
        metrics: Pre-calculated metrics dictionary
        llm_client: Optional LLM client for enhanced analysis
    
    Returns:
        Dictionary with summaries, insights, and recommendations
    """
    
    analyzer = GraphLLMAnalyzer(llm_client)
    
    # Perform analysis
    results = analyzer.analyze_graph(graph, metrics)
    
    return results


# Standalone HTML report generator
def generate_enhanced_html_report(
    graph,
    metrics,
    llm_analysis,
    output_file='lineage_report.html'
):
    """Generate enhanced HTML report with LLM summaries"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Lineage Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .summary-box {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .insight {{
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }}
            .insight.critical {{
                border-left: 5px solid #e74c3c;
                background: #fff5f5;
            }}
            .insight.high {{
                border-left: 5px solid #f39c12;
                background: #fffaf0;
            }}
            .insight.medium {{
                border-left: 5px solid #3498db;
                background: #f0f8ff;
            }}
            .recommendation {{
                background: #e8f5e9;
                border: 1px solid #4caf50;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendation h3 {{
                color: #2e7d32;
                margin-top: 0;
            }}
            .subgraph {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .badge {{
                display: inline-block;
                padding: 3px 10px;
                border-radius: 15px;
                font-size: 0.85em;
                margin: 2px;
            }}
            .badge.critical {{ background: #e74c3c; color: white; }}
            .badge.high {{ background: #f39c12; color: white; }}
            .badge.medium {{ background: #3498db; color: white; }}
            .badge.low {{ background: #95a5a6; color: white; }}
            pre {{
                background: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .action-list {{
                list-style: none;
                padding: 0;
            }}
            .action-list li {{
                padding: 10px;
                margin: 5px 0;
                background: #f8f9fa;
                border-left: 3px solid #667eea;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Data Lineage Analysis Report</h1>
            <p>AI-Powered Pipeline Intelligence</p>
        </div>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>{llm_analysis.get('overall_summary', 'No summary available')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('total_nodes', 0)}</div>
                <div class="metric-label">Total Assets</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('total_edges', 0)}</div>
                <div class="metric-label">Transformations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('connected_components', 0)}</div>
                <div class="metric-label">Components</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('longest_path_length', 0)}</div>
                <div class="metric-label">Critical Path</div>
            </div>
        </div>
        
        <div class="summary-box">
            <h2>🔍 Key Insights</h2>
            {generate_insights_html(llm_analysis.get('insights', []))}
        </div>
        
        <div class="summary-box">
            <h2>📋 Recommendations</h2>
            {generate_recommendations_html(llm_analysis.get('recommendations', []))}
        </div>
        
        <div class="summary-box">
            <h2>🔗 Component Analysis</h2>
            {generate_subgraphs_html(llm_analysis.get('subgraph_summaries', []))}
        </div>
        
        <div class="summary-box">
            <h2>📝 Natural Language Report</h2>
            <pre>{llm_analysis.get('natural_language_report', 'No detailed report available')}</pre>
        </div>
        
        <div style="text-align: center; color: #666; margin-top: 40px;">
            <small>Generated with AI-Enhanced Data Lineage Analysis</small>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file


def generate_insights_html(insights):
    """Generate HTML for insights section"""
    if not insights:
        return "<p>No significant insights detected.</p>"
    
    html = []
    for insight in insights[:10]:
        severity_class = insight['severity'].lower()
        html.append(f"""
        <div class="insight {severity_class}">
            <h3>{insight['title']} <span class="badge {severity_class}">{insight['severity']}</span></h3>
            <p>{insight['description']}</p>
            <p><strong>Affected nodes:</strong> {len(insight.get('affected_nodes', []))} assets</p>
            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
        </div>
        """)
    
    return ''.join(html)


def generate_recommendations_html(recommendations):
    """Generate HTML for recommendations section"""
    if not recommendations:
        return "<p>No specific recommendations at this time.</p>"
    
    html = []
    for rec in recommendations[:5]:
        html.append(f"""
        <div class="recommendation">
            <h3>{rec['title']}</h3>
            <p><strong>Priority:</strong> <span class="badge {rec['priority'].lower()}">{rec['priority']}</span></p>
            <p>{rec['description']}</p>
            <h4>Actions:</h4>
            <ul class="action-list">
                {''.join(f"<li>{action}</li>" for action in rec.get('actions', [])[:3])}
            </ul>
            <p><strong>Expected Impact:</strong> {rec.get('impact', 'N/A')}</p>
            <p><strong>Effort:</strong> {rec.get('effort', 'N/A')}</p>
        </div>
        """)
    
    return ''.join(html)


def generate_subgraphs_html(subgraphs):
    """Generate HTML for subgraphs section"""
    if not subgraphs:
        return "<p>No distinct components identified.</p>"
    
    html = []
    for sg in subgraphs[:5]:
        html.append(f"""
        <div class="subgraph">
            <h3>{sg['name']}</h3>
            <p><strong>Domain:</strong> {sg['domain']}</p>
            <p><strong>Size:</strong> {sg['node_count']} nodes, {sg['edge_count']} edges</p>
            <p><strong>Purpose:</strong> {sg['purpose']}</p>
            <p><strong>Data Flow:</strong> {sg['data_flow']}</p>
            <p><strong>Key Assets:</strong> {', '.join(sg.get('key_assets', [])[:5])}</p>
        </div>
        """)
    
    return ''.join(html)


if __name__ == "__main__":
    # Example usage
    import networkx as nx
    
    # Create sample graph
    G = nx.DiGraph()
    G.add_edges_from([
        ('raw_data', 'staging'),
        ('staging', 'transform_1'),
        ('staging', 'transform_2'),
        ('transform_1', 'aggregate'),
        ('transform_2', 'aggregate'),
        ('aggregate', 'report'),
        ('aggregate', 'dashboard')
    ])
    
    # Create analyzer
    analyzer = GraphLLMAnalyzer()
    
    # Analyze
    results = analyzer.analyze_graph(G)
    
    # Print results
    print("=" * 60)
    print("NATURAL LANGUAGE SUMMARY")
    print("=" * 60)
    print(results['overall_summary'])
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    for insight in results['insights'][:3]:
        print(f"- {insight['title']} ({insight['severity']})")
        print(f"  {insight['description']}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    for rec in results['recommendations'][:3]:
        print(f"- {rec['title']} (Priority: {rec['priority']})")
        print(f"  {rec['description']}")
