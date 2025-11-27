"""
Data Lineage Agent - Sistema de IA para Análise de Linhagem de Dados
Suporta: Python, SQL, Terraform, Databricks
Autor: Claude AI Assistant
"""

import ast
import re
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token
from sqlparse.tokens import Keyword, DML
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import difflib


@dataclass
class DataAsset:
    """Representa um ativo de dados no pipeline"""
    name: str
    type: str  # table, view, file, stream, dataset, etc
    source_file: str
    line_number: int = 0
    schema: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(f"{self.name}_{self.type}")
    
    def __eq__(self, other):
        if not isinstance(other, DataAsset):
            return False
        return self.name == other.name and self.type == other.type


@dataclass
class Transformation:
    """Representa uma transformação entre ativos"""
    source: DataAsset
    target: DataAsset
    operation: str  # SELECT, INSERT, UPDATE, CREATE, etc
    transformation_logic: str
    source_file: str
    line_number: int = 0
    confidence_score: float = 1.0
    
    def __hash__(self):
        return hash(f"{self.source.name}_{self.target.name}_{self.operation}")


class DataLineageAgent:
    """
    Agente de IA para análise de linhagem de dados
    Extrai, analisa e visualiza dependências em pipelines de dados
    """
    
    def __init__(self):
        self.assets: Dict[str, DataAsset] = {}
        self.transformations: List[Transformation] = []
        self.graph = nx.DiGraph()
        self.impact_analysis_cache = {}
        self.parsers = {
            '.py': self._parse_python,
            '.sql': self._parse_sql,
            '.tf': self._parse_terraform,
            '.json': self._parse_terraform,  # terraform json
            '.scala': self._parse_databricks,
            '.py': self._parse_databricks_python,
            '.sql': self._parse_databricks_sql
        }
        
    def analyze_pipeline(self, file_paths: List[str]) -> Dict:
        """
        Analisa um pipeline completo de dados
        """
        print(f"🔍 Iniciando análise de {len(file_paths)} arquivos...")
        
        # Reset state
        self.assets.clear()
        self.transformations.clear()
        self.graph.clear()
        
        # Processa cada arquivo
        for file_path in file_paths:
            self._process_file(file_path)
        
        # Constrói o grafo de linhagem
        self._build_lineage_graph()
        
        # Calcula métricas
        metrics = self._calculate_metrics()
        
        return {
            'assets': list(self.assets.values()),
            'transformations': self.transformations,
            'metrics': metrics,
            'graph': self.graph
        }
    
    def _process_file(self, file_path: str):
        """Processa um arquivo individual"""
        ext = Path(file_path).suffix.lower()
        
        if ext not in self.parsers:
            print(f"⚠️ Formato não suportado: {ext}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser = self.parsers[ext]
            parser(content, file_path)
            print(f"✅ Processado: {file_path}")
            
        except Exception as e:
            print(f"❌ Erro ao processar {file_path}: {e}")
    
    def _parse_python(self, content: str, file_path: str):
        """Parser para arquivos Python"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Detecta imports de bibliotecas de dados
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(lib in alias.name for lib in ['pandas', 'pyspark', 'dask', 'polars']):
                            self._extract_python_data_operations(tree, file_path)
                            
                # Detecta operações com dataframes
                elif isinstance(node, ast.Call):
                    self._analyze_python_call(node, file_path)
                    
        except Exception as e:
            print(f"Erro no parser Python: {e}")
    
    def _extract_python_data_operations(self, tree: ast.AST, file_path: str):
        """Extrai operações de dados em Python"""
        class DataOperationVisitor(ast.NodeVisitor):
            def __init__(self, agent, file_path):
                self.agent = agent
                self.file_path = file_path
                
            def visit_Call(self, node):
                # read_csv, read_parquet, read_sql, etc
                if hasattr(node.func, 'attr'):
                    if 'read' in node.func.attr:
                        self._extract_read_operation(node)
                    elif 'write' in node.func.attr or 'to_' in node.func.attr:
                        self._extract_write_operation(node)
                        
                self.generic_visit(node)
                
            def _extract_read_operation(self, node):
                # Extrai origem de dados
                if node.args:
                    source_name = self._extract_string_from_node(node.args[0])
                    if source_name:
                        asset = DataAsset(
                            name=source_name,
                            type='file' if '.' in source_name else 'table',
                            source_file=self.file_path,
                            line_number=node.lineno if hasattr(node, 'lineno') else 0
                        )
                        self.agent.assets[source_name] = asset
                        
            def _extract_write_operation(self, node):
                # Extrai destino de dados
                if node.args:
                    target_name = self._extract_string_from_node(node.args[0])
                    if target_name:
                        asset = DataAsset(
                            name=target_name,
                            type='file' if '.' in target_name else 'table',
                            source_file=self.file_path,
                            line_number=node.lineno if hasattr(node, 'lineno') else 0
                        )
                        self.agent.assets[target_name] = asset
                        
            def _extract_string_from_node(self, node):
                if isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.Constant):
                    return str(node.value)
                return None
                
        visitor = DataOperationVisitor(self, file_path)
        visitor.visit(tree)
    
    def _analyze_python_call(self, node: ast.Call, file_path: str):
        """Analisa chamadas de função em Python"""
        # Detecta operações Spark
        if hasattr(node.func, 'attr'):
            spark_ops = ['select', 'filter', 'join', 'groupBy', 'agg', 'createOrReplaceTempView']
            if node.func.attr in spark_ops:
                # Registra transformação
                pass  # Implementar lógica detalhada
    
    def _parse_sql(self, content: str, file_path: str):
        """Parser para arquivos SQL"""
        statements = sqlparse.split(content)
        
        for statement in statements:
            parsed = sqlparse.parse(statement)[0] if sqlparse.parse(statement) else None
            if not parsed:
                continue
                
            # Extrai tabelas fonte e destino
            sources = self._extract_sql_sources(parsed)
            targets = self._extract_sql_targets(parsed)
            
            # Registra assets
            for source in sources:
                if source not in self.assets:
                    self.assets[source] = DataAsset(
                        name=source,
                        type='table',
                        source_file=file_path
                    )
                    
            for target in targets:
                if target not in self.assets:
                    self.assets[target] = DataAsset(
                        name=target,
                        type='table',
                        source_file=file_path
                    )
                    
            # Registra transformações
            for source in sources:
                for target in targets:
                    if source != target:
                        trans = Transformation(
                            source=self.assets[source],
                            target=self.assets[target],
                            operation=self._get_sql_operation(parsed),
                            transformation_logic=statement[:200],
                            source_file=file_path
                        )
                        self.transformations.append(trans)
    
    def _extract_sql_sources(self, parsed) -> Set[str]:
        """Extrai tabelas fonte de uma query SQL"""
        sources = set()
        from_seen = False
        
        for token in parsed.flatten():
            if token.ttype is Keyword and token.value.upper() in ['FROM', 'JOIN']:
                from_seen = True
            elif from_seen and not token.is_whitespace:
                if isinstance(token, Identifier):
                    sources.add(token.get_real_name())
                    from_seen = False
                elif token.ttype is None:
                    # Nome de tabela simples
                    table_name = token.value.strip('`"[]')
                    if table_name and not token.value.upper() in sqlparse.keywords.KEYWORDS:
                        sources.add(table_name)
                        from_seen = False
                        
        return sources
    
    def _extract_sql_targets(self, parsed) -> Set[str]:
        """Extrai tabelas destino de uma query SQL"""
        targets = set()
        
        # Procura por INSERT INTO, UPDATE, CREATE TABLE
        tokens = list(parsed.flatten())
        for i, token in enumerate(tokens):
            if token.ttype is Keyword:
                if token.value.upper() in ['INSERT', 'UPDATE', 'CREATE']:
                    # Procura próximo identificador
                    for j in range(i + 1, min(i + 10, len(tokens))):
                        if not tokens[j].is_whitespace and tokens[j].ttype is None:
                            table_name = tokens[j].value.strip('`"[]')
                            if table_name and not tokens[j].value.upper() in sqlparse.keywords.KEYWORDS:
                                targets.add(table_name)
                                break
                                
        return targets
    
    def _get_sql_operation(self, parsed) -> str:
        """Identifica o tipo de operação SQL"""
        for token in parsed.tokens:
            if token.ttype is DML:
                return token.value.upper()
            elif token.ttype is Keyword and token.value.upper() in ['CREATE', 'ALTER', 'DROP']:
                return token.value.upper()
        return 'UNKNOWN'
    
    def _parse_terraform(self, content: str, file_path: str):
        """Parser para arquivos Terraform"""
        # Pattern matching para recursos de dados
        resource_pattern = r'resource\s+"([^"]+)"\s+"([^"]+)"'
        data_source_pattern = r'data\s+"([^"]+)"\s+"([^"]+)"'
        
        # Extrai recursos
        resources = re.findall(resource_pattern, content)
        for resource_type, resource_name in resources:
            if any(data_type in resource_type for data_type in ['database', 'table', 'schema', 'dataset']):
                asset = DataAsset(
                    name=f"{resource_type}.{resource_name}",
                    type='terraform_resource',
                    source_file=file_path
                )
                self.assets[asset.name] = asset
        
        # Extrai data sources
        data_sources = re.findall(data_source_pattern, content)
        for source_type, source_name in data_sources:
            asset = DataAsset(
                name=f"data.{source_type}.{source_name}",
                type='terraform_data',
                source_file=file_path
            )
            self.assets[asset.name] = asset
        
        # Extrai referências entre recursos
        reference_pattern = r'\$\{([^}]+)\}'
        references = re.findall(reference_pattern, content)
        # TODO: Processar referências para criar transformações
    
    def _parse_databricks(self, content: str, file_path: str):
        """Parser para notebooks Databricks (Scala)"""
        # Padrões para Spark Scala
        table_pattern = r'spark\.table\("([^"]+)"\)'
        write_pattern = r'\.write\..*\.saveAsTable\("([^"]+)"\)'
        create_pattern = r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?(?:TABLE|VIEW)\s+(\w+)'
        
        # Extrai leituras de tabelas
        for match in re.finditer(table_pattern, content):
            table_name = match.group(1)
            asset = DataAsset(
                name=table_name,
                type='databricks_table',
                source_file=file_path,
                line_number=content[:match.start()].count('\n') + 1
            )
            self.assets[table_name] = asset
        
        # Extrai escritas de tabelas
        for match in re.finditer(write_pattern, content):
            table_name = match.group(1)
            asset = DataAsset(
                name=table_name,
                type='databricks_table',
                source_file=file_path,
                line_number=content[:match.start()].count('\n') + 1
            )
            self.assets[table_name] = asset
    
    def _parse_databricks_python(self, content: str, file_path: str):
        """Parser para notebooks Databricks Python"""
        # Detecta células magic commands
        if '# COMMAND ----------' in content or '%python' in content:
            # É um notebook Databricks
            cells = re.split(r'# COMMAND ----------|%\w+', content)
            for cell in cells:
                if cell.strip():
                    self._parse_python(cell, file_path)
        else:
            # Python regular
            self._parse_python(content, file_path)
    
    def _parse_databricks_sql(self, content: str, file_path: str):
        """Parser para SQL em Databricks"""
        # Além do SQL padrão, detecta comandos específicos do Databricks
        delta_pattern = r'DELTA\.`([^`]+)`'
        location_pattern = r"LOCATION\s+'([^']+)'"
        
        # Processa SQL padrão
        self._parse_sql(content, file_path)
        
        # Extrai tabelas Delta
        for match in re.finditer(delta_pattern, content):
            path = match.group(1)
            asset = DataAsset(
                name=path,
                type='delta_table',
                source_file=file_path
            )
            self.assets[path] = asset
    
    def _build_lineage_graph(self):
        """Constrói o grafo de linhagem"""
        # Adiciona nós
        for asset_name, asset in self.assets.items():
            self.graph.add_node(
                asset_name,
                type=asset.type,
                source_file=asset.source_file,
                schema=asset.schema
            )
        
        # Adiciona arestas
        for trans in self.transformations:
            self.graph.add_edge(
                trans.source.name,
                trans.target.name,
                operation=trans.operation,
                confidence=trans.confidence_score,
                source_file=trans.source_file
            )
    
    def _calculate_metrics(self) -> Dict:
        """Calcula métricas do pipeline"""
        metrics = {
            'total_assets': len(self.assets),
            'total_transformations': len(self.transformations),
            'asset_types': {},
            'operation_types': {},
            'complexity_metrics': {}
        }
        
        # Conta tipos de assets
        for asset in self.assets.values():
            metrics['asset_types'][asset.type] = metrics['asset_types'].get(asset.type, 0) + 1
        
        # Conta tipos de operações
        for trans in self.transformations:
            metrics['operation_types'][trans.operation] = metrics['operation_types'].get(trans.operation, 0) + 1
        
        # Métricas de complexidade do grafo
        if self.graph.number_of_nodes() > 0:
            metrics['complexity_metrics'] = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                'strongly_connected_components': nx.number_strongly_connected_components(self.graph),
                'weakly_connected_components': nx.number_weakly_connected_components(self.graph)
            }
            
            # Detecta ciclos
            try:
                metrics['complexity_metrics']['has_cycles'] = not nx.is_directed_acyclic_graph(self.graph)
                if metrics['complexity_metrics']['has_cycles']:
                    metrics['complexity_metrics']['cycles'] = list(nx.simple_cycles(self.graph))[:5]  # Primeiros 5 ciclos
            except:
                metrics['complexity_metrics']['has_cycles'] = False
        
        return metrics
    
    def get_upstream_dependencies(self, asset_name: str) -> List[str]:
        """Retorna todas as dependências upstream de um asset"""
        if asset_name not in self.graph:
            return []
        
        upstream = set()
        to_visit = [asset_name]
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            predecessors = list(self.graph.predecessors(current))
            upstream.update(predecessors)
            to_visit.extend(predecessors)
        
        return list(upstream)
    
    def get_downstream_impact(self, asset_name: str) -> List[str]:
        """Retorna todos os assets impactados downstream"""
        if asset_name not in self.graph:
            return []
        
        downstream = set()
        to_visit = [asset_name]
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            successors = list(self.graph.successors(current))
            downstream.update(successors)
            to_visit.extend(successors)
        
        return list(downstream)
    
    def analyze_change_impact(self, changed_assets: List[str]) -> Dict:
        """
        Analisa o impacto de mudanças em assets específicos
        """
        impact = {
            'directly_affected': changed_assets,
            'downstream_affected': set(),
            'upstream_dependencies': set(),
            'risk_level': 'LOW',
            'affected_pipelines': [],
            'recommendations': []
        }
        
        for asset in changed_assets:
            # Impacto downstream
            downstream = self.get_downstream_impact(asset)
            impact['downstream_affected'].update(downstream)
            
            # Dependências upstream
            upstream = self.get_upstream_dependencies(asset)
            impact['upstream_dependencies'].update(upstream)
        
        # Calcula nível de risco
        total_affected = len(impact['directly_affected']) + len(impact['downstream_affected'])
        if total_affected > 10:
            impact['risk_level'] = 'HIGH'
        elif total_affected > 5:
            impact['risk_level'] = 'MEDIUM'
        
        # Gera recomendações
        if impact['risk_level'] == 'HIGH':
            impact['recommendations'].append("⚠️ Alto impacto detectado. Considere testes extensivos antes do deploy.")
            impact['recommendations'].append("📊 Recomenda-se análise detalhada dos pipelines críticos afetados.")
        
        if len(impact['upstream_dependencies']) > 5:
            impact['recommendations'].append("🔄 Muitas dependências upstream. Verifique compatibilidade de schemas.")
        
        # Identifica pipelines críticos afetados
        critical_nodes = [n for n in impact['downstream_affected'] 
                         if self.graph.out_degree(n) > 3]  # Nós com muitas saídas
        if critical_nodes:
            impact['affected_pipelines'] = critical_nodes
            impact['recommendations'].append(f"🎯 Pipelines críticos afetados: {', '.join(critical_nodes[:5])}")
        
        return impact
    
    def compare_versions(self, old_pipeline: List[str], new_pipeline: List[str]) -> Dict:
        """
        Compara duas versões de pipeline e identifica mudanças
        """
        # Analisa ambos os pipelines
        old_analysis = self.analyze_pipeline(old_pipeline)
        old_assets = set(self.assets.keys())
        old_graph = self.graph.copy()
        
        new_analysis = self.analyze_pipeline(new_pipeline)
        new_assets = set(self.assets.keys())
        new_graph = self.graph.copy()
        
        comparison = {
            'added_assets': list(new_assets - old_assets),
            'removed_assets': list(old_assets - new_assets),
            'modified_assets': [],
            'added_connections': [],
            'removed_connections': [],
            'schema_changes': [],
            'risk_assessment': {}
        }
        
        # Identifica conexões mudadas
        old_edges = set(old_graph.edges())
        new_edges = set(new_graph.edges())
        
        comparison['added_connections'] = list(new_edges - old_edges)
        comparison['removed_connections'] = list(old_edges - new_edges)
        
        # Avalia risco das mudanças
        if comparison['removed_assets']:
            comparison['risk_assessment']['removed_assets_impact'] = {
                asset: self.get_downstream_impact(asset) 
                for asset in comparison['removed_assets'] 
                if asset in old_graph
            }
        
        return comparison
    
    def generate_documentation(self) -> str:
        """
        Gera documentação automática da linhagem
        """
        doc = []
        doc.append("# 📊 Documentação de Linhagem de Dados\n")
        doc.append(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resumo
        doc.append("## 📈 Resumo\n")
        doc.append(f"- **Total de Assets:** {len(self.assets)}\n")
        doc.append(f"- **Total de Transformações:** {len(self.transformations)}\n")
        doc.append(f"- **Tipos de Assets:** {', '.join(set(a.type for a in self.assets.values()))}\n\n")
        
        # Assets
        doc.append("## 🗂️ Assets de Dados\n")
        for asset_type in set(a.type for a in self.assets.values()):
            assets_of_type = [a for a in self.assets.values() if a.type == asset_type]
            doc.append(f"\n### {asset_type.upper()}\n")
            for asset in assets_of_type:
                doc.append(f"- **{asset.name}**\n")
                doc.append(f"  - Arquivo: `{asset.source_file}`\n")
                if asset.schema:
                    doc.append(f"  - Schema: {asset.schema}\n")
        
        # Fluxos principais
        doc.append("\n## 🔄 Fluxos de Dados Principais\n")
        
        # Identifica nós importantes
        important_nodes = []
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            if in_degree > 2 or out_degree > 2:
                important_nodes.append((node, in_degree, out_degree))
        
        important_nodes.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        for node, in_deg, out_deg in important_nodes[:10]:
            doc.append(f"\n**{node}**\n")
            doc.append(f"- Entradas: {in_deg} | Saídas: {out_deg}\n")
            
            # Lista algumas conexões
            predecessors = list(self.graph.predecessors(node))[:3]
            if predecessors:
                doc.append(f"- Recebe de: {', '.join(predecessors)}\n")
            
            successors = list(self.graph.successors(node))[:3]
            if successors:
                doc.append(f"- Envia para: {', '.join(successors)}\n")
        
        # Alertas
        doc.append("\n## ⚠️ Alertas e Observações\n")
        
        # Detecta ciclos
        try:
            if not nx.is_directed_acyclic_graph(self.graph):
                doc.append("- **ALERTA:** Ciclos detectados no pipeline!\n")
                cycles = list(nx.simple_cycles(self.graph))[:3]
                for cycle in cycles:
                    doc.append(f"  - Ciclo: {' -> '.join(cycle + [cycle[0]])}\n")
        except:
            pass
        
        # Nós isolados
        isolated = list(nx.isolates(self.graph))
        if isolated:
            doc.append(f"- **Nós isolados:** {', '.join(isolated[:5])}\n")
        
        return ''.join(doc)
