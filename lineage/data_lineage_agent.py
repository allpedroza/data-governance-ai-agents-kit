"""
Data Lineage Agent - Sistema de IA para Análise de Linhagem de Dados
Suporta: Python, SQL, Terraform, Databricks
Autor: Claude AI Assistant
"""

import ast
import re
import json
import os
from openai import OpenAI
import time
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
        self.llm_model = os.getenv("DATA_LINEAGE_LLM_MODEL", "gpt-5")
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        self.llm_client = None  # Inicializado sob demanda quando necessário
        self.llm_disabled = False  # Evita tentativas repetidas após falha de autenticação
        self._llm_warning_logged = False
        self.parsers = {
            '.py': self._parse_python,
            '.sql': self._parse_sql,
            '.tf': self._parse_terraform,
            '.json': self._parse_terraform,  # terraform json
            '.scala': self._parse_databricks,
            '.dag': self._parse_airflow,  # Airflow DAG files
        }
        self.airflow_dags = []  # Store Airflow DAGs separately
        
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

        # Identifica componentes críticos
        critical_components = self._identify_critical_components()

        # Gera insights automáticos
        print("🤖 Gerando insights automáticos...")
        insights = self._generate_graph_insights(metrics, critical_components)

        return {
            'assets': list(self.assets.values()),
            'transformations': self.transformations,
            'metrics': metrics,
            'critical_components': critical_components,
            'insights': insights,
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
        assets_before = len(self.assets)
        transformations_before = len(self.transformations)
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

        # Fallback via LLM quando não encontramos nada
        if len(self.assets) == assets_before and len(self.transformations) == transformations_before:
            self._parse_python_with_llm(content, file_path)
    
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
    
    def _parse_airflow(self, content: str, file_path: str):
        """Parser para Airflow DAGs"""
        from parsers.airflow_parser import AirflowParser
        
        parser = AirflowParser()
        dags, tasks = parser.parse_dag_file(file_path)
        
        # Store Airflow DAGs
        self.airflow_dags.extend(dags)
        
        # Convert Airflow tasks to our data model
        for task in tasks:
            # Create asset for each task
            asset = DataAsset(
                name=f"{task.dag_id}.{task.task_id}",
                type='airflow_task',
                source_file=file_path,
                line_number=task.line_number,
                metadata={
                    'operator': task.operator_type,
                    'dag_id': task.dag_id,
                    'task_id': task.task_id
                }
            )
            self.assets[asset.name] = asset
            
            # Create assets for data inputs/outputs
            for data_input in task.data_inputs:
                input_asset = DataAsset(
                    name=data_input,
                    type='data_source',
                    source_file=file_path,
                    metadata={'referenced_by': task.get_full_id()}
                )
                self.assets[data_input] = input_asset
                
                # Create transformation
                transform = Transformation(
                    source=input_asset,
                    target=asset,
                    operation='read',
                    transformation_logic=task.operator_type,
                    source_file=file_path,
                    line_number=task.line_number
                )
                self.transformations.append(transform)
            
            for data_output in task.data_outputs:
                output_asset = DataAsset(
                    name=data_output,
                    type='data_sink',
                    source_file=file_path,
                    metadata={'produced_by': task.get_full_id()}
                )
                self.assets[data_output] = output_asset
                
                # Create transformation
                transform = Transformation(
                    source=asset,
                    target=output_asset,
                    operation='write',
                    transformation_logic=task.operator_type,
                    source_file=file_path,
                    line_number=task.line_number
                )
                self.transformations.append(transform)
        
        # Process task dependencies
        for dag in dags:
            for task_id, task in dag.tasks.items():
                for downstream_id in task.downstream:
                    source_name = f"{dag.dag_id}.{task_id}"
                    target_name = f"{dag.dag_id}.{downstream_id}"
                    
                    if source_name in self.assets and target_name in self.assets:
                        transform = Transformation(
                            source=self.assets[source_name],
                            target=self.assets[target_name],
                            operation='task_dependency',
                            transformation_logic='airflow_dependency',
                            source_file=file_path
                        )
                        self.transformations.append(transform)
    
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

            # Se não for possível extrair de forma determinística, tenta LLM
            if not sources or not targets:
                llm_pairs = self._infer_lineage_with_llm(statement, file_path, language="SQL")
                for pair in llm_pairs:
                    source = pair.get('source')
                    target = pair.get('target')
                    if source and source not in self.assets:
                        self.assets[source] = DataAsset(
                            name=source,
                            type='table',
                            source_file=file_path
                        )
                    if target and target not in self.assets:
                        self.assets[target] = DataAsset(
                            name=target,
                            type='table',
                            source_file=file_path
                        )
                    if source and target and source != target:
                        trans = Transformation(
                            source=self.assets[source],
                            target=self.assets[target],
                            operation=pair.get('operation', 'INFERRED'),
                            transformation_logic=pair.get('logic', statement[:200]),
                            source_file=file_path,
                            confidence_score=pair.get('confidence', 0.5)
                        )
                        self.transformations.append(trans)
                if llm_pairs:
                    continue

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

    def _parse_python_with_llm(self, content: str, file_path: str):
        """Fallback baseado em LLM para blocos Python não estruturados"""
        llm_pairs = self._infer_lineage_with_llm(content, file_path, language="Python")
        for pair in llm_pairs:
            source = pair.get('source')
            target = pair.get('target')
            if source and source not in self.assets:
                self.assets[source] = DataAsset(
                    name=source,
                    type='table',
                    source_file=file_path
                )
            if target and target not in self.assets:
                self.assets[target] = DataAsset(
                    name=target,
                    type='table',
                    source_file=file_path
                )
            if source and target and source != target:
                trans = Transformation(
                    source=self.assets[source],
                    target=self.assets[target],
                    operation=pair.get('operation', 'INFERRED'),
                    transformation_logic=pair.get('logic', content[:200]),
                    source_file=file_path,
                    confidence_score=pair.get('confidence', 0.5)
                )
                self.transformations.append(trans)

    def _infer_lineage_with_llm(self, snippet: str, file_path: str, language: str) -> List[Dict[str, Any]]:
        """
        Usa um LLM (via API OpenAI compatível) para inferir pares fonte->destino.
        Retorna lista de dicionários com chaves: source, target, operation, logic, confidence.
        """
        if self.llm_disabled:
            return []

        if not self.llm_api_key:
            if not self._llm_warning_logged:
                print("ℹ️ Fallback LLM desabilitado: defina OPENAI_API_KEY para habilitar a extração contextual.")
                self._llm_warning_logged = True
            self.llm_disabled = True
            return []

        if self.llm_client is None:
            self.llm_client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)

        # Combina system prompt e user prompt em um único input
        full_prompt = (
            "Você é um assistente de engenharia de dados. Leia o trecho de código "
            "e extraia pares de linhagem (fonte -> destino) em formato JSON. "
            "Responda apenas JSON com lista de objetos contendo: source, target, operation, logic, confidence.\n\n"
            f"Arquivo: {file_path}\nLinguagem: {language}\nTrecho:\n{snippet[:4000]}"
        )

        for attempt in range(3):
            try:
                response = self.llm_client.responses.create(
                    model=self.llm_model,
                    input=full_prompt,
                    reasoning={"effort": "none"},
                    text={"verbosity": "low"}
                )

                content = response.output_text or ""
                if not content and getattr(response, "output", None):
                    text_chunks = []
                    for item in response.output:
                        for piece in item.get("content", []):
                            if piece.get("type") == "output_text":
                                text_chunks.append(piece.get("text", ""))
                    content = "".join(text_chunks)

                parsed = json.loads(content or "{}")
                if isinstance(parsed, dict) and 'lineage' in parsed:
                    return parsed.get('lineage', [])
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 401:
                    print(
                        "⚠️ Falha ao usar LLM para "
                        f"{file_path}: autenticação inválida (401). "
                        "Verifique OPENAI_API_KEY/OPENAI_API_URL. Fallback desativado."
                    )
                    self.llm_disabled = True
                    break
                if attempt == 2:
                    print(f"⚠️ Falha ao usar LLM para {file_path}: {e}")
                else:
                    time.sleep(2 ** attempt)

        return []
    
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

    def _identify_critical_components(self) -> Dict[str, Any]:
        """
        Identifica componentes críticos do grafo:
        - Pontos únicos de falha (articulation points)
        - Caminhos críticos (longest paths)
        - Assets com maior impacto (high fan-out/fan-in)
        - Subgrafos por domínio
        """
        critical_info = {
            'single_points_of_failure': [],
            'critical_paths': [],
            'high_impact_assets': [],
            'bottleneck_assets': [],
            'subgraphs': [],
            'isolated_components': []
        }

        if self.graph.number_of_nodes() == 0:
            return critical_info

        # Converte para grafo não direcionado para encontrar articulation points
        undirected = self.graph.to_undirected()

        # Pontos de articulação (single points of failure)
        articulation_points = list(nx.articulation_points(undirected))
        for node in articulation_points:
            downstream_count = len(self.get_downstream_impact(node))
            critical_info['single_points_of_failure'].append({
                'asset': node,
                'downstream_impact': downstream_count,
                'type': self.graph.nodes[node].get('type', 'unknown')
            })

        # Assets com alto impacto (muitas dependências downstream)
        for node in self.graph.nodes():
            out_degree = self.graph.out_degree(node)
            in_degree = self.graph.in_degree(node)

            # High fan-out (muitas saídas)
            if out_degree >= 3:
                critical_info['high_impact_assets'].append({
                    'asset': node,
                    'downstream_count': out_degree,
                    'upstream_count': in_degree,
                    'type': self.graph.nodes[node].get('type', 'unknown'),
                    'reason': 'high_fan_out'
                })

            # Bottlenecks (muitas entradas e saídas)
            if in_degree >= 2 and out_degree >= 2:
                critical_info['bottleneck_assets'].append({
                    'asset': node,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'type': self.graph.nodes[node].get('type', 'unknown')
                })

        # Caminhos críticos (longest paths em DAG)
        if nx.is_directed_acyclic_graph(self.graph):
            try:
                # Encontra o caminho mais longo
                longest_path = nx.dag_longest_path(self.graph)
                if len(longest_path) > 1:
                    critical_info['critical_paths'].append({
                        'path': longest_path,
                        'length': len(longest_path),
                        'description': f"Caminho crítico com {len(longest_path)} assets"
                    })
            except:
                pass

        # Componentes fracamente conectados (subgrafos)
        weak_components = list(nx.weakly_connected_components(self.graph))
        for i, component in enumerate(weak_components):
            if len(component) > 1:
                # Identifica tipo dominante no componente
                types = [self.graph.nodes[n].get('type', 'unknown') for n in component]
                dominant_type = max(set(types), key=types.count)

                critical_info['subgraphs'].append({
                    'id': i,
                    'size': len(component),
                    'assets': list(component)[:10],  # Primeiros 10
                    'dominant_type': dominant_type,
                    'description': f"Componente com {len(component)} assets ({dominant_type})"
                })

        # Nós isolados
        isolated = list(nx.isolates(self.graph))
        if isolated:
            critical_info['isolated_components'] = isolated[:20]  # Primeiros 20

        return critical_info

    def _generate_graph_insights(self, metrics: Dict, critical_info: Dict) -> Dict[str, Any]:
        """
        Gera insights e recomendações usando LLM baseado nas métricas e componentes críticos
        """
        insights = {
            'summary': '',
            'recommendations': [],
            'risk_assessment': '',
            'subgraph_summaries': []
        }

        # Se LLM não disponível, gera insights baseados em regras
        if self.llm_disabled or not self.llm_api_key:
            insights = self._generate_rule_based_insights(metrics, critical_info)
            return insights

        # Prepara contexto para o LLM
        context = self._prepare_insights_context(metrics, critical_info)

        # Gera insights com LLM
        llm_insights = self._call_llm_for_insights(context)

        if llm_insights:
            insights.update(llm_insights)
        else:
            # Fallback para regras se LLM falhar
            insights = self._generate_rule_based_insights(metrics, critical_info)

        return insights

    def _prepare_insights_context(self, metrics: Dict, critical_info: Dict) -> str:
        """Prepara contexto estruturado para enviar ao LLM"""
        context_parts = []

        # Resumo do grafo
        context_parts.append(f"# Análise de Linhagem de Dados\n")
        context_parts.append(f"Total de Assets: {metrics.get('total_assets', 0)}")
        context_parts.append(f"Total de Transformações: {metrics.get('total_transformations', 0)}")

        # Tipos de assets
        if metrics.get('asset_types'):
            context_parts.append(f"\nTipos de Assets:")
            for asset_type, count in metrics['asset_types'].items():
                context_parts.append(f"  - {asset_type}: {count}")

        # Métricas de complexidade
        complexity = metrics.get('complexity_metrics', {})
        if complexity:
            context_parts.append(f"\nComplexidade:")
            context_parts.append(f"  - Nós: {complexity.get('nodes', 0)}")
            context_parts.append(f"  - Arestas: {complexity.get('edges', 0)}")
            context_parts.append(f"  - Densidade: {complexity.get('density', 0):.3f}")
            context_parts.append(f"  - Grau médio: {complexity.get('avg_degree', 0):.2f}")
            context_parts.append(f"  - Ciclos: {'Sim' if complexity.get('has_cycles') else 'Não'}")

        # Pontos críticos
        spof = critical_info.get('single_points_of_failure', [])
        if spof:
            context_parts.append(f"\nPontos Únicos de Falha ({len(spof)}):")
            for point in spof[:5]:
                context_parts.append(f"  - {point['asset']}: impacta {point['downstream_impact']} assets downstream")

        # Assets de alto impacto
        high_impact = critical_info.get('high_impact_assets', [])
        if high_impact:
            context_parts.append(f"\nAssets de Alto Impacto ({len(high_impact)}):")
            for asset in high_impact[:5]:
                context_parts.append(f"  - {asset['asset']}: {asset['downstream_count']} dependências downstream")

        # Bottlenecks
        bottlenecks = critical_info.get('bottleneck_assets', [])
        if bottlenecks:
            context_parts.append(f"\nBottlenecks ({len(bottlenecks)}):")
            for bn in bottlenecks[:5]:
                context_parts.append(f"  - {bn['asset']}: {bn['in_degree']} → {bn['out_degree']}")

        # Subgrafos
        subgraphs = critical_info.get('subgraphs', [])
        if subgraphs:
            context_parts.append(f"\nComponentes/Subgrafos ({len(subgraphs)}):")
            for sg in subgraphs[:3]:
                context_parts.append(f"  - {sg['description']}")

        return "\n".join(context_parts)

    def _call_llm_for_insights(self, context: str) -> Optional[Dict]:
        """Chama LLM para gerar insights"""
        if self.llm_client is None:
            try:
                self.llm_client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
            except Exception as e:
                print(f"⚠️ Erro ao inicializar LLM client: {e}")
                self.llm_disabled = True
                return None

        # Combina system prompt e contexto em um único input
        full_prompt = (
            "Você é um especialista em engenharia de dados e análise de pipelines. "
            "Analise o grafo de linhagem de dados fornecido e gere:\n"
            "1. Um resumo executivo em português (2-3 parágrafos)\n"
            "2. Lista de recomendações específicas de melhorias\n"
            "3. Avaliação de risco (LOW/MEDIUM/HIGH) com justificativa\n"
            "4. Para cada componente/subgrafo, uma breve descrição\n\n"
            "Responda em formato JSON com as chaves: summary, recommendations (array), "
            "risk_assessment, subgraph_summaries (array).\n\n"
            f"{context}"
        )

        try:
            response = self.llm_client.responses.create(
                model=self.llm_model,
                input=full_prompt,
                reasoning={"effort": "low"},
                text={"verbosity": "medium"}
            )

            content = response.output_text or ""
            if not content and hasattr(response, "output"):
                text_chunks = []
                for item in response.output:
                    for piece in item.get("content", []):
                        if piece.get("type") == "output_text":
                            text_chunks.append(piece.get("text", ""))
                content = "".join(text_chunks)

            if content:
                return json.loads(content)
        except Exception as e:
            print(f"⚠️ Erro ao gerar insights com LLM: {e}")
            self.llm_disabled = True

        return None

    def _generate_rule_based_insights(self, metrics: Dict, critical_info: Dict) -> Dict:
        """Gera insights baseados em regras quando LLM não está disponível"""
        insights = {
            'summary': '',
            'recommendations': [],
            'risk_assessment': 'LOW',
            'subgraph_summaries': []
        }

        total_assets = metrics.get('total_assets', 0)
        total_trans = metrics.get('total_transformations', 0)
        complexity = metrics.get('complexity_metrics', {})

        # Gera resumo
        summary_parts = []
        summary_parts.append(f"O pipeline analisado contém {total_assets} assets de dados com {total_trans} transformações.")

        if complexity.get('has_cycles'):
            summary_parts.append("⚠️ ALERTA: O pipeline contém ciclos, o que pode indicar dependências circulares problemáticas.")

        spof = critical_info.get('single_points_of_failure', [])
        if spof:
            summary_parts.append(f"Foram identificados {len(spof)} pontos únicos de falha que requerem atenção especial.")

        insights['summary'] = " ".join(summary_parts)

        # Gera recomendações
        if spof:
            insights['recommendations'].append(
                f"🔴 Implementar redundância para {len(spof)} pontos únicos de falha identificados"
            )

        bottlenecks = critical_info.get('bottleneck_assets', [])
        if bottlenecks:
            insights['recommendations'].append(
                f"⚠️ Otimizar {len(bottlenecks)} bottlenecks identificados para melhorar performance"
            )

        if complexity.get('has_cycles'):
            insights['recommendations'].append(
                "🔄 Refatorar dependências circulares para simplificar o pipeline"
            )

        isolated = critical_info.get('isolated_components', [])
        if isolated:
            insights['recommendations'].append(
                f"📦 Revisar {len(isolated)} assets isolados que podem estar órfãos"
            )

        high_impact = critical_info.get('high_impact_assets', [])
        if len(high_impact) > 5:
            insights['recommendations'].append(
                "📊 Considerar modularização de assets com alto fan-out para facilitar manutenção"
            )

        # Avaliação de risco
        risk_score = 0
        if spof: risk_score += len(spof) * 2
        if complexity.get('has_cycles'): risk_score += 5
        if bottlenecks: risk_score += len(bottlenecks)

        if risk_score >= 10:
            insights['risk_assessment'] = 'HIGH - Pipeline com múltiplos pontos críticos'
        elif risk_score >= 5:
            insights['risk_assessment'] = 'MEDIUM - Alguns pontos de atenção identificados'
        else:
            insights['risk_assessment'] = 'LOW - Pipeline com estrutura saudável'

        # Resumos de subgrafos
        for subgraph in critical_info.get('subgraphs', []):
            insights['subgraph_summaries'].append({
                'id': subgraph['id'],
                'summary': f"Componente {subgraph['dominant_type']} com {subgraph['size']} assets interconectados"
            })

        return insights

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
