"""
Advanced Databricks Parser for Data Lineage Analysis
Analyzes Databricks notebooks, Delta tables, and workflows
"""

import re
import json
import ast
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from pathlib import Path


@dataclass
class DatabricksAsset:
    """Represents a Databricks data asset"""
    name: str
    type: str  # notebook, delta_table, view, job, pipeline, model
    catalog: str = "hive_metastore"
    schema: str = "default"
    location: Optional[str] = None
    format: str = "delta"
    properties: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    source_file: str = ""
    cell_number: int = 0
    
    def __hash__(self):
        return hash(f"{self.catalog}.{self.schema}.{self.name}")
    
    def get_full_name(self):
        if self.catalog and self.schema:
            return f"{self.catalog}.{self.schema}.{self.name}"
        elif self.schema:
            return f"{self.schema}.{self.name}"
        return self.name


@dataclass
class DatabricksTransformation:
    """Represents a transformation in Databricks"""
    source_tables: List[str]
    target_table: str
    operation: str  # merge, append, overwrite, create, insert
    transformation_sql: str
    notebook: str
    cell_number: int
    is_streaming: bool = False
    is_incremental: bool = False
    
    def __hash__(self):
        return hash(f"{'_'.join(self.source_tables)}_{self.target_table}_{self.notebook}")


class DatabricksParser:
    """
    Advanced parser for Databricks notebooks and configurations
    """
    
    # Common Databricks SQL patterns
    DATABRICKS_PATTERNS = {
        'create_table': r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s\(]+)',
        'create_view': r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMPORARY\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s\(]+)',
        'merge_into': r'MERGE\s+INTO\s+([^\s]+)\s+(?:AS\s+\w+\s+)?USING\s+([^\s]+)',
        'insert_into': r'INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([^\s\(]+)',
        'copy_into': r'COPY\s+INTO\s+([^\s]+)\s+FROM\s+([^\s\(]+)',
        'create_stream': r'CREATE\s+(?:OR\s+REPLACE\s+)?STREAM\s+([^\s]+)\s+ON\s+TABLE\s+([^\s]+)',
        'delta_table': r'delta\.`([^`]+)`|DELTA\.`([^`]+)`',
        'location_path': r"LOCATION\s+['\"]([^'\"]+)['\"]",
        'using_format': r'USING\s+(\w+)',
        'dbfs_path': r'dbfs:/[^\s\)]+',
        's3_path': r's3a?://[^\s\)]+',
        'azure_path': r'wasbs?://[^\s\)]+|abfss?://[^\s\)]+',
        'catalog_table': r'(\w+)\.(\w+)\.(\w+)',  # catalog.schema.table
        'schema_table': r'(\w+)\.(\w+)',  # schema.table
    }
    
    # Delta Lake specific operations
    DELTA_OPERATIONS = {
        'optimize': r'OPTIMIZE\s+([^\s]+)',
        'vacuum': r'VACUUM\s+([^\s]+)',
        'zorder': r'ZORDER\s+BY\s+\(([^)]+)\)',
        'history': r'DESCRIBE\s+HISTORY\s+([^\s]+)',
        'detail': r'DESCRIBE\s+DETAIL\s+([^\s]+)',
        'clone': r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+([^\s]+)\s+(?:SHALLOW|DEEP)?\s*CLONE\s+([^\s]+)',
        'restore': r'RESTORE\s+TABLE\s+([^\s]+)\s+TO\s+VERSION\s+AS\s+OF\s+(\d+)',
        'time_travel': r'SELECT\s+.*\s+FROM\s+([^\s]+)\s+VERSION\s+AS\s+OF\s+(\d+)',
    }
    
    def __init__(self):
        self.assets: Dict[str, DatabricksAsset] = {}
        self.transformations: List[DatabricksTransformation] = []
        self.notebooks: Dict[str, Dict] = {}
        self.workflows: Dict[str, Dict] = {}
        self.delta_tables: Set[str] = set()
        self.streaming_jobs: List[Dict] = []
        
    def parse_notebook(self, content: str, notebook_path: str) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
        """
        Parse Databricks notebook (Python or SQL)
        Supports both .py with # COMMAND ---------- and .sql formats
        """
        assets = []
        transformations = []
        
        # Detect notebook format
        if '# COMMAND ----------' in content or '-- COMMAND ----------' in content:
            cells = self._split_notebook_cells(content)
        else:
            # Treat entire content as single cell
            cells = [{'content': content, 'language': self._detect_language(content), 'number': 1}]
        
        for cell in cells:
            cell_content = cell['content']
            cell_language = cell['language']
            cell_number = cell['number']
            
            if cell_language == 'python':
                cell_assets, cell_transforms = self._parse_python_cell(
                    cell_content, notebook_path, cell_number
                )
            elif cell_language == 'sql':
                cell_assets, cell_transforms = self._parse_sql_cell(
                    cell_content, notebook_path, cell_number
                )
            elif cell_language == 'scala':
                cell_assets, cell_transforms = self._parse_scala_cell(
                    cell_content, notebook_path, cell_number
                )
            else:
                continue
            
            assets.extend(cell_assets)
            transformations.extend(cell_transforms)
        
        # Store notebook metadata
        self.notebooks[notebook_path] = {
            'cells': len(cells),
            'assets': [a.name for a in assets],
            'transformations': len(transformations),
            'languages': list(set(c['language'] for c in cells))
        }
        
        return assets, transformations
    
    def _split_notebook_cells(self, content: str) -> List[Dict]:
        """Split Databricks notebook into cells"""
        cells = []
        
        # Split by command separator
        cell_separator = re.compile(r'^[#-]+\s*COMMAND\s+[-]+\s*$', re.MULTILINE)
        cell_contents = cell_separator.split(content)
        
        for i, cell_content in enumerate(cell_contents):
            if not cell_content.strip():
                continue
            
            # Detect language from magic commands
            language = 'python'  # default
            if cell_content.strip().startswith('%sql'):
                language = 'sql'
                cell_content = cell_content.replace('%sql', '', 1)
            elif cell_content.strip().startswith('%scala'):
                language = 'scala'
                cell_content = cell_content.replace('%scala', '', 1)
            elif cell_content.strip().startswith('%python'):
                language = 'python'
                cell_content = cell_content.replace('%python', '', 1)
            elif cell_content.strip().startswith('%r'):
                language = 'r'
                cell_content = cell_content.replace('%r', '', 1)
            elif cell_content.strip().startswith('%md'):
                language = 'markdown'
                continue  # Skip markdown cells
            
            cells.append({
                'content': cell_content.strip(),
                'language': language,
                'number': i
            })
        
        return cells
    
    def _detect_language(self, content: str) -> str:
        """Detect programming language from content"""
        # Simple heuristic
        sql_keywords = ['SELECT', 'CREATE', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'WITH']
        python_keywords = ['import', 'def', 'class', 'from', 'spark.', 'dbutils.']
        scala_keywords = ['val', 'var', 'def', 'class', 'import', 'object']
        
        content_upper = content.upper()
        
        sql_score = sum(1 for kw in sql_keywords if kw in content_upper)
        python_score = sum(1 for kw in python_keywords if kw in content)
        scala_score = sum(1 for kw in scala_keywords if kw in content)
        
        if sql_score > python_score and sql_score > scala_score:
            return 'sql'
        elif scala_score > python_score:
            return 'scala'
        else:
            return 'python'
    
    def _parse_python_cell(self, content: str, notebook_path: str, cell_number: int) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
        """Parse Python cell in Databricks notebook"""
        assets = []
        transformations = []
        
        # Patterns for PySpark operations
        patterns = {
            'read_table': r'spark\.(?:read|table|sql)\(["\']([^"\']+)["\']',
            'write_table': r'\.write(?:Stream)?\..*\.(?:save|saveAsTable|insertInto)\(["\']([^"\']+)["\']',
            'create_temp_view': r'\.createOrReplace(?:Global)?TempView\(["\']([^"\']+)["\']',
            'delta_table': r'DeltaTable\.forPath\(spark,\s*["\']([^"\']+)["\']|DeltaTable\.forName\(spark,\s*["\']([^"\']+)["\']',
            'dbutils_cp': r'dbutils\.fs\.cp\(["\']([^"\']+)["\'],\s*["\']([^"\']+)["\']',
            'sql_query': r'spark\.sql\(f?["\']([^"\']+)["\']',
            'read_format': r'spark\.read\.format\(["\'](\w+)["\']',
            'streaming': r'spark\.readStream|writeStream',
            'autoloader': r'\.format\(["\']cloudFiles["\']',
            'ml_model': r'mlflow\..*\.log_model|MLmodel|model\.save\(["\']([^"\']+)["\']'
        }
        
        # Extract tables and operations
        for pattern_name, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                if pattern_name == 'read_table':
                    table_name = match.group(1)
                    if not table_name.startswith('('):  # Avoid SQL queries
                        asset = DatabricksAsset(
                            name=table_name,
                            type='table',
                            source_file=notebook_path,
                            cell_number=cell_number
                        )
                        assets.append(asset)
                        
                elif pattern_name == 'write_table':
                    table_name = match.group(1)
                    asset = DatabricksAsset(
                        name=table_name,
                        type='delta_table',
                        source_file=notebook_path,
                        cell_number=cell_number
                    )
                    assets.append(asset)
                    
                elif pattern_name == 'delta_table':
                    path_or_name = match.group(1) or match.group(2)
                    if path_or_name:
                        asset = DatabricksAsset(
                            name=path_or_name,
                            type='delta_table',
                            location=path_or_name if '/' in path_or_name else None,
                            source_file=notebook_path,
                            cell_number=cell_number
                        )
                        assets.append(asset)
                        self.delta_tables.add(path_or_name)
                        
                elif pattern_name == 'streaming':
                    # Mark as streaming job
                    self.streaming_jobs.append({
                        'notebook': notebook_path,
                        'cell': cell_number,
                        'type': 'structured_streaming'
                    })
                    
                elif pattern_name == 'ml_model':
                    model_path = match.group(1) if match.lastindex else 'ml_model'
                    asset = DatabricksAsset(
                        name=model_path,
                        type='ml_model',
                        source_file=notebook_path,
                        cell_number=cell_number
                    )
                    assets.append(asset)
        
        # Extract SQL queries from spark.sql()
        sql_pattern = r'spark\.sql\(f?["\']([^"\']+)["\']'
        for match in re.finditer(sql_pattern, content, re.DOTALL):
            sql_query = match.group(1)
            sql_assets, sql_transforms = self._parse_sql_statement(
                sql_query, notebook_path, cell_number
            )
            assets.extend(sql_assets)
            transformations.extend(sql_transforms)
        
        # Extract DataFrame transformations
        df_pattern = r'(\w+)\s*=\s*(\w+)\.(?:select|filter|join|groupBy|agg|withColumn)'
        for match in re.finditer(df_pattern, content):
            target_df = match.group(1)
            source_df = match.group(2)
            # This is a simplified extraction - in practice, you'd track DataFrame lineage more carefully
        
        return assets, transformations
    
    def _parse_sql_cell(self, content: str, notebook_path: str, cell_number: int) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
        """Parse SQL cell in Databricks notebook"""
        assets = []
        transformations = []
        
        # Parse each SQL statement
        statements = sqlparse.split(content)
        
        for statement in statements:
            if not statement.strip():
                continue
            
            stmt_assets, stmt_transforms = self._parse_sql_statement(
                statement, notebook_path, cell_number
            )
            assets.extend(stmt_assets)
            transformations.extend(stmt_transforms)
        
        return assets, transformations
    
    def _parse_sql_statement(self, sql: str, source_file: str, cell_number: int) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
        """Parse a single SQL statement"""
        assets = []
        transformations = []
        
        sql_upper = sql.upper()
        
        # CREATE TABLE
        create_table_match = re.search(self.DATABRICKS_PATTERNS['create_table'], sql, re.IGNORECASE)
        if create_table_match:
            table_name = create_table_match.group(1)
            
            # Extract location if specified
            location_match = re.search(self.DATABRICKS_PATTERNS['location_path'], sql, re.IGNORECASE)
            location = location_match.group(1) if location_match else None
            
            # Extract format
            format_match = re.search(self.DATABRICKS_PATTERNS['using_format'], sql, re.IGNORECASE)
            format_type = format_match.group(1).lower() if format_match else 'delta'
            
            asset = DatabricksAsset(
                name=table_name,
                type='delta_table' if format_type == 'delta' else 'table',
                location=location,
                format=format_type,
                source_file=source_file,
                cell_number=cell_number
            )
            
            # Parse catalog.schema.table format
            self._parse_table_identifier(table_name, asset)
            
            assets.append(asset)
            
            # Check if it's CREATE TABLE AS SELECT
            if ' AS SELECT ' in sql_upper or ' AS (' in sql_upper:
                # Extract source tables
                source_tables = self._extract_source_tables(sql)
                if source_tables:
                    transformation = DatabricksTransformation(
                        source_tables=source_tables,
                        target_table=asset.get_full_name(),
                        operation='create_as_select',
                        transformation_sql=sql[:500],  # Truncate for storage
                        notebook=source_file,
                        cell_number=cell_number
                    )
                    transformations.append(transformation)
        
        # MERGE INTO
        merge_match = re.search(self.DATABRICKS_PATTERNS['merge_into'], sql, re.IGNORECASE)
        if merge_match:
            target_table = merge_match.group(1)
            source_table = merge_match.group(2)
            
            transformation = DatabricksTransformation(
                source_tables=[source_table],
                target_table=target_table,
                operation='merge',
                transformation_sql=sql[:500],
                notebook=source_file,
                cell_number=cell_number,
                is_incremental=True
            )
            transformations.append(transformation)
            
            # Add tables as assets
            for table in [target_table, source_table]:
                asset = DatabricksAsset(
                    name=table,
                    type='delta_table',
                    source_file=source_file,
                    cell_number=cell_number
                )
                self._parse_table_identifier(table, asset)
                assets.append(asset)
        
        # INSERT INTO / OVERWRITE
        insert_match = re.search(self.DATABRICKS_PATTERNS['insert_into'], sql, re.IGNORECASE)
        if insert_match:
            target_table = insert_match.group(1)
            source_tables = self._extract_source_tables(sql)
            
            operation = 'insert_overwrite' if 'OVERWRITE' in sql_upper else 'insert_into'
            
            transformation = DatabricksTransformation(
                source_tables=source_tables,
                target_table=target_table,
                operation=operation,
                transformation_sql=sql[:500],
                notebook=source_file,
                cell_number=cell_number,
                is_incremental='INTO' in sql_upper and 'OVERWRITE' not in sql_upper
            )
            transformations.append(transformation)
        
        # COPY INTO
        copy_match = re.search(self.DATABRICKS_PATTERNS['copy_into'], sql, re.IGNORECASE)
        if copy_match:
            target_table = copy_match.group(1)
            source_location = copy_match.group(2)
            
            transformation = DatabricksTransformation(
                source_tables=[source_location],
                target_table=target_table,
                operation='copy_into',
                transformation_sql=sql[:500],
                notebook=source_file,
                cell_number=cell_number
            )
            transformations.append(transformation)
            
            asset = DatabricksAsset(
                name=target_table,
                type='delta_table',
                source_file=source_file,
                cell_number=cell_number
            )
            assets.append(asset)
        
        # Delta operations (OPTIMIZE, VACUUM, etc.)
        for op_name, op_pattern in self.DELTA_OPERATIONS.items():
            op_match = re.search(op_pattern, sql, re.IGNORECASE)
            if op_match:
                table_name = op_match.group(1)
                
                if op_name == 'clone':
                    target_table = op_match.group(1)
                    source_table = op_match.group(2)
                    
                    transformation = DatabricksTransformation(
                        source_tables=[source_table],
                        target_table=target_table,
                        operation=f'delta_{op_name}',
                        transformation_sql=sql[:500],
                        notebook=source_file,
                        cell_number=cell_number
                    )
                    transformations.append(transformation)
                    
                    # Add both tables as assets
                    for table in [target_table, source_table]:
                        asset = DatabricksAsset(
                            name=table,
                            type='delta_table',
                            source_file=source_file,
                            cell_number=cell_number
                        )
                        assets.append(asset)
                else:
                    # Record Delta maintenance operation
                    asset = DatabricksAsset(
                        name=table_name,
                        type='delta_table',
                        source_file=source_file,
                        cell_number=cell_number,
                        properties={'delta_operation': op_name}
                    )
                    assets.append(asset)
                    self.delta_tables.add(table_name)
        
        return assets, transformations
    
    def _parse_scala_cell(self, content: str, notebook_path: str, cell_number: int) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
        """Parse Scala cell in Databricks notebook"""
        assets = []
        transformations = []
        
        # Patterns for Scala/Spark operations
        patterns = {
            'read_table': r'spark\.(?:read|table)\("([^"]+)"\)',
            'write_table': r'\.write(?:Stream)?\..*\.(?:save|saveAsTable)\("([^"]+)"\)',
            'delta_table': r'DeltaTable\.forPath\(spark,\s*"([^"]+)"\)|DeltaTable\.forName\(spark,\s*"([^"]+)"\)',
            'sql_query': r'spark\.sql\("([^"]+)"\)'
        }
        
        for pattern_name, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                if pattern_name in ['read_table', 'write_table']:
                    table_name = match.group(1)
                    asset = DatabricksAsset(
                        name=table_name,
                        type='table',
                        source_file=notebook_path,
                        cell_number=cell_number
                    )
                    assets.append(asset)
                    
                elif pattern_name == 'sql_query':
                    sql_query = match.group(1)
                    sql_assets, sql_transforms = self._parse_sql_statement(
                        sql_query, notebook_path, cell_number
                    )
                    assets.extend(sql_assets)
                    transformations.extend(sql_transforms)
        
        return assets, transformations
    
    def _parse_table_identifier(self, table_identifier: str, asset: DatabricksAsset):
        """Parse catalog.schema.table format"""
        parts = table_identifier.split('.')
        
        if len(parts) == 3:
            asset.catalog = parts[0]
            asset.schema = parts[1]
            asset.name = parts[2]
        elif len(parts) == 2:
            asset.schema = parts[0]
            asset.name = parts[1]
        else:
            asset.name = table_identifier
    
    def _extract_source_tables(self, sql: str) -> List[str]:
        """Extract source tables from SQL query"""
        tables = []
        
        # Parse SQL to find FROM and JOIN clauses
        parsed = sqlparse.parse(sql)[0] if sqlparse.parse(sql) else None
        if not parsed:
            # Fallback to regex
            from_pattern = r'FROM\s+([^\s,]+)'
            join_pattern = r'JOIN\s+([^\s]+)'
            
            for pattern in [from_pattern, join_pattern]:
                for match in re.finditer(pattern, sql, re.IGNORECASE):
                    table_name = match.group(1)
                    # Clean table name
                    table_name = table_name.strip('`').strip('"').strip()
                    if table_name and not table_name.startswith('('):
                        tables.append(table_name)
        
        return list(set(tables))  # Remove duplicates
    
    def parse_workflow(self, workflow_config: Dict, workflow_name: str) -> Dict:
        """
        Parse Databricks workflow/job configuration
        """
        workflow_info = {
            'name': workflow_name,
            'tasks': [],
            'dependencies': [],
            'schedule': None,
            'alerts': []
        }
        
        if 'tasks' in workflow_config:
            for task in workflow_config['tasks']:
                task_info = {
                    'name': task.get('task_key', ''),
                    'type': self._get_task_type(task),
                    'dependencies': task.get('depends_on', []),
                    'notebook': None,
                    'sql_query': None,
                    'python_file': None
                }
                
                # Extract task details based on type
                if 'notebook_task' in task:
                    task_info['notebook'] = task['notebook_task'].get('notebook_path', '')
                    task_info['parameters'] = task['notebook_task'].get('base_parameters', {})
                    
                elif 'spark_python_task' in task:
                    task_info['python_file'] = task['spark_python_task'].get('python_file', '')
                    
                elif 'sql_task' in task:
                    task_info['sql_query'] = task['sql_task'].get('query', {}).get('query_id', '')
                    
                elif 'dbt_task' in task:
                    task_info['dbt_commands'] = task['dbt_task'].get('commands', [])
                
                workflow_info['tasks'].append(task_info)
                
                # Build task dependencies
                for dep in task_info['dependencies']:
                    workflow_info['dependencies'].append({
                        'source': dep.get('task_key', ''),
                        'target': task_info['name']
                    })
        
        # Extract schedule
        if 'schedule' in workflow_config:
            workflow_info['schedule'] = {
                'quartz_cron': workflow_config['schedule'].get('quartz_cron_expression', ''),
                'timezone': workflow_config['schedule'].get('timezone_id', 'UTC'),
                'pause_status': workflow_config['schedule'].get('pause_status', 'UNPAUSED')
            }
        
        # Extract alerts
        if 'email_notifications' in workflow_config:
            workflow_info['alerts'] = workflow_config['email_notifications']
        
        self.workflows[workflow_name] = workflow_info
        
        return workflow_info
    
    def _get_task_type(self, task: Dict) -> str:
        """Determine task type from configuration"""
        task_types = {
            'notebook_task': 'notebook',
            'spark_python_task': 'python',
            'python_wheel_task': 'wheel',
            'spark_jar_task': 'jar',
            'spark_submit_task': 'spark_submit',
            'sql_task': 'sql',
            'dbt_task': 'dbt',
            'pipeline_task': 'delta_live_table'
        }
        
        for key, type_name in task_types.items():
            if key in task:
                return type_name
        
        return 'unknown'
    
    def get_lineage_graph(self) -> Dict:
        """Generate lineage graph from parsed assets"""
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': {}
        }
        
        # Create nodes for assets
        for asset_id, asset in self.assets.items():
            node = {
                'id': asset.get_full_name(),
                'label': asset.name,
                'type': asset.type,
                'catalog': asset.catalog,
                'schema': asset.schema,
                'location': asset.location,
                'format': asset.format,
                'properties': asset.properties
            }
            graph['nodes'].append(node)
        
        # Create edges from transformations
        for transformation in self.transformations:
            for source in transformation.source_tables:
                graph['edges'].append({
                    'source': source,
                    'target': transformation.target_table,
                    'operation': transformation.operation,
                    'is_streaming': transformation.is_streaming,
                    'is_incremental': transformation.is_incremental
                })
        
        # Group by catalog/schema
        for node in graph['nodes']:
            cluster_key = f"{node['catalog']}.{node['schema']}"
            if cluster_key not in graph['clusters']:
                graph['clusters'][cluster_key] = {
                    'label': cluster_key,
                    'nodes': []
                }
            graph['clusters'][cluster_key]['nodes'].append(node['id'])
        
        return graph
    
    def analyze_delta_operations(self) -> Dict:
        """Analyze Delta Lake specific operations and optimizations"""
        delta_analysis = {
            'total_delta_tables': len(self.delta_tables),
            'delta_tables': list(self.delta_tables),
            'streaming_jobs': self.streaming_jobs,
            'incremental_loads': [],
            'merge_operations': [],
            'optimization_opportunities': []
        }
        
        # Identify incremental loads
        for transformation in self.transformations:
            if transformation.is_incremental:
                delta_analysis['incremental_loads'].append({
                    'target': transformation.target_table,
                    'sources': transformation.source_tables,
                    'operation': transformation.operation
                })
            
            if transformation.operation == 'merge':
                delta_analysis['merge_operations'].append({
                    'target': transformation.target_table,
                    'source': transformation.source_tables[0] if transformation.source_tables else None
                })
        
        # Suggest optimizations
        for table in self.delta_tables:
            # Tables with many merges might benefit from Z-ordering
            merge_count = sum(1 for m in delta_analysis['merge_operations'] if m['target'] == table)
            if merge_count > 5:
                delta_analysis['optimization_opportunities'].append({
                    'table': table,
                    'suggestion': 'Consider Z-ordering on merge keys',
                    'reason': f'High merge frequency ({merge_count} operations)'
                })
        
        return delta_analysis


# Utility functions
def parse_databricks_notebook(file_path: str) -> Tuple[List[DatabricksAsset], List[DatabricksTransformation]]:
    """Parse a single Databricks notebook"""
    parser = DatabricksParser()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return parser.parse_notebook(content, file_path)


def parse_databricks_workspace(directory: str) -> Dict:
    """Parse all Databricks notebooks in a directory"""
    parser = DatabricksParser()
    all_assets = []
    all_transformations = []
    
    from pathlib import Path
    
    # Look for notebook files
    notebook_extensions = ['*.py', '*.sql', '*.scala', '*.ipynb']
    notebook_files = []
    
    for ext in notebook_extensions:
        notebook_files.extend(Path(directory).glob(f'**/{ext}'))
    
    for notebook_file in notebook_files:
        with open(notebook_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if not a Databricks notebook
        if '# Databricks notebook source' in content or '-- Databricks notebook source' in content or str(notebook_file).endswith('.sql'):
            assets, transformations = parser.parse_notebook(content, str(notebook_file))
            all_assets.extend(assets)
            all_transformations.extend(transformations)
    
    # Look for workflow configurations
    workflow_files = list(Path(directory).glob('**/*.json'))
    for workflow_file in workflow_files:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            try:
                workflow_config = json.load(f)
                if 'tasks' in workflow_config:  # Likely a workflow config
                    parser.parse_workflow(workflow_config, workflow_file.stem)
            except json.JSONDecodeError:
                continue
    
    return {
        'assets': all_assets,
        'transformations': all_transformations,
        'graph': parser.get_lineage_graph(),
        'delta_analysis': parser.analyze_delta_operations(),
        'workflows': parser.workflows,
        'notebooks': parser.notebooks
    }
