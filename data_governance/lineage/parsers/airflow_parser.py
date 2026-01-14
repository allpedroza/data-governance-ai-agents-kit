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
Advanced Apache Airflow Parser for Data Lineage Analysis
Analyzes Airflow DAGs, tasks, dependencies, and data flows
"""

import re
import ast
import json
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import yaml


@dataclass
class AirflowTask:
    """Represents an Airflow task"""
    task_id: str
    operator_type: str
    dag_id: str
    dependencies: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    trigger_rule: str = "all_success"
    retries: int = 0
    retry_delay: Optional[timedelta] = None
    pool: Optional[str] = None
    queue: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    data_inputs: List[str] = field(default_factory=list)
    data_outputs: List[str] = field(default_factory=list)
    sql_query: Optional[str] = None
    python_callable: Optional[str] = None
    bash_command: Optional[str] = None
    source_file: str = ""
    line_number: int = 0
    
    def __hash__(self):
        return hash(f"{self.dag_id}.{self.task_id}")
    
    def get_full_id(self):
        return f"{self.dag_id}.{self.task_id}"


@dataclass
class AirflowDAG:
    """Represents an Airflow DAG"""
    dag_id: str
    description: Optional[str] = None
    schedule_interval: Optional[Union[str, timedelta]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    catchup: bool = True
    tags: List[str] = field(default_factory=list)
    default_args: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, AirflowTask] = field(default_factory=dict)
    task_groups: Dict[str, List[str]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    doc_md: Optional[str] = None
    source_file: str = ""
    
    def __hash__(self):
        return hash(self.dag_id)


@dataclass
class AirflowDataAsset:
    """Represents a data asset referenced in Airflow"""
    name: str
    type: str  # table, file, api, database, bucket, etc.
    location: Optional[str] = None
    connection_id: Optional[str] = None
    format: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    task_references: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(f"{self.name}_{self.type}")


class AirflowParser:
    """
    Advanced parser for Apache Airflow DAGs
    Extracts tasks, dependencies, and data lineage
    """
    
    # Common Airflow operators and their data patterns
    OPERATOR_PATTERNS = {
        # SQL Operators
        'PostgresOperator': {
            'type': 'sql',
            'params': ['sql', 'postgres_conn_id', 'database'],
            'data_pattern': 'database'
        },
        'MySqlOperator': {
            'type': 'sql',
            'params': ['sql', 'mysql_conn_id', 'database'],
            'data_pattern': 'database'
        },
        'SqliteOperator': {
            'type': 'sql',
            'params': ['sql', 'sqlite_conn_id'],
            'data_pattern': 'database'
        },
        'MsSqlOperator': {
            'type': 'sql',
            'params': ['sql', 'mssql_conn_id', 'database'],
            'data_pattern': 'database'
        },
        'OracleOperator': {
            'type': 'sql',
            'params': ['sql', 'oracle_conn_id'],
            'data_pattern': 'database'
        },
        'JdbcOperator': {
            'type': 'sql',
            'params': ['sql', 'jdbc_conn_id'],
            'data_pattern': 'database'
        },
        
        # BigQuery Operators
        'BigQueryOperator': {
            'type': 'bigquery',
            'params': ['sql', 'destination_dataset_table', 'use_legacy_sql'],
            'data_pattern': 'bigquery'
        },
        'BigQueryCreateEmptyTableOperator': {
            'type': 'bigquery',
            'params': ['dataset_id', 'table_id', 'schema_fields'],
            'data_pattern': 'bigquery'
        },
        'BigQueryInsertJobOperator': {
            'type': 'bigquery',
            'params': ['configuration', 'job_id'],
            'data_pattern': 'bigquery'
        },
        
        # AWS Operators
        'S3FileTransformOperator': {
            'type': 's3',
            'params': ['source_s3_key', 'dest_s3_key', 'transform_script'],
            'data_pattern': 's3'
        },
        'S3ToRedshiftOperator': {
            'type': 's3_to_redshift',
            'params': ['s3_bucket', 's3_key', 'schema', 'table'],
            'data_pattern': 's3_to_database'
        },
        'AthenaOperator': {
            'type': 'athena',
            'params': ['query', 'database', 'output_location'],
            'data_pattern': 'athena'
        },
        'EmrAddStepsOperator': {
            'type': 'emr',
            'params': ['job_flow_id', 'steps'],
            'data_pattern': 'emr'
        },
        'GlueJobOperator': {
            'type': 'glue',
            'params': ['job_name', 'script_location'],
            'data_pattern': 'glue'
        },
        
        # GCP Operators
        'GCSToBigQueryOperator': {
            'type': 'gcs_to_bigquery',
            'params': ['bucket', 'source_objects', 'destination_project_dataset_table'],
            'data_pattern': 'gcs_to_bigquery'
        },
        'DataflowTemplatedJobStartOperator': {
            'type': 'dataflow',
            'params': ['template', 'parameters', 'dataflow_default_options'],
            'data_pattern': 'dataflow'
        },
        'DataprocSubmitJobOperator': {
            'type': 'dataproc',
            'params': ['job', 'project_id', 'region'],
            'data_pattern': 'dataproc'
        },
        
        # Azure Operators
        'AzureDataLakeStorageDeleteOperator': {
            'type': 'adls',
            'params': ['path', 'azure_data_lake_conn_id'],
            'data_pattern': 'adls'
        },
        'AzureSynapseRunPipelineOperator': {
            'type': 'synapse',
            'params': ['pipeline_name', 'azure_synapse_conn_id'],
            'data_pattern': 'synapse'
        },
        
        # Databricks Operators
        'DatabricksSubmitRunOperator': {
            'type': 'databricks',
            'params': ['json', 'spark_jar_task', 'notebook_task', 'spark_python_task'],
            'data_pattern': 'databricks'
        },
        'DatabricksRunNowOperator': {
            'type': 'databricks',
            'params': ['job_id', 'notebook_params', 'python_params'],
            'data_pattern': 'databricks'
        },
        
        # Transfer Operators
        'S3ToGCSOperator': {
            'type': 'transfer',
            'params': ['bucket', 'prefix', 'dest_gcs_bucket'],
            'data_pattern': 's3_to_gcs'
        },
        'GCSToS3Operator': {
            'type': 'transfer',
            'params': ['bucket', 'prefix', 'dest_s3_key'],
            'data_pattern': 'gcs_to_s3'
        },
        
        # Other Common Operators
        'BashOperator': {
            'type': 'bash',
            'params': ['bash_command', 'env'],
            'data_pattern': 'bash'
        },
        'PythonOperator': {
            'type': 'python',
            'params': ['python_callable', 'op_kwargs', 'op_args'],
            'data_pattern': 'python'
        },
        'EmailOperator': {
            'type': 'email',
            'params': ['to', 'subject', 'html_content'],
            'data_pattern': 'notification'
        },
        'SlackWebhookOperator': {
            'type': 'slack',
            'params': ['message', 'channel', 'webhook_token'],
            'data_pattern': 'notification'
        },
        'HttpOperator': {
            'type': 'http',
            'params': ['endpoint', 'method', 'data', 'headers'],
            'data_pattern': 'api'
        },
        'SimpleHttpOperator': {
            'type': 'http',
            'params': ['endpoint', 'method', 'data'],
            'data_pattern': 'api'
        },
        
        # Spark Operators
        'SparkSubmitOperator': {
            'type': 'spark',
            'params': ['application', 'conf', 'files', 'py_files'],
            'data_pattern': 'spark'
        },
        'SparkJDBCOperator': {
            'type': 'spark',
            'params': ['spark_conf', 'cmd', 'jdbc_conn_id'],
            'data_pattern': 'spark'
        },
        'SparkSqlOperator': {
            'type': 'spark',
            'params': ['sql', 'conf'],
            'data_pattern': 'spark'
        },
        
        # Kubernetes Operators
        'KubernetesPodOperator': {
            'type': 'kubernetes',
            'params': ['image', 'cmds', 'arguments', 'env_vars'],
            'data_pattern': 'kubernetes'
        },
        
        # Docker Operators
        'DockerOperator': {
            'type': 'docker',
            'params': ['image', 'command', 'environment'],
            'data_pattern': 'docker'
        }
    }
    
    # Sensor patterns
    SENSOR_PATTERNS = {
        'S3KeySensor': {'type': 's3', 'params': ['bucket_key', 'bucket_name']},
        'SqlSensor': {'type': 'sql', 'params': ['sql', 'conn_id']},
        'FileSensor': {'type': 'file', 'params': ['filepath']},
        'ExternalTaskSensor': {'type': 'task', 'params': ['external_dag_id', 'external_task_id']},
        'HttpSensor': {'type': 'http', 'params': ['endpoint']},
        'TimeDeltaSensor': {'type': 'time', 'params': ['delta']},
        'TimeSensor': {'type': 'time', 'params': ['target_time']},
    }
    
    def __init__(self):
        self.dags: Dict[str, AirflowDAG] = {}
        self.tasks: Dict[str, AirflowTask] = {}
        self.data_assets: Dict[str, AirflowDataAsset] = {}
        self.connections: Dict[str, Dict] = {}
        self.variables: Dict[str, Any] = {}
        self.task_dependencies: List[Tuple[str, str]] = []
        
    def parse_dag_file(self, file_path: str) -> Tuple[List[AirflowDAG], List[AirflowTask]]:
        """
        Parse an Airflow DAG file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        dags = []
        tasks = []
        
        # Parse Python AST
        try:
            tree = ast.parse(content)
            
            # Extract DAG definitions
            dag_defs = self._extract_dag_definitions(tree, file_path)
            dags.extend(dag_defs)
            
            # Extract tasks for each DAG
            for dag in dag_defs:
                dag_tasks = self._extract_tasks_from_dag(tree, dag.dag_id, file_path)
                tasks.extend(dag_tasks)
                
                # Add tasks to DAG
                for task in dag_tasks:
                    dag.tasks[task.task_id] = task
            
            # Extract dependencies
            self._extract_dependencies(tree, dags)
            
            # Extract data assets
            self._extract_data_assets_from_tasks(tasks)
            
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        
        return dags, tasks
    
    def _extract_dag_definitions(self, tree: ast.AST, file_path: str) -> List[AirflowDAG]:
        """Extract DAG definitions from AST"""
        dags = []
        
        class DAGVisitor(ast.NodeVisitor):
            def __init__(self, parser, file_path):
                self.parser = parser
                self.file_path = file_path
                self.dags = []
                
            def visit_With(self, node):
                # Handle: with DAG(...) as dag:
                for item in node.items:
                    if self._is_dag_constructor(item.context_expr):
                        dag = self._extract_dag_from_constructor(
                            item.context_expr,
                            item.optional_vars.id if item.optional_vars else None
                        )
                        if dag:
                            self.dags.append(dag)
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                # Handle: dag = DAG(...)
                if self._is_dag_constructor(node.value):
                    dag = self._extract_dag_from_constructor(
                        node.value,
                        node.targets[0].id if node.targets else None
                    )
                    if dag:
                        self.dags.append(dag)
                self.generic_visit(node)
                
            def _is_dag_constructor(self, node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'DAG':
                        return True
                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'DAG':
                        return True
                return False
                
            def _extract_dag_from_constructor(self, node, var_name):
                dag_id = None
                params = {}
                
                # Extract arguments
                for keyword in node.keywords:
                    key = keyword.arg
                    value = self._extract_value(keyword.value)
                    params[key] = value
                    
                    if key == 'dag_id':
                        dag_id = value
                
                # Check positional arguments
                if node.args and not dag_id:
                    dag_id = self._extract_value(node.args[0])
                
                if dag_id:
                    dag = AirflowDAG(
                        dag_id=dag_id,
                        description=params.get('description'),
                        schedule_interval=params.get('schedule_interval'),
                        start_date=params.get('start_date'),
                        end_date=params.get('end_date'),
                        catchup=params.get('catchup', True),
                        tags=params.get('tags', []),
                        default_args=params.get('default_args', {}),
                        params=params,
                        source_file=self.file_path
                    )
                    return dag
                
                return None
                
            def _extract_value(self, node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.List):
                    return [self._extract_value(elt) for elt in node.elts]
                elif isinstance(node, ast.Dict):
                    return {
                        self._extract_value(k): self._extract_value(v)
                        for k, v in zip(node.keys, node.values)
                    }
                elif isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self._extract_value(node.value)}.{node.attr}"
                elif isinstance(node, ast.Call):
                    # Handle datetime, timedelta, etc.
                    if hasattr(node.func, 'id'):
                        return f"{node.func.id}(...)"
                    elif hasattr(node.func, 'attr'):
                        return f"{node.func.attr}(...)"
                return None
        
        visitor = DAGVisitor(self, file_path)
        visitor.visit(tree)
        
        return visitor.dags
    
    def _extract_tasks_from_dag(self, tree: ast.AST, dag_id: str, file_path: str) -> List[AirflowTask]:
        """Extract tasks from a DAG"""
        tasks = []
        
        class TaskVisitor(ast.NodeVisitor):
            def __init__(self, parser, dag_id, file_path):
                self.parser = parser
                self.dag_id = dag_id
                self.file_path = file_path
                self.tasks = []
                self.current_dag_context = None
                
            def visit_With(self, node):
                # Track DAG context
                for item in node.items:
                    if hasattr(item.optional_vars, 'id'):
                        self.current_dag_context = item.optional_vars.id
                
                self.generic_visit(node)
                self.current_dag_context = None
                
            def visit_Assign(self, node):
                # Check if this is a task assignment
                if isinstance(node.value, ast.Call):
                    task = self._extract_task_from_call(node.value, node.targets)
                    if task:
                        self.tasks.append(task)
                self.generic_visit(node)
                
            def _extract_task_from_call(self, node, targets):
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    return None
                
                # Get operator name
                if isinstance(node.func, ast.Name):
                    operator_name = node.func.id
                else:
                    operator_name = node.func.attr
                
                # Check if it's an Airflow operator
                if not any(op in operator_name for op in ['Operator', 'Sensor']):
                    return None
                
                # Extract task parameters
                params = {}
                for keyword in node.keywords:
                    key = keyword.arg
                    value = self._extract_value(keyword.value)
                    params[key] = value
                
                task_id = params.get('task_id', 'unnamed_task')
                
                # Get task variable name
                task_var = None
                if targets and hasattr(targets[0], 'id'):
                    task_var = targets[0].id
                
                # Create task object
                task = AirflowTask(
                    task_id=task_id,
                    operator_type=operator_name,
                    dag_id=self.dag_id,
                    trigger_rule=params.get('trigger_rule', 'all_success'),
                    retries=params.get('retries', 0),
                    pool=params.get('pool'),
                    queue=params.get('queue'),
                    params=params,
                    source_file=self.file_path,
                    line_number=node.lineno if hasattr(node, 'lineno') else 0
                )
                
                # Extract operator-specific information
                self._extract_operator_details(task, operator_name, params)
                
                return task
                
            def _extract_operator_details(self, task, operator_name, params):
                """Extract details specific to operator type"""
                
                # SQL operators
                if 'Sql' in operator_name or 'SQL' in operator_name:
                    task.sql_query = params.get('sql', '')
                    self._extract_sql_tables(task, task.sql_query)
                
                # Python operator
                elif operator_name == 'PythonOperator':
                    task.python_callable = params.get('python_callable', '')
                    
                # Bash operator
                elif operator_name == 'BashOperator':
                    task.bash_command = params.get('bash_command', '')
                    
                # S3 operators
                elif 'S3' in operator_name:
                    if 'bucket' in params:
                        task.data_inputs.append(f"s3://{params['bucket']}")
                    if 's3_key' in params:
                        task.data_inputs.append(f"s3://{params.get('bucket', 'unknown')}/{params['s3_key']}")
                        
                # BigQuery operators
                elif 'BigQuery' in operator_name:
                    if 'destination_dataset_table' in params:
                        task.data_outputs.append(f"bigquery:{params['destination_dataset_table']}")
                    if 'dataset_id' in params and 'table_id' in params:
                        task.data_outputs.append(f"bigquery:{params['dataset_id']}.{params['table_id']}")
                        
                # GCS operators
                elif 'GCS' in operator_name or 'Gcs' in operator_name:
                    if 'bucket' in params:
                        task.data_inputs.append(f"gs://{params['bucket']}")
                        
                # Databricks operators
                elif 'Databricks' in operator_name:
                    if 'notebook_task' in params:
                        task.data_inputs.append(f"databricks:{params['notebook_task'].get('notebook_path', 'unknown')}")
                
            def _extract_sql_tables(self, task, sql):
                """Extract table references from SQL"""
                if not sql:
                    return
                    
                # Simple regex patterns for table extraction
                patterns = [
                    r'FROM\s+([^\s,;]+)',
                    r'JOIN\s+([^\s,;]+)',
                    r'INTO\s+([^\s,;]+)',
                    r'UPDATE\s+([^\s,;]+)',
                    r'INSERT\s+INTO\s+([^\s,;]+)',
                    r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s,;(]+)',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, sql, re.IGNORECASE)
                    for match in matches:
                        table_name = match.strip('`"[]')
                        if 'FROM' in pattern or 'JOIN' in pattern:
                            task.data_inputs.append(f"table:{table_name}")
                        else:
                            task.data_outputs.append(f"table:{table_name}")
                
            def _extract_value(self, node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Str):
                    return node.s
                elif isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.List):
                    return [self._extract_value(elt) for elt in node.elts]
                elif isinstance(node, ast.Dict):
                    return {
                        self._extract_value(k): self._extract_value(v)
                        for k, v in zip(node.keys, node.values)
                        if k is not None
                    }
                elif isinstance(node, ast.Attribute):
                    return f"{self._extract_value(node.value)}.{node.attr}"
                return None
        
        visitor = TaskVisitor(self, dag_id, file_path)
        visitor.visit(tree)
        
        return visitor.tasks
    
    def _extract_dependencies(self, tree: ast.AST, dags: List[AirflowDAG]):
        """Extract task dependencies"""
        
        class DependencyVisitor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.dependencies = []
                
            def visit_BinOp(self, node):
                # Handle: task1 >> task2 or task1 << task2
                if isinstance(node.op, ast.RShift):  # >>
                    upstream = self._get_task_id(node.left)
                    downstream = self._get_task_id(node.right)
                    if upstream and downstream:
                        self.dependencies.append((upstream, downstream))
                elif isinstance(node.op, ast.LShift):  # <<
                    downstream = self._get_task_id(node.left)
                    upstream = self._get_task_id(node.right)
                    if upstream and downstream:
                        self.dependencies.append((upstream, downstream))
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # Handle: task.set_upstream(other_task) or task.set_downstream(other_task)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'set_upstream':
                        task = self._get_task_id(node.func.value)
                        for arg in node.args:
                            upstream = self._get_task_id(arg)
                            if task and upstream:
                                self.dependencies.append((upstream, task))
                    elif node.func.attr == 'set_downstream':
                        task = self._get_task_id(node.func.value)
                        for arg in node.args:
                            downstream = self._get_task_id(arg)
                            if task and downstream:
                                self.dependencies.append((task, downstream))
                self.generic_visit(node)
                
            def _get_task_id(self, node):
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return node.attr
                elif isinstance(node, ast.List):
                    # Handle list of tasks
                    return [self._get_task_id(elt) for elt in node.elts]
                return None
        
        visitor = DependencyVisitor(self)
        visitor.visit(tree)
        
        # Add dependencies to tasks
        for upstream_id, downstream_id in visitor.dependencies:
            self.task_dependencies.append((upstream_id, downstream_id))
            
            # Update task objects if they exist
            for dag in dags:
                if upstream_id in dag.tasks and downstream_id in dag.tasks:
                    dag.tasks[upstream_id].downstream.append(downstream_id)
                    dag.tasks[downstream_id].dependencies.append(upstream_id)
    
    def _extract_data_assets_from_tasks(self, tasks: List[AirflowTask]):
        """Extract data assets from tasks"""
        
        for task in tasks:
            # Process data inputs
            for input_ref in task.data_inputs:
                asset = self._create_data_asset_from_reference(input_ref)
                if asset:
                    asset.task_references.append(task.get_full_id())
                    self.data_assets[asset.name] = asset
            
            # Process data outputs
            for output_ref in task.data_outputs:
                asset = self._create_data_asset_from_reference(output_ref)
                if asset:
                    asset.task_references.append(task.get_full_id())
                    self.data_assets[asset.name] = asset
    
    def _create_data_asset_from_reference(self, reference: str) -> Optional[AirflowDataAsset]:
        """Create a data asset from a reference string"""
        
        if not reference:
            return None
        
        # Parse reference format (type:name or protocol://path)
        if ':' in reference:
            if '://' in reference:
                # URL format (s3://, gs://, etc.)
                parts = reference.split('://')
                protocol = parts[0]
                path = parts[1] if len(parts) > 1 else ''
                
                return AirflowDataAsset(
                    name=reference,
                    type=protocol,
                    location=reference
                )
            else:
                # Type:name format
                parts = reference.split(':', 1)
                asset_type = parts[0]
                asset_name = parts[1] if len(parts) > 1 else reference
                
                return AirflowDataAsset(
                    name=asset_name,
                    type=asset_type
                )
        
        return AirflowDataAsset(
            name=reference,
            type='unknown'
        )
    
    def parse_dag_folder(self, folder_path: str) -> Dict:
        """Parse all DAG files in a folder"""
        
        dag_files = []
        folder = Path(folder_path)
        
        # Find all Python files that might contain DAGs
        for py_file in folder.glob('**/*.py'):
            # Skip common non-DAG files
            if any(skip in py_file.name for skip in ['__pycache__', '.pyc', '__init__', 'test_']):
                continue
            dag_files.append(py_file)
        
        all_dags = []
        all_tasks = []
        
        for dag_file in dag_files:
            dags, tasks = self.parse_dag_file(str(dag_file))
            all_dags.extend(dags)
            all_tasks.extend(tasks)
            
            # Store in parser state
            for dag in dags:
                self.dags[dag.dag_id] = dag
            for task in tasks:
                self.tasks[task.get_full_id()] = task
        
        return {
            'dags': all_dags,
            'tasks': all_tasks,
            'data_assets': list(self.data_assets.values()),
            'dependencies': self.task_dependencies,
            'graph': self.build_lineage_graph()
        }
    
    def build_lineage_graph(self) -> Dict:
        """Build a lineage graph from parsed DAGs"""
        
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': {}
        }
        
        # Add task nodes
        for task_id, task in self.tasks.items():
            node = {
                'id': task_id,
                'label': task.task_id,
                'type': 'task',
                'operator': task.operator_type,
                'dag': task.dag_id,
                'data_inputs': task.data_inputs,
                'data_outputs': task.data_outputs,
                'properties': task.params
            }
            graph['nodes'].append(node)
        
        # Add data asset nodes
        for asset_name, asset in self.data_assets.items():
            node = {
                'id': f"data:{asset_name}",
                'label': asset.name,
                'type': 'data_asset',
                'asset_type': asset.type,
                'location': asset.location,
                'references': asset.task_references
            }
            graph['nodes'].append(node)
        
        # Add task dependencies
        for upstream, downstream in self.task_dependencies:
            # Find full task IDs
            upstream_full = None
            downstream_full = None
            
            for task_id in self.tasks.keys():
                if task_id.endswith(f".{upstream}"):
                    upstream_full = task_id
                if task_id.endswith(f".{downstream}"):
                    downstream_full = task_id
            
            if upstream_full and downstream_full:
                graph['edges'].append({
                    'source': upstream_full,
                    'target': downstream_full,
                    'type': 'task_dependency'
                })
        
        # Add data flow edges
        for task_id, task in self.tasks.items():
            # Connect data inputs to task
            for input_ref in task.data_inputs:
                asset_id = f"data:{input_ref}"
                if asset_id in [n['id'] for n in graph['nodes']]:
                    graph['edges'].append({
                        'source': asset_id,
                        'target': task_id,
                        'type': 'data_input'
                    })
            
            # Connect task to data outputs
            for output_ref in task.data_outputs:
                asset_id = f"data:{output_ref}"
                if asset_id in [n['id'] for n in graph['nodes']]:
                    graph['edges'].append({
                        'source': task_id,
                        'target': asset_id,
                        'type': 'data_output'
                    })
        
        # Group by DAG
        for dag_id, dag in self.dags.items():
            graph['clusters'][dag_id] = {
                'label': dag_id,
                'description': dag.description,
                'schedule': str(dag.schedule_interval),
                'tasks': [f"{dag_id}.{task_id}" for task_id in dag.tasks.keys()]
            }
        
        return graph
    
    def analyze_dag_complexity(self, dag: AirflowDAG) -> Dict:
        """Analyze DAG complexity metrics"""
        
        metrics = {
            'task_count': len(dag.tasks),
            'max_depth': 0,
            'parallelism': 0,
            'critical_path': [],
            'bottlenecks': [],
            'data_sources': set(),
            'data_sinks': set()
        }
        
        # Build task graph for this DAG
        task_graph = {}
        for task_id, task in dag.tasks.items():
            task_graph[task_id] = {
                'dependencies': task.dependencies,
                'downstream': task.downstream
            }
        
        # Calculate max depth (longest path)
        def calculate_depth(task_id, visited=None):
            if visited is None:
                visited = set()
            if task_id in visited:
                return 0
            visited.add(task_id)
            
            if task_id not in task_graph:
                return 0
            
            downstream = task_graph[task_id]['downstream']
            if not downstream:
                return 1
            
            max_child_depth = 0
            for child in downstream:
                if child in dag.tasks:
                    child_depth = calculate_depth(child, visited.copy())
                    max_child_depth = max(max_child_depth, child_depth)
            
            return 1 + max_child_depth
        
        # Find root tasks (no dependencies)
        root_tasks = [t for t, task in dag.tasks.items() if not task.dependencies]
        
        for root in root_tasks:
            depth = calculate_depth(root)
            metrics['max_depth'] = max(metrics['max_depth'], depth)
        
        # Calculate parallelism (max tasks at same level)
        levels = {}
        for task_id, task in dag.tasks.items():
            level = len(task.dependencies)  # Simple level calculation
            if level not in levels:
                levels[level] = []
            levels[level].append(task_id)
        
        metrics['parallelism'] = max(len(tasks) for tasks in levels.values()) if levels else 0
        
        # Find bottlenecks (tasks with many dependencies or dependents)
        for task_id, task in dag.tasks.items():
            fan_in = len(task.dependencies)
            fan_out = len(task.downstream)
            
            if fan_in > 3 or fan_out > 3:
                metrics['bottlenecks'].append({
                    'task': task_id,
                    'fan_in': fan_in,
                    'fan_out': fan_out
                })
        
        # Identify data sources and sinks
        for task in dag.tasks.values():
            metrics['data_sources'].update(task.data_inputs)
            metrics['data_sinks'].update(task.data_outputs)
        
        metrics['data_sources'] = list(metrics['data_sources'])
        metrics['data_sinks'] = list(metrics['data_sinks'])
        
        return metrics
    
    def detect_patterns(self, dags: List[AirflowDAG]) -> Dict:
        """Detect common patterns in DAGs"""
        
        patterns = {
            'etl_pipelines': [],
            'data_quality_checks': [],
            'notification_tasks': [],
            'sensor_patterns': [],
            'branching_patterns': [],
            'retry_patterns': [],
            'cross_dag_dependencies': []
        }
        
        for dag in dags:
            # Detect ETL pattern
            has_extract = any('extract' in t.task_id.lower() or 
                            any(op in t.operator_type for op in ['S3', 'GCS', 'SFTP', 'Http'])
                            for t in dag.tasks.values())
            has_transform = any('transform' in t.task_id.lower() or 
                              'Spark' in t.operator_type or 
                              'Python' in t.operator_type
                              for t in dag.tasks.values())
            has_load = any('load' in t.task_id.lower() or 
                         any(op in t.operator_type for op in ['BigQuery', 'Redshift', 'Postgres'])
                         for t in dag.tasks.values())
            
            if has_extract and has_transform and has_load:
                patterns['etl_pipelines'].append(dag.dag_id)
            
            # Detect data quality checks
            for task in dag.tasks.values():
                if any(check in task.task_id.lower() for check in ['check', 'validate', 'quality', 'test']):
                    patterns['data_quality_checks'].append(f"{dag.dag_id}.{task.task_id}")
            
            # Detect notification tasks
            for task in dag.tasks.values():
                if any(op in task.operator_type for op in ['Email', 'Slack', 'SNS', 'Webhook']):
                    patterns['notification_tasks'].append(f"{dag.dag_id}.{task.task_id}")
            
            # Detect sensors
            for task in dag.tasks.values():
                if 'Sensor' in task.operator_type:
                    patterns['sensor_patterns'].append({
                        'task': f"{dag.dag_id}.{task.task_id}",
                        'type': task.operator_type
                    })
            
            # Detect branching
            for task in dag.tasks.values():
                if 'Branch' in task.operator_type:
                    patterns['branching_patterns'].append(f"{dag.dag_id}.{task.task_id}")
            
            # Detect retry patterns
            for task in dag.tasks.values():
                if task.retries > 0:
                    patterns['retry_patterns'].append({
                        'task': f"{dag.dag_id}.{task.task_id}",
                        'retries': task.retries,
                        'retry_delay': str(task.retry_delay) if task.retry_delay else None
                    })
            
            # Detect cross-DAG dependencies
            for task in dag.tasks.values():
                if task.operator_type == 'ExternalTaskSensor':
                    patterns['cross_dag_dependencies'].append({
                        'from_dag': dag.dag_id,
                        'from_task': task.task_id,
                        'to_dag': task.params.get('external_dag_id'),
                        'to_task': task.params.get('external_task_id')
                    })
        
        return patterns
    
    def generate_mermaid_diagram(self, dag: AirflowDAG) -> str:
        """Generate Mermaid diagram for DAG visualization"""
        
        lines = ["graph TD"]
        
        # Add tasks
        for task_id, task in dag.tasks.items():
            # Determine node shape based on operator type
            if 'Sensor' in task.operator_type:
                shape = f"{task_id}[/{task_id}/]"  # Parallelogram for sensors
            elif 'Branch' in task.operator_type:
                shape = f"{task_id}{{{task_id}}}"  # Diamond for branching
            else:
                shape = f"{task_id}[{task_id}]"  # Rectangle for regular tasks
            
            lines.append(f"    {shape}")
        
        # Add dependencies
        for task_id, task in dag.tasks.items():
            for downstream in task.downstream:
                if downstream in dag.tasks:
                    lines.append(f"    {task_id} --> {downstream}")
        
        # Add data flows if available
        for task_id, task in dag.tasks.items():
            for input_ref in task.data_inputs:
                safe_ref = input_ref.replace(':', '_').replace('/', '_')
                lines.append(f"    {safe_ref}[({input_ref})] -.-> {task_id}")
            
            for output_ref in task.data_outputs:
                safe_ref = output_ref.replace(':', '_').replace('/', '_')
                lines.append(f"    {task_id} -.-> {safe_ref}[(

)]")
        
        return '\n'.join(lines)


# Utility functions for integration

def parse_airflow_dag(file_path: str) -> Tuple[List[AirflowDAG], List[AirflowTask]]:
    """Parse a single Airflow DAG file"""
    parser = AirflowParser()
    return parser.parse_dag_file(file_path)


def parse_airflow_dags_folder(folder_path: str) -> Dict:
    """Parse all DAG files in an Airflow DAGs folder"""
    parser = AirflowParser()
    return parser.parse_dag_folder(folder_path)


def analyze_airflow_project(project_path: str) -> Dict:
    """Analyze complete Airflow project including DAGs, plugins, and configs"""
    
    parser = AirflowParser()
    project = Path(project_path)
    
    results = {
        'dags': [],
        'tasks': [],
        'data_assets': [],
        'plugins': [],
        'connections': [],
        'variables': [],
        'patterns': {},
        'metrics': {}
    }
    
    # Parse DAGs folder
    dags_folder = project / 'dags'
    if dags_folder.exists():
        dag_results = parser.parse_dag_folder(str(dags_folder))
        results.update(dag_results)
    
    # Detect patterns
    if results['dags']:
        results['patterns'] = parser.detect_patterns(results['dags'])
    
    # Calculate metrics for each DAG
    results['metrics'] = {}
    for dag in results['dags']:
        results['metrics'][dag.dag_id] = parser.analyze_dag_complexity(dag)
    
    # Parse plugins if exists
    plugins_folder = project / 'plugins'
    if plugins_folder.exists():
        # Would parse custom operators, hooks, etc.
        pass
    
    # Parse airflow.cfg if exists
    config_file = project / 'airflow.cfg'
    if config_file.exists():
        # Would parse configuration
        pass
    
    return results
