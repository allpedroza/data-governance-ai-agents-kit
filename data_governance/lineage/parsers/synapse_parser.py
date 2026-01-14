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
Advanced Azure Data Factory / Synapse Analytics Parser for Data Lineage Analysis
Analyzes pipelines, datasets, activities, and data flows from JSON definitions
"""

import json
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


@dataclass
class SynapseDataset:
    """Represents an Azure Data Factory/Synapse dataset"""
    name: str
    type: str  # Json, Parquet, AzureSqlTable, DelimitedText, etc.
    linked_service: str
    location: Optional[Dict[str, Any]] = None
    schema: Optional[List[Dict[str, Any]]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""

    def __hash__(self):
        return hash(self.name)

    def get_full_location(self) -> Optional[str]:
        """Build full location path from location properties"""
        if not self.location:
            return None

        location_type = self.location.get('type', '')
        parts = []

        # Handle different storage types
        if 'AzureBlobStorage' in location_type:
            container = self.location.get('container', '')
            folder = self.location.get('folderPath', '')
            filename = self.location.get('fileName', '')
            parts = [p for p in [container, folder, filename] if p]
        elif 'AzureDataLakeStorage' in location_type:
            filesystem = self.location.get('fileSystem', '')
            folder = self.location.get('folderPath', '')
            filename = self.location.get('fileName', '')
            parts = [p for p in [filesystem, folder, filename] if p]
        elif 'AzureSql' in location_type:
            schema = self.location.get('schema', '')
            table = self.location.get('table', '')
            parts = [p for p in [schema, table] if p]

        return '/'.join(parts) if parts else None


@dataclass
class SynapseActivity:
    """Represents an activity in a Data Factory/Synapse pipeline"""
    name: str
    type: str  # Copy, Lookup, Script, DataFlow, etc.
    pipeline_name: str
    dependencies: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    source_type: Optional[str] = None
    sink_type: Optional[str] = None
    source_dataset: Optional[str] = None
    sink_dataset: Optional[str] = None
    sql_query: Optional[str] = None
    script_content: Optional[str] = None
    dataflow_name: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    line_number: int = 0

    def __hash__(self):
        return hash(f"{self.pipeline_name}.{self.name}")

    def get_full_id(self):
        return f"{self.pipeline_name}.{self.name}"


@dataclass
class SynapsePipeline:
    """Represents a Data Factory/Synapse pipeline"""
    name: str
    description: Optional[str] = None
    activities: Dict[str, SynapseActivity] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    annotations: List[str] = field(default_factory=list)
    source_file: str = ""

    def __hash__(self):
        return hash(self.name)


@dataclass
class SynapseLinkedService:
    """Represents a linked service (connection to data source)"""
    name: str
    type: str  # AzureBlobStorage, AzureSqlDatabase, AzureSynapseAnalytics, etc.
    connection_string: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""

    def __hash__(self):
        return hash(self.name)


@dataclass
class SynapseDataFlow:
    """Represents a mapping data flow"""
    name: str
    sources: List[str] = field(default_factory=list)
    sinks: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    script: Optional[str] = None
    source_file: str = ""

    def __hash__(self):
        return hash(self.name)


class SynapseParser:
    """
    Advanced parser for Azure Data Factory and Synapse Analytics JSON definitions
    Extracts pipelines, datasets, activities, and data lineage
    """

    # Activity types that perform data movement or transformation
    DATA_ACTIVITY_TYPES = {
        'Copy': 'Data copy between source and sink',
        'DataFlow': 'Mapping data flow transformation',
        'ExecuteDataFlow': 'Execute a data flow',
        'Script': 'Execute SQL script',
        'SqlServerStoredProcedure': 'Execute stored procedure',
        'AzureSynapseAnalyticsLinkedService': 'Synapse query',
        'Lookup': 'Lookup data from dataset',
        'GetMetadata': 'Get metadata from dataset',
    }

    # Source and sink types
    SOURCE_SINK_TYPES = {
        'AzureBlobStorageLocation': 'blob_storage',
        'AzureDataLakeStorageGen2Location': 'adls_gen2',
        'AzureSqlTableDataset': 'sql_table',
        'AzureSynapseAnalyticsTableDataset': 'synapse_table',
        'DelimitedTextDataset': 'csv',
        'JsonDataset': 'json',
        'ParquetDataset': 'parquet',
        'AvroDataset': 'avro',
    }

    def __init__(self):
        self.pipelines: Dict[str, SynapsePipeline] = {}
        self.datasets: Dict[str, SynapseDataset] = {}
        self.linked_services: Dict[str, SynapseLinkedService] = {}
        self.data_flows: Dict[str, SynapseDataFlow] = {}
        self.activities: Dict[str, SynapseActivity] = {}

    def parse_pipeline(self, file_path: str) -> Optional[SynapsePipeline]:
        """Parse a Data Factory/Synapse pipeline JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # Extract pipeline properties
            properties = content.get('properties', content)
            name = content.get('name', Path(file_path).stem)

            pipeline = SynapsePipeline(
                name=name,
                description=properties.get('description'),
                parameters=properties.get('parameters', {}),
                variables=properties.get('variables', {}),
                annotations=properties.get('annotations', []),
                source_file=file_path
            )

            # Parse activities
            activities = properties.get('activities', [])
            for idx, activity_data in enumerate(activities):
                activity = self._parse_activity(activity_data, name, idx)
                if activity:
                    pipeline.activities[activity.name] = activity
                    self.activities[activity.get_full_id()] = activity

            self.pipelines[name] = pipeline
            return pipeline

        except Exception as e:
            print(f"Error parsing pipeline {file_path}: {str(e)}")
            return None

    def _parse_activity(self, activity_data: Dict, pipeline_name: str, idx: int) -> Optional[SynapseActivity]:
        """Parse a single activity from pipeline JSON"""
        name = activity_data.get('name', f'activity_{idx}')
        activity_type = activity_data.get('type', 'Unknown')

        activity = SynapseActivity(
            name=name,
            type=activity_type,
            pipeline_name=pipeline_name,
            properties=activity_data,
            line_number=idx
        )

        # Extract dependencies
        depends_on = activity_data.get('dependsOn', [])
        for dep in depends_on:
            dep_activity = dep.get('activity')
            if dep_activity:
                activity.dependencies.append(dep_activity)

        # Parse activity-specific properties
        if activity_type == 'Copy':
            self._parse_copy_activity(activity, activity_data)
        elif activity_type in ['DataFlow', 'ExecuteDataFlow']:
            self._parse_dataflow_activity(activity, activity_data)
        elif activity_type == 'Script':
            self._parse_script_activity(activity, activity_data)
        elif activity_type == 'Lookup':
            self._parse_lookup_activity(activity, activity_data)

        return activity

    def _parse_copy_activity(self, activity: SynapseActivity, data: Dict):
        """Parse Copy activity to extract source and sink information"""
        type_properties = data.get('typeProperties', {})

        # Extract source
        source = type_properties.get('source', {})
        activity.source_type = source.get('type')

        # Extract SQL query if present
        if 'sqlReaderQuery' in source:
            activity.sql_query = source['sqlReaderQuery']
        elif 'query' in source:
            activity.sql_query = source['query']

        # Extract sink
        sink = type_properties.get('sink', {})
        activity.sink_type = sink.get('type')

        # Extract dataset references
        inputs = data.get('inputs', [])
        for inp in inputs:
            dataset_ref = inp.get('referenceName')
            if dataset_ref:
                activity.inputs.append(dataset_ref)
                activity.source_dataset = dataset_ref

        outputs = data.get('outputs', [])
        for out in outputs:
            dataset_ref = out.get('referenceName')
            if dataset_ref:
                activity.outputs.append(dataset_ref)
                activity.sink_dataset = dataset_ref

    def _parse_dataflow_activity(self, activity: SynapseActivity, data: Dict):
        """Parse DataFlow activity"""
        type_properties = data.get('typeProperties', {})
        dataflow = type_properties.get('dataflow', {})

        dataflow_ref = dataflow.get('referenceName')
        if dataflow_ref:
            activity.dataflow_name = dataflow_ref

    def _parse_script_activity(self, activity: SynapseActivity, data: Dict):
        """Parse Script activity to extract SQL scripts"""
        type_properties = data.get('typeProperties', {})
        scripts = type_properties.get('scripts', [])

        script_contents = []
        for script in scripts:
            text = script.get('text')
            if text:
                script_contents.append(text)

        if script_contents:
            activity.script_content = '\n'.join(script_contents)

    def _parse_lookup_activity(self, activity: SynapseActivity, data: Dict):
        """Parse Lookup activity"""
        type_properties = data.get('typeProperties', {})
        source = type_properties.get('source', {})

        # Extract SQL query
        if 'sqlReaderQuery' in source:
            activity.sql_query = source['sqlReaderQuery']

        # Extract dataset reference
        dataset = data.get('dataset', {})
        dataset_ref = dataset.get('referenceName')
        if dataset_ref:
            activity.inputs.append(dataset_ref)
            activity.source_dataset = dataset_ref

    def parse_dataset(self, file_path: str) -> Optional[SynapseDataset]:
        """Parse a Data Factory/Synapse dataset JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            properties = content.get('properties', content)
            name = content.get('name', Path(file_path).stem)

            # Extract linked service reference
            linked_service_ref = properties.get('linkedServiceName', {})
            linked_service = linked_service_ref.get('referenceName', 'unknown')

            # Extract type
            dataset_type = properties.get('type', 'Unknown')

            # Extract type properties
            type_properties = properties.get('typeProperties', {})
            location = type_properties.get('location')
            schema = properties.get('schema')

            dataset = SynapseDataset(
                name=name,
                type=dataset_type,
                linked_service=linked_service,
                location=location,
                schema=schema,
                properties=type_properties,
                source_file=file_path
            )

            self.datasets[name] = dataset
            return dataset

        except Exception as e:
            print(f"Error parsing dataset {file_path}: {str(e)}")
            return None

    def parse_linked_service(self, file_path: str) -> Optional[SynapseLinkedService]:
        """Parse a Data Factory/Synapse linked service JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            properties = content.get('properties', content)
            name = content.get('name', Path(file_path).stem)

            service_type = properties.get('type', 'Unknown')
            type_properties = properties.get('typeProperties', {})

            # Extract connection string if available
            connection_string = type_properties.get('connectionString')

            linked_service = SynapseLinkedService(
                name=name,
                type=service_type,
                connection_string=connection_string,
                properties=type_properties,
                source_file=file_path
            )

            self.linked_services[name] = linked_service
            return linked_service

        except Exception as e:
            print(f"Error parsing linked service {file_path}: {str(e)}")
            return None

    def parse_dataflow(self, file_path: str) -> Optional[SynapseDataFlow]:
        """Parse a Data Factory/Synapse data flow JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            properties = content.get('properties', content)
            name = content.get('name', Path(file_path).stem)

            # Extract sources
            sources = []
            source_list = properties.get('sources', [])
            for source in source_list:
                source_name = source.get('name')
                if source_name:
                    sources.append(source_name)

            # Extract sinks
            sinks = []
            sink_list = properties.get('sinks', [])
            for sink in sink_list:
                sink_name = sink.get('name')
                if sink_name:
                    sinks.append(sink_name)

            # Extract transformations
            transformations = []
            transform_list = properties.get('transformations', [])
            for transform in transform_list:
                transform_name = transform.get('name')
                if transform_name:
                    transformations.append(transform_name)

            # Extract script
            script = properties.get('script')

            dataflow = SynapseDataFlow(
                name=name,
                sources=sources,
                sinks=sinks,
                transformations=transformations,
                script=script,
                source_file=file_path
            )

            self.data_flows[name] = dataflow
            return dataflow

        except Exception as e:
            print(f"Error parsing data flow {file_path}: {str(e)}")
            return None

    def parse_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Parse all Data Factory/Synapse JSON files in a directory
        Returns summary statistics
        """
        dir_path = Path(directory_path)
        stats = {
            'pipelines': 0,
            'datasets': 0,
            'linked_services': 0,
            'data_flows': 0,
            'activities': 0
        }

        # Common Data Factory folder structure
        folders = {
            'pipeline': self.parse_pipeline,
            'dataset': self.parse_dataset,
            'linkedService': self.parse_linked_service,
            'dataflow': self.parse_dataflow
        }

        # Try to parse organized structure first
        for folder_name, parse_func in folders.items():
            folder_path = dir_path / folder_name
            if folder_path.exists():
                for json_file in folder_path.glob('*.json'):
                    result = parse_func(str(json_file))
                    if result:
                        if folder_name == 'pipeline':
                            stats['pipelines'] += 1
                            stats['activities'] += len(result.activities)
                        elif folder_name == 'dataset':
                            stats['datasets'] += 1
                        elif folder_name == 'linkedService':
                            stats['linked_services'] += 1
                        elif folder_name == 'dataflow':
                            stats['data_flows'] += 1

        # Also try to parse JSON files in root directory
        for json_file in dir_path.glob('*.json'):
            if json_file.is_file():
                # Try to auto-detect type
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)

                    properties = content.get('properties', {})

                    # Detect by structure
                    if 'activities' in properties:
                        result = self.parse_pipeline(str(json_file))
                        if result:
                            stats['pipelines'] += 1
                            stats['activities'] += len(result.activities)
                    elif 'linkedServiceName' in properties:
                        result = self.parse_dataset(str(json_file))
                        if result:
                            stats['datasets'] += 1
                    elif 'type' in properties and 'typeProperties' in properties:
                        # Could be linked service
                        result = self.parse_linked_service(str(json_file))
                        if result:
                            stats['linked_services'] += 1
                except:
                    continue

        return stats

    def get_lineage_graph(self) -> Dict[str, Any]:
        """
        Build a lineage graph from parsed pipelines and datasets
        Returns a graph structure with nodes and edges
        """
        nodes = []
        edges = []

        # Add datasets as nodes
        for dataset_name, dataset in self.datasets.items():
            location = dataset.get_full_location()
            nodes.append({
                'id': dataset_name,
                'name': dataset_name,
                'type': 'dataset',
                'dataset_type': dataset.type,
                'linked_service': dataset.linked_service,
                'location': location,
                'source_file': dataset.source_file
            })

        # Add activities and their lineage edges
        for activity_id, activity in self.activities.items():
            # Add activity as node
            nodes.append({
                'id': activity_id,
                'name': activity.name,
                'type': 'activity',
                'activity_type': activity.type,
                'pipeline': activity.pipeline_name,
                'source_file': f"{activity.pipeline_name} (line {activity.line_number})"
            })

            # Add edges from inputs to activity
            for input_dataset in activity.inputs:
                edges.append({
                    'source': input_dataset,
                    'target': activity_id,
                    'type': 'reads',
                    'activity_type': activity.type
                })

            # Add edges from activity to outputs
            for output_dataset in activity.outputs:
                edges.append({
                    'source': activity_id,
                    'target': output_dataset,
                    'type': 'writes',
                    'activity_type': activity.type
                })

            # Add dependency edges between activities
            for dep_activity in activity.dependencies:
                dep_id = f"{activity.pipeline_name}.{dep_activity}"
                edges.append({
                    'source': dep_id,
                    'target': activity_id,
                    'type': 'depends_on',
                    'activity_type': activity.type
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'summary': {
                'total_pipelines': len(self.pipelines),
                'total_datasets': len(self.datasets),
                'total_activities': len(self.activities),
                'total_linked_services': len(self.linked_services),
                'total_data_flows': len(self.data_flows)
            }
        }

    def analyze_pipeline_complexity(self, pipeline_name: str) -> Dict[str, Any]:
        """Analyze complexity metrics for a specific pipeline"""
        if pipeline_name not in self.pipelines:
            return {}

        pipeline = self.pipelines[pipeline_name]
        activities = list(pipeline.activities.values())

        # Calculate metrics
        total_activities = len(activities)
        copy_activities = sum(1 for a in activities if a.type == 'Copy')
        dataflow_activities = sum(1 for a in activities if a.type in ['DataFlow', 'ExecuteDataFlow'])
        script_activities = sum(1 for a in activities if a.type == 'Script')

        # Calculate max dependency depth
        def get_depth(activity_name, visited=None):
            if visited is None:
                visited = set()
            if activity_name in visited:
                return 0
            visited.add(activity_name)

            activity = pipeline.activities.get(activity_name)
            if not activity or not activity.dependencies:
                return 1

            max_dep_depth = max((get_depth(dep, visited.copy()) for dep in activity.dependencies), default=0)
            return max_dep_depth + 1

        max_depth = max((get_depth(a.name) for a in activities), default=0)

        # Find activities with no dependencies (entry points)
        entry_points = [a.name for a in activities if not a.dependencies]

        # Find activities with no downstream (exit points)
        all_deps = set()
        for a in activities:
            all_deps.update(a.dependencies)
        exit_points = [a.name for a in activities if a.name not in all_deps]

        return {
            'pipeline_name': pipeline_name,
            'total_activities': total_activities,
            'copy_activities': copy_activities,
            'dataflow_activities': dataflow_activities,
            'script_activities': script_activities,
            'max_dependency_depth': max_depth,
            'entry_points': entry_points,
            'exit_points': exit_points,
            'has_parameters': len(pipeline.parameters) > 0,
            'has_variables': len(pipeline.variables) > 0,
            'parameter_count': len(pipeline.parameters),
            'variable_count': len(pipeline.variables)
        }

    def generate_mermaid_diagram(self, pipeline_name: str) -> str:
        """Generate a Mermaid diagram for pipeline visualization"""
        if pipeline_name not in self.pipelines:
            return ""

        pipeline = self.pipelines[pipeline_name]
        lines = ["graph TD"]

        # Add activity nodes
        for activity_name, activity in pipeline.activities.items():
            # Sanitize name for Mermaid
            node_id = activity_name.replace(' ', '_').replace('-', '_')

            # Different shapes for different activity types
            if activity.type == 'Copy':
                lines.append(f"    {node_id}[{activity_name}<br/>Copy]")
            elif activity.type in ['DataFlow', 'ExecuteDataFlow']:
                lines.append(f"    {node_id}{{{{'{activity_name}<br/>DataFlow'}}}}")
            elif activity.type == 'Script':
                lines.append(f"    {node_id}[({activity_name}<br/>Script)]")
            else:
                lines.append(f"    {node_id}[{activity_name}<br/>{activity.type}]")

            # Add dependency edges
            for dep in activity.dependencies:
                dep_id = dep.replace(' ', '_').replace('-', '_')
                lines.append(f"    {dep_id} --> {node_id}")

        return '\n'.join(lines)


# Utility functions for easy usage
def parse_synapse_pipeline(file_path: str) -> Optional[SynapsePipeline]:
    """Parse a single Synapse/Data Factory pipeline file"""
    parser = SynapseParser()
    return parser.parse_pipeline(file_path)


def parse_synapse_directory(directory_path: str) -> SynapseParser:
    """Parse all Synapse/Data Factory files in a directory"""
    parser = SynapseParser()
    parser.parse_directory(directory_path)
    return parser


def analyze_synapse_project(directory_path: str) -> Dict[str, Any]:
    """Complete analysis of a Synapse/Data Factory project"""
    parser = parse_synapse_directory(directory_path)

    return {
        'parser': parser,
        'lineage_graph': parser.get_lineage_graph(),
        'pipeline_analyses': {
            name: parser.analyze_pipeline_complexity(name)
            for name in parser.pipelines.keys()
        }
    }
