"""
Advanced Parquet File Parser for Data Lineage and Schema Analysis
Analyzes Parquet files to extract schema, metadata, statistics, and data lineage information
"""

import os
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("Warning: pyarrow not available. Install with: pip install pyarrow")


@dataclass
class ParquetColumn:
    """Represents a column in a Parquet file"""
    name: str
    type: str
    physical_type: Optional[str] = None
    logical_type: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    compression: Optional[str] = None
    encoding: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert column to dictionary representation"""
        return {
            'name': self.name,
            'type': self.type,
            'physical_type': self.physical_type,
            'logical_type': self.logical_type,
            'nullable': self.nullable,
            'statistics': self.statistics
        }


@dataclass
class ParquetSchema:
    """Represents the schema of a Parquet file"""
    columns: List[ParquetColumn] = field(default_factory=list)
    partition_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_column(self, name: str) -> Optional[ParquetColumn]:
        """Get column by name"""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation"""
        return {
            'columns': [col.to_dict() for col in self.columns],
            'partition_columns': self.partition_columns,
            'metadata': self.metadata
        }


@dataclass
class ParquetFile:
    """Represents a Parquet file with its metadata and schema"""
    file_path: str
    name: str
    schema: ParquetSchema
    num_rows: int = 0
    num_row_groups: int = 0
    file_size: int = 0
    created_by: Optional[str] = None
    compression: Optional[str] = None
    format_version: Optional[str] = None
    serialized_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    partition_info: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.file_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about the Parquet file"""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'num_rows': self.num_rows,
            'num_columns': len(self.schema.columns),
            'num_row_groups': self.num_row_groups,
            'file_size_mb': round(self.file_size / (1024 * 1024), 2),
            'compression': self.compression,
            'format_version': self.format_version,
            'partition_info': self.partition_info
        }


@dataclass
class ParquetDataset:
    """Represents a collection of Parquet files (e.g., partitioned dataset)"""
    name: str
    root_path: str
    files: List[ParquetFile] = field(default_factory=list)
    common_schema: Optional[ParquetSchema] = None
    partition_columns: List[str] = field(default_factory=list)
    total_rows: int = 0
    total_size: int = 0

    def __hash__(self):
        return hash(self.root_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about the dataset"""
        return {
            'name': self.name,
            'root_path': self.root_path,
            'num_files': len(self.files),
            'total_rows': self.total_rows,
            'total_size_mb': round(self.total_size / (1024 * 1024), 2),
            'partition_columns': self.partition_columns,
            'num_columns': len(self.common_schema.columns) if self.common_schema else 0
        }


class ParquetParser:
    """
    Advanced parser for Parquet files
    Extracts schema, metadata, statistics, and data lineage information
    """

    # Mapping from Parquet types to common data types
    TYPE_MAPPING = {
        'BOOLEAN': 'boolean',
        'INT32': 'int32',
        'INT64': 'int64',
        'INT96': 'timestamp',
        'FLOAT': 'float',
        'DOUBLE': 'double',
        'BYTE_ARRAY': 'string',
        'FIXED_LEN_BYTE_ARRAY': 'binary',
        # Logical types
        'STRING': 'string',
        'UTF8': 'string',
        'ENUM': 'enum',
        'UUID': 'uuid',
        'DECIMAL': 'decimal',
        'DATE': 'date',
        'TIME_MILLIS': 'time',
        'TIME_MICROS': 'time',
        'TIMESTAMP_MILLIS': 'timestamp',
        'TIMESTAMP_MICROS': 'timestamp',
        'JSON': 'json',
        'BSON': 'bson',
        'LIST': 'array',
        'MAP': 'map',
        'STRUCT': 'struct'
    }

    def __init__(self):
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required for ParquetParser. Install with: pip install pyarrow")

        self.files: Dict[str, ParquetFile] = {}
        self.datasets: Dict[str, ParquetDataset] = {}

    def parse_file(self, file_path: str) -> Optional[ParquetFile]:
        """Parse a single Parquet file"""
        try:
            # Read Parquet file metadata
            parquet_file = pq.ParquetFile(file_path)

            # Extract basic metadata
            metadata = parquet_file.metadata
            schema_arrow = parquet_file.schema_arrow

            # Build schema
            schema = self._extract_schema(schema_arrow, metadata)

            # Get file statistics
            file_size = os.path.getsize(file_path)
            file_name = Path(file_path).stem

            # Extract partition information from path
            partition_info = self._extract_partition_info(file_path)

            # Create ParquetFile object
            parquet_obj = ParquetFile(
                file_path=file_path,
                name=file_name,
                schema=schema,
                num_rows=metadata.num_rows,
                num_row_groups=metadata.num_row_groups,
                file_size=file_size,
                created_by=metadata.created_by,
                compression=self._detect_compression(metadata),
                format_version=metadata.format_version,
                serialized_size=metadata.serialized_size,
                metadata=self._extract_metadata(metadata),
                partition_info=partition_info
            )

            self.files[file_path] = parquet_obj
            return parquet_obj

        except Exception as e:
            print(f"Error parsing Parquet file {file_path}: {str(e)}")
            return None

    def _extract_schema(self, arrow_schema: Any, metadata: Any) -> ParquetSchema:
        """Extract schema information from Parquet file"""
        columns = []

        for i, field in enumerate(arrow_schema):
            # Get column statistics if available
            stats = {}
            try:
                if metadata.num_row_groups > 0:
                    row_group = metadata.row_group(0)
                    if i < row_group.num_columns:
                        col_chunk = row_group.column(i)
                        if col_chunk.is_stats_set:
                            col_stats = col_chunk.statistics
                            stats = {
                                'min': str(col_stats.min) if hasattr(col_stats, 'min') else None,
                                'max': str(col_stats.max) if hasattr(col_stats, 'max') else None,
                                'null_count': col_stats.null_count if hasattr(col_stats, 'null_count') else None,
                                'distinct_count': col_stats.distinct_count if hasattr(col_stats, 'distinct_count') else None,
                            }
            except:
                pass

            # Map Arrow type to common type
            arrow_type = str(field.type)
            common_type = self._map_arrow_type(arrow_type)

            column = ParquetColumn(
                name=field.name,
                type=common_type,
                physical_type=arrow_type,
                logical_type=arrow_type,
                nullable=field.nullable,
                statistics=stats
            )
            columns.append(column)

        # Extract schema metadata
        schema_metadata = {}
        if arrow_schema.metadata:
            for key, value in arrow_schema.metadata.items():
                try:
                    schema_metadata[key.decode('utf-8')] = value.decode('utf-8')
                except:
                    schema_metadata[str(key)] = str(value)

        return ParquetSchema(
            columns=columns,
            metadata=schema_metadata
        )

    def _map_arrow_type(self, arrow_type: str) -> str:
        """Map Arrow type to common type"""
        arrow_type_upper = arrow_type.upper()

        # Check direct mappings
        for parquet_type, common_type in self.TYPE_MAPPING.items():
            if parquet_type in arrow_type_upper:
                return common_type

        # Handle complex types
        if 'LIST' in arrow_type_upper or 'ARRAY' in arrow_type_upper:
            return 'array'
        elif 'MAP' in arrow_type_upper:
            return 'map'
        elif 'STRUCT' in arrow_type_upper:
            return 'struct'
        elif 'TIMESTAMP' in arrow_type_upper:
            return 'timestamp'
        elif 'DATE' in arrow_type_upper:
            return 'date'
        elif 'DECIMAL' in arrow_type_upper:
            return 'decimal'
        elif 'STRING' in arrow_type_upper or 'UTF8' in arrow_type_upper:
            return 'string'
        elif 'INT64' in arrow_type_upper or 'BIGINT' in arrow_type_upper:
            return 'int64'
        elif 'INT32' in arrow_type_upper or 'INT' in arrow_type_upper:
            return 'int32'
        elif 'DOUBLE' in arrow_type_upper or 'FLOAT64' in arrow_type_upper:
            return 'double'
        elif 'FLOAT' in arrow_type_upper or 'FLOAT32' in arrow_type_upper:
            return 'float'
        elif 'BOOL' in arrow_type_upper:
            return 'boolean'

        return arrow_type.lower()

    def _detect_compression(self, metadata: Any) -> Optional[str]:
        """Detect compression codec used in the Parquet file"""
        try:
            if metadata.num_row_groups > 0:
                row_group = metadata.row_group(0)
                if row_group.num_columns > 0:
                    col_chunk = row_group.column(0)
                    return col_chunk.compression
        except:
            pass
        return None

    def _extract_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Extract custom metadata from Parquet file"""
        custom_metadata = {}

        try:
            if hasattr(metadata, 'metadata') and metadata.metadata:
                for key, value in metadata.metadata.items():
                    try:
                        custom_metadata[key.decode('utf-8')] = value.decode('utf-8')
                    except:
                        custom_metadata[str(key)] = str(value)
        except:
            pass

        return custom_metadata

    def _extract_partition_info(self, file_path: str) -> Dict[str, str]:
        """Extract partition information from file path"""
        partition_info = {}

        # Look for Hive-style partitioning (e.g., year=2024/month=01/day=15)
        path_parts = Path(file_path).parts
        for part in path_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                partition_info[key] = value

        return partition_info

    def parse_directory(self, directory_path: str, recursive: bool = True) -> Optional[ParquetDataset]:
        """
        Parse all Parquet files in a directory
        Treats them as a partitioned dataset
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            print(f"Directory not found: {directory_path}")
            return None

        # Find all Parquet files
        if recursive:
            parquet_files = list(dir_path.rglob('*.parquet')) + list(dir_path.rglob('*.pq'))
        else:
            parquet_files = list(dir_path.glob('*.parquet')) + list(dir_path.glob('*.pq'))

        if not parquet_files:
            print(f"No Parquet files found in {directory_path}")
            return None

        # Parse all files
        files = []
        total_rows = 0
        total_size = 0
        all_partition_columns = set()

        for file_path in parquet_files:
            parquet_file = self.parse_file(str(file_path))
            if parquet_file:
                files.append(parquet_file)
                total_rows += parquet_file.num_rows
                total_size += parquet_file.file_size
                all_partition_columns.update(parquet_file.partition_info.keys())

        # Determine common schema
        common_schema = files[0].schema if files else None

        # Create dataset
        dataset_name = dir_path.name
        dataset = ParquetDataset(
            name=dataset_name,
            root_path=directory_path,
            files=files,
            common_schema=common_schema,
            partition_columns=sorted(list(all_partition_columns)),
            total_rows=total_rows,
            total_size=total_size
        )

        self.datasets[directory_path] = dataset
        return dataset

    def compare_schemas(self, file_path1: str, file_path2: str) -> Dict[str, Any]:
        """Compare schemas of two Parquet files"""
        file1 = self.files.get(file_path1) or self.parse_file(file_path1)
        file2 = self.files.get(file_path2) or self.parse_file(file_path2)

        if not file1 or not file2:
            return {'error': 'Could not parse one or both files'}

        schema1_cols = {col.name: col for col in file1.schema.columns}
        schema2_cols = {col.name: col for col in file2.schema.columns}

        # Find differences
        only_in_first = set(schema1_cols.keys()) - set(schema2_cols.keys())
        only_in_second = set(schema2_cols.keys()) - set(schema1_cols.keys())
        common = set(schema1_cols.keys()) & set(schema2_cols.keys())

        # Check type differences in common columns
        type_differences = []
        for col_name in common:
            if schema1_cols[col_name].type != schema2_cols[col_name].type:
                type_differences.append({
                    'column': col_name,
                    'type_in_first': schema1_cols[col_name].type,
                    'type_in_second': schema2_cols[col_name].type
                })

        return {
            'file1': file_path1,
            'file2': file_path2,
            'only_in_first': sorted(list(only_in_first)),
            'only_in_second': sorted(list(only_in_second)),
            'common_columns': sorted(list(common)),
            'type_differences': type_differences,
            'schemas_match': len(only_in_first) == 0 and len(only_in_second) == 0 and len(type_differences) == 0
        }

    def detect_relationships(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Detect potential relationships between Parquet files based on column names
        (e.g., foreign key patterns like user_id, customer_id, etc.)
        """
        if dataset_path not in self.datasets:
            self.parse_directory(dataset_path)

        dataset = self.datasets.get(dataset_path)
        if not dataset:
            return []

        relationships = []

        # Common foreign key patterns
        fk_patterns = [
            r'(.+)_id$',  # user_id, customer_id, etc.
            r'(.+)_key$',  # user_key, customer_key, etc.
            r'id_(.+)$',  # id_user, id_customer, etc.
        ]

        # Build index of columns by name
        columns_by_name: Dict[str, List[Tuple[str, ParquetColumn]]] = {}
        for file in dataset.files:
            for col in file.schema.columns:
                if col.name not in columns_by_name:
                    columns_by_name[col.name] = []
                columns_by_name[col.name].append((file.name, col))

        # Look for potential relationships
        for col_name, file_col_pairs in columns_by_name.items():
            # Check if this looks like a foreign key
            for pattern in fk_patterns:
                match = re.match(pattern, col_name.lower())
                if match:
                    entity = match.group(1)

                    # Look for a file with this entity name
                    for file in dataset.files:
                        if entity in file.name.lower():
                            relationships.append({
                                'foreign_key_column': col_name,
                                'in_files': [f[0] for f in file_col_pairs],
                                'references_entity': entity,
                                'potential_primary_table': file.name,
                                'confidence': 'medium'
                            })

        return relationships

    def get_lineage_graph(self) -> Dict[str, Any]:
        """
        Build a lineage graph from parsed Parquet files
        Returns a graph structure with nodes and edges
        """
        nodes = []
        edges = []

        # Add files as nodes
        for file_path, parquet_file in self.files.items():
            nodes.append({
                'id': file_path,
                'name': parquet_file.name,
                'type': 'parquet_file',
                'num_rows': parquet_file.num_rows,
                'num_columns': len(parquet_file.schema.columns),
                'file_size_mb': round(parquet_file.file_size / (1024 * 1024), 2),
                'compression': parquet_file.compression,
                'partition_info': parquet_file.partition_info
            })

        # Add datasets as nodes
        for dataset_path, dataset in self.datasets.items():
            nodes.append({
                'id': dataset_path,
                'name': dataset.name,
                'type': 'parquet_dataset',
                'num_files': len(dataset.files),
                'total_rows': dataset.total_rows,
                'partition_columns': dataset.partition_columns
            })

            # Add edges from dataset to files
            for file in dataset.files:
                edges.append({
                    'source': dataset_path,
                    'target': file.file_path,
                    'type': 'contains'
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'summary': {
                'total_files': len(self.files),
                'total_datasets': len(self.datasets)
            }
        }


# Utility functions for easy usage
def parse_parquet_file(file_path: str) -> Optional[ParquetFile]:
    """Parse a single Parquet file"""
    parser = ParquetParser()
    return parser.parse_file(file_path)


def parse_parquet_directory(directory_path: str, recursive: bool = True) -> Optional[ParquetDataset]:
    """Parse all Parquet files in a directory"""
    parser = ParquetParser()
    return parser.parse_directory(directory_path, recursive)


def analyze_parquet_dataset(directory_path: str) -> Dict[str, Any]:
    """Complete analysis of a Parquet dataset"""
    parser = ParquetParser()
    dataset = parser.parse_directory(directory_path)

    if not dataset:
        return {'error': 'Could not parse dataset'}

    return {
        'parser': parser,
        'dataset': dataset,
        'summary': dataset.get_summary(),
        'lineage_graph': parser.get_lineage_graph(),
        'relationships': parser.detect_relationships(directory_path)
    }
