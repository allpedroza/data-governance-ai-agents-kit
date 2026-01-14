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
Data Sampler - Collects samples from various data sources

Supports:
- Parquet files
- Delta Lake tables
- SQL databases (via SQLAlchemy)
- CSV files
- JSON files
"""

import re
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ColumnProfile:
    """Profile of a column based on sample data"""
    name: str
    data_type: str
    sample_values: List[Any]
    null_count: int
    distinct_count: int
    total_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_length: Optional[float] = None  # For strings
    patterns: List[str] = field(default_factory=list)  # Detected patterns (email, phone, etc.)
    inferred_semantic_type: Optional[str] = None  # pii, date, currency, id, etc.

    @property
    def null_ratio(self) -> float:
        return self.null_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def distinct_ratio(self) -> float:
        return self.distinct_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "sample_values": [str(v)[:100] for v in self.sample_values[:5]],  # Limit for display
            "null_count": self.null_count,
            "null_ratio": round(self.null_ratio, 4),
            "distinct_count": self.distinct_count,
            "distinct_ratio": round(self.distinct_ratio, 4),
            "total_count": self.total_count,
            "min_value": str(self.min_value) if self.min_value is not None else None,
            "max_value": str(self.max_value) if self.max_value is not None else None,
            "avg_length": round(self.avg_length, 2) if self.avg_length else None,
            "patterns": self.patterns,
            "inferred_semantic_type": self.inferred_semantic_type
        }


@dataclass
class SampleResult:
    """Result of data sampling"""
    source: str
    table_name: str
    row_count: int
    sample_size: int
    columns: List[ColumnProfile]
    sample_rows: List[Dict[str, Any]]
    schema: Dict[str, str]  # column_name -> data_type
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "sample_size": self.sample_size,
            "columns": [col.to_dict() for col in self.columns],
            "sample_rows": self.sample_rows[:10],  # Limit rows
            "schema": self.schema,
            "metadata": self.metadata,
            "sampled_at": self.sampled_at
        }

    def get_column_names(self) -> List[str]:
        return [col.name for col in self.columns]

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_text_summary(self) -> str:
        """Generate text summary for LLM context"""
        lines = [
            f"Tabela: {self.table_name}",
            f"Fonte: {self.source}",
            f"Total de linhas: {self.row_count:,}",
            f"Amostra: {self.sample_size} linhas",
            "",
            "Colunas:"
        ]

        for col in self.columns:
            col_info = f"  - {col.name} ({col.data_type})"
            if col.inferred_semantic_type:
                col_info += f" [tipo inferido: {col.inferred_semantic_type}]"
            if col.patterns:
                col_info += f" [padrões: {', '.join(col.patterns)}]"
            col_info += f" | nulos: {col.null_ratio:.1%}, distintos: {col.distinct_count}"

            # Sample values
            if col.sample_values:
                samples = [str(v)[:30] for v in col.sample_values[:3]]
                col_info += f" | ex: {', '.join(samples)}"

            lines.append(col_info)

        return "\n".join(lines)


class DataSampler(ABC):
    """Abstract base class for data samplers"""

    # Patterns for semantic type inference
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone": r'^[\+]?[(]?[0-9]{2,3}[)]?[-\s\.]?[0-9]{4,5}[-\s\.]?[0-9]{4}$',
        "cpf": r'^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$',
        "cnpj": r'^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$',
        "cep": r'^\d{5}-?\d{3}$',
        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "date_iso": r'^\d{4}-\d{2}-\d{2}',
        "date_br": r'^\d{2}/\d{2}/\d{4}$',
        "currency": r'^R?\$?\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?$',
        "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        "url": r'^https?://[^\s]+$',
        "credit_card": r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
    }

    # Column name patterns for semantic type inference
    NAME_PATTERNS = {
        "pii": ["cpf", "cnpj", "rg", "ssn", "passport", "documento"],
        "email": ["email", "e_mail", "e-mail", "mail"],
        "phone": ["phone", "telefone", "celular", "fone", "tel"],
        "name": ["nome", "name", "primeiro_nome", "sobrenome", "full_name"],
        "address": ["endereco", "address", "logradouro", "rua", "cep", "zip"],
        "date": ["data", "date", "dt_", "_dt", "created", "updated", "born"],
        "id": ["_id", "id_", "codigo", "code", "key", "pk", "fk"],
        "amount": ["valor", "amount", "price", "preco", "total", "subtotal"],
        "flag": ["flag", "is_", "has_", "ind_", "flg_", "ativo", "active"]
    }

    @abstractmethod
    def sample(
        self,
        source: str,
        sample_size: int = 100,
        random: bool = True
    ) -> SampleResult:
        """
        Sample data from source

        Args:
            source: Path or connection string
            sample_size: Number of rows to sample
            random: Whether to use random sampling

        Returns:
            SampleResult with profiles and sample data
        """
        pass

    def _infer_semantic_type(
        self,
        column_name: str,
        values: List[Any],
        data_type: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Infer semantic type from column name and values

        Returns:
            Tuple of (semantic_type, detected_patterns)
        """
        detected_patterns = []
        semantic_type = None

        # Check column name patterns
        col_lower = column_name.lower()
        for sem_type, patterns in self.NAME_PATTERNS.items():
            if any(p in col_lower for p in patterns):
                semantic_type = sem_type
                break

        # Check value patterns
        if values:
            non_null_values = [str(v) for v in values if v is not None]

            if non_null_values:
                for pattern_name, pattern in self.PATTERNS.items():
                    matches = sum(1 for v in non_null_values[:20]
                                if re.match(pattern, str(v), re.IGNORECASE))
                    if matches >= len(non_null_values[:20]) * 0.5:  # 50% threshold
                        detected_patterns.append(pattern_name)

                        # Override semantic type based on value patterns
                        if pattern_name in ["email", "phone", "cpf", "cnpj", "credit_card"]:
                            semantic_type = "pii"
                        elif pattern_name in ["date_iso", "date_br"]:
                            semantic_type = "date"
                        elif pattern_name == "currency":
                            semantic_type = "amount"
                        elif pattern_name == "uuid":
                            semantic_type = "id"

        return semantic_type, detected_patterns

    def _profile_column(
        self,
        name: str,
        values: List[Any],
        data_type: str
    ) -> ColumnProfile:
        """Profile a single column"""
        non_null = [v for v in values if v is not None]
        null_count = len(values) - len(non_null)

        # Get distinct values
        try:
            distinct_values = list(set(str(v) for v in non_null))
        except:
            distinct_values = []

        distinct_count = len(distinct_values)

        # Calculate min/max for comparable types
        min_val = None
        max_val = None
        try:
            if non_null and data_type not in ["object", "str", "string"]:
                min_val = min(non_null)
                max_val = max(non_null)
        except:
            pass

        # Calculate average length for strings
        avg_length = None
        if data_type in ["object", "str", "string", "varchar", "text"]:
            lengths = [len(str(v)) for v in non_null if v]
            if lengths:
                avg_length = sum(lengths) / len(lengths)

        # Infer semantic type
        semantic_type, patterns = self._infer_semantic_type(name, non_null, data_type)

        # Get sample values (unique, non-null)
        sample_values = distinct_values[:10] if distinct_values else []

        return ColumnProfile(
            name=name,
            data_type=data_type,
            sample_values=sample_values,
            null_count=null_count,
            distinct_count=distinct_count,
            total_count=len(values),
            min_value=min_val,
            max_value=max_val,
            avg_length=avg_length,
            patterns=patterns,
            inferred_semantic_type=semantic_type
        )


class ParquetSampler(DataSampler):
    """Sampler for Parquet files"""

    def sample(
        self,
        source: str,
        sample_size: int = 100,
        random: bool = True
    ) -> SampleResult:
        try:
            import pyarrow.parquet as pq
            import pandas as pd
        except ImportError:
            raise ImportError("pyarrow is required for Parquet sampling. Install with: pip install pyarrow")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {source}")

        # Read parquet file
        table = pq.read_table(source)
        df = table.to_pandas()

        total_rows = len(df)

        # Sample
        if random and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df.head(sample_size)

        # Get schema
        schema = {col: str(df[col].dtype) for col in df.columns}

        # Profile columns
        columns = []
        for col in df.columns:
            profile = self._profile_column(
                name=col,
                values=sample_df[col].tolist(),
                data_type=str(df[col].dtype)
            )
            columns.append(profile)

        # Convert sample rows
        sample_rows = sample_df.head(20).to_dict(orient='records')

        return SampleResult(
            source=source,
            table_name=path.stem,
            row_count=total_rows,
            sample_size=len(sample_df),
            columns=columns,
            sample_rows=sample_rows,
            schema=schema,
            metadata={"format": "parquet", "file_size": path.stat().st_size}
        )


class CSVSampler(DataSampler):
    """Sampler for CSV files"""

    def sample(
        self,
        source: str,
        sample_size: int = 100,
        random: bool = True,
        encoding: str = "utf-8",
        separator: str = ","
    ) -> SampleResult:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV sampling. Install with: pip install pandas")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {source}")

        # Read CSV
        df = pd.read_csv(source, encoding=encoding, sep=separator)

        total_rows = len(df)

        # Sample
        if random and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df.head(sample_size)

        # Get schema
        schema = {col: str(df[col].dtype) for col in df.columns}

        # Profile columns
        columns = []
        for col in df.columns:
            profile = self._profile_column(
                name=col,
                values=sample_df[col].tolist(),
                data_type=str(df[col].dtype)
            )
            columns.append(profile)

        # Convert sample rows
        sample_rows = sample_df.head(20).to_dict(orient='records')

        return SampleResult(
            source=source,
            table_name=path.stem,
            row_count=total_rows,
            sample_size=len(sample_df),
            columns=columns,
            sample_rows=sample_rows,
            schema=schema,
            metadata={"format": "csv", "encoding": encoding, "separator": separator}
        )


class SQLSampler(DataSampler):
    """Sampler for SQL databases via SQLAlchemy"""

    def __init__(self, connection_string: str):
        """
        Initialize SQL Sampler

        Args:
            connection_string: SQLAlchemy connection string
                e.g., "postgresql://user:pass@host:5432/db"
                      "mysql+pymysql://user:pass@host/db"
                      "sqlite:///path/to/db.sqlite"
        """
        try:
            from sqlalchemy import create_engine, text, inspect
        except ImportError:
            raise ImportError("sqlalchemy is required for SQL sampling. Install with: pip install sqlalchemy")

        self.connection_string = connection_string
        self.engine = create_engine(connection_string)

    def sample(
        self,
        source: str,  # table name
        sample_size: int = 100,
        random: bool = True,
        schema: Optional[str] = None
    ) -> SampleResult:
        from sqlalchemy import text, inspect
        import pandas as pd

        table_name = source
        full_table_name = f"{schema}.{table_name}" if schema else table_name

        # Get total row count
        with self.engine.connect() as conn:
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}"))
            total_rows = count_result.scalar()

        # Sample query
        if random:
            # Database-specific random sampling
            dialect = self.engine.dialect.name
            if dialect == "postgresql":
                query = f"SELECT * FROM {full_table_name} ORDER BY RANDOM() LIMIT {sample_size}"
            elif dialect == "mysql":
                query = f"SELECT * FROM {full_table_name} ORDER BY RAND() LIMIT {sample_size}"
            elif dialect == "sqlite":
                query = f"SELECT * FROM {full_table_name} ORDER BY RANDOM() LIMIT {sample_size}"
            else:
                query = f"SELECT * FROM {full_table_name} LIMIT {sample_size}"
        else:
            query = f"SELECT * FROM {full_table_name} LIMIT {sample_size}"

        # Execute sample query
        df = pd.read_sql(query, self.engine)

        # Get schema from database
        inspector = inspect(self.engine)
        db_columns = inspector.get_columns(table_name, schema=schema)
        db_schema = {col['name']: str(col['type']) for col in db_columns}

        # Profile columns
        columns = []
        for col in df.columns:
            profile = self._profile_column(
                name=col,
                values=df[col].tolist(),
                data_type=db_schema.get(col, str(df[col].dtype))
            )
            columns.append(profile)

        # Convert sample rows
        sample_rows = df.head(20).to_dict(orient='records')

        return SampleResult(
            source=self.connection_string.split("@")[-1] if "@" in self.connection_string else source,
            table_name=full_table_name,
            row_count=total_rows,
            sample_size=len(df),
            columns=columns,
            sample_rows=sample_rows,
            schema=db_schema,
            metadata={"format": "sql", "dialect": self.engine.dialect.name}
        )

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List available tables"""
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema)


class DeltaSampler(DataSampler):
    """Sampler for Delta Lake tables"""

    def sample(
        self,
        source: str,  # Delta table path
        sample_size: int = 100,
        random: bool = True,
        version: Optional[int] = None
    ) -> SampleResult:
        try:
            from deltalake import DeltaTable
            import pandas as pd
        except ImportError:
            raise ImportError("deltalake is required for Delta sampling. Install with: pip install deltalake")

        # Load Delta table
        if version is not None:
            dt = DeltaTable(source, version=version)
        else:
            dt = DeltaTable(source)

        # Get metadata
        metadata = dt.metadata()
        table_name = metadata.name if metadata.name else Path(source).name

        # Convert to pandas
        df = dt.to_pandas()
        total_rows = len(df)

        # Sample
        if random and len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df.head(sample_size)

        # Get schema
        schema_info = dt.schema()
        db_schema = {field.name: str(field.type) for field in schema_info.fields}

        # Profile columns
        columns = []
        for col in df.columns:
            profile = self._profile_column(
                name=col,
                values=sample_df[col].tolist(),
                data_type=db_schema.get(col, str(df[col].dtype))
            )
            columns.append(profile)

        # Convert sample rows
        sample_rows = sample_df.head(20).to_dict(orient='records')

        return SampleResult(
            source=source,
            table_name=table_name,
            row_count=total_rows,
            sample_size=len(sample_df),
            columns=columns,
            sample_rows=sample_rows,
            schema=db_schema,
            metadata={
                "format": "delta",
                "version": dt.version(),
                "description": metadata.description,
                "partition_columns": metadata.partition_columns
            }
        )


def create_sampler(source_type: str, **kwargs) -> DataSampler:
    """
    Factory function to create appropriate sampler

    Args:
        source_type: One of 'parquet', 'csv', 'sql', 'delta'
        **kwargs: Additional arguments for specific samplers

    Returns:
        Appropriate DataSampler instance
    """
    samplers = {
        "parquet": ParquetSampler,
        "csv": CSVSampler,
        "sql": SQLSampler,
        "delta": DeltaSampler
    }

    if source_type not in samplers:
        raise ValueError(f"Unknown source type: {source_type}. Supported: {list(samplers.keys())}")

    if source_type == "sql":
        if "connection_string" not in kwargs:
            raise ValueError("SQL sampler requires 'connection_string' argument")
        return SQLSampler(kwargs["connection_string"])

    return samplers[source_type]()
