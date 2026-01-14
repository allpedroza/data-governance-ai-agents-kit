"""
Data Connectors - Unified interface for data sources

Supports:
- Parquet files
- CSV files
- SQL databases
- Delta Lake tables
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataSource:
    """Information about a data source"""
    name: str
    source_type: str
    path_or_connection: str
    schema: Dict[str, str] = field(default_factory=dict)
    row_count: Optional[int] = None
    last_modified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "path_or_connection": self.path_or_connection,
            "schema": self.schema,
            "row_count": self.row_count,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "metadata": self.metadata
        }


class DataConnector(ABC):
    """Abstract base class for data connectors"""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type of data source"""
        pass

    @abstractmethod
    def get_info(self) -> DataSource:
        """Get information about the data source"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Get schema as {column_name: data_type}"""
        pass

    @abstractmethod
    def read_sample(self, n_rows: int = 1000) -> List[Dict[str, Any]]:
        """Read a sample of rows as list of dictionaries"""
        pass

    @abstractmethod
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all rows (use with caution for large datasets)"""
        pass

    @abstractmethod
    def get_row_count(self) -> int:
        """Get total row count"""
        pass

    def get_columns(self) -> List[str]:
        """Get list of column names"""
        return list(self.get_schema().keys())


class ParquetConnector(DataConnector):
    """Connector for Parquet files"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        self._df = None
        self._load()

    def _load(self):
        try:
            import pyarrow.parquet as pq
            import pandas as pd
        except ImportError:
            raise ImportError("pyarrow required: pip install pyarrow")

        self._table = pq.read_table(self.file_path)
        self._df = self._table.to_pandas()

    @property
    def source_type(self) -> str:
        return "parquet"

    def get_info(self) -> DataSource:
        stat = self.file_path.stat()
        return DataSource(
            name=self.file_path.stem,
            source_type=self.source_type,
            path_or_connection=str(self.file_path),
            schema=self.get_schema(),
            row_count=len(self._df),
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            metadata={"file_size": stat.st_size}
        )

    def get_schema(self) -> Dict[str, str]:
        return {col: str(self._df[col].dtype) for col in self._df.columns}

    def read_sample(self, n_rows: int = 1000) -> List[Dict[str, Any]]:
        sample = self._df.head(n_rows)
        return sample.to_dict(orient='records')

    def read_all(self) -> List[Dict[str, Any]]:
        return self._df.to_dict(orient='records')

    def get_row_count(self) -> int:
        return len(self._df)


class CSVConnector(DataConnector):
    """Connector for CSV files"""

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        separator: str = ","
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.encoding = encoding
        self.separator = separator
        self._df = None
        self._load()

    def _load(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")

        self._df = pd.read_csv(
            self.file_path,
            encoding=self.encoding,
            sep=self.separator
        )

    @property
    def source_type(self) -> str:
        return "csv"

    def get_info(self) -> DataSource:
        stat = self.file_path.stat()
        return DataSource(
            name=self.file_path.stem,
            source_type=self.source_type,
            path_or_connection=str(self.file_path),
            schema=self.get_schema(),
            row_count=len(self._df),
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            metadata={
                "file_size": stat.st_size,
                "encoding": self.encoding,
                "separator": self.separator
            }
        )

    def get_schema(self) -> Dict[str, str]:
        return {col: str(self._df[col].dtype) for col in self._df.columns}

    def read_sample(self, n_rows: int = 1000) -> List[Dict[str, Any]]:
        sample = self._df.head(n_rows)
        return sample.to_dict(orient='records')

    def read_all(self) -> List[Dict[str, Any]]:
        return self._df.to_dict(orient='records')

    def get_row_count(self) -> int:
        return len(self._df)


class SQLConnector(DataConnector):
    """Connector for SQL databases"""

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        schema: Optional[str] = None
    ):
        try:
            from sqlalchemy import create_engine, text, inspect
            import pandas as pd
        except ImportError:
            raise ImportError("sqlalchemy required: pip install sqlalchemy")

        self.connection_string = connection_string
        self.table_name = table_name
        self.schema_name = schema
        self.engine = create_engine(connection_string)
        self._inspector = inspect(self.engine)

        self.full_table_name = f"{schema}.{table_name}" if schema else table_name

    @property
    def source_type(self) -> str:
        return "sql"

    def get_info(self) -> DataSource:
        return DataSource(
            name=self.full_table_name,
            source_type=self.source_type,
            path_or_connection=self.connection_string.split("@")[-1],
            schema=self.get_schema(),
            row_count=self.get_row_count(),
            metadata={"dialect": self.engine.dialect.name}
        )

    def get_schema(self) -> Dict[str, str]:
        columns = self._inspector.get_columns(
            self.table_name,
            schema=self.schema_name
        )
        return {col["name"]: str(col["type"]) for col in columns}

    def get_detailed_schema(self) -> List[Dict[str, Any]]:
        """Get detailed schema including nullable, primary key, etc."""
        columns = self._inspector.get_columns(
            self.table_name,
            schema=self.schema_name
        )
        pk_columns = self._inspector.get_pk_constraint(
            self.table_name,
            schema=self.schema_name
        ).get("constrained_columns", [])

        result = []
        for col in columns:
            result.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col["name"] in pk_columns,
                "default": str(col.get("default")) if col.get("default") else None
            })
        return result

    def read_sample(self, n_rows: int = 1000) -> List[Dict[str, Any]]:
        import pandas as pd
        query = f"SELECT * FROM {self.full_table_name} LIMIT {n_rows}"
        df = pd.read_sql(query, self.engine)
        return df.to_dict(orient='records')

    def read_all(self) -> List[Dict[str, Any]]:
        import pandas as pd
        query = f"SELECT * FROM {self.full_table_name}"
        df = pd.read_sql(query, self.engine)
        return df.to_dict(orient='records')

    def get_row_count(self) -> int:
        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {self.full_table_name}")
            )
            return result.scalar()

    def read_with_timestamp_filter(
        self,
        timestamp_column: str,
        since: datetime,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Read rows with timestamp filter"""
        import pandas as pd
        query = f"""
            SELECT * FROM {self.full_table_name}
            WHERE {timestamp_column} >= '{since.isoformat()}'
            ORDER BY {timestamp_column} DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, self.engine)
        return df.to_dict(orient='records')


class DeltaConnector(DataConnector):
    """Connector for Delta Lake tables"""

    def __init__(self, table_path: str, version: Optional[int] = None):
        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError("deltalake required: pip install deltalake")

        self.table_path = table_path
        self.version = version

        if version is not None:
            self._dt = DeltaTable(table_path, version=version)
        else:
            self._dt = DeltaTable(table_path)

        self._df = None

    def _ensure_loaded(self):
        if self._df is None:
            self._df = self._dt.to_pandas()

    @property
    def source_type(self) -> str:
        return "delta"

    def get_info(self) -> DataSource:
        self._ensure_loaded()
        metadata = self._dt.metadata()
        return DataSource(
            name=metadata.name or Path(self.table_path).name,
            source_type=self.source_type,
            path_or_connection=self.table_path,
            schema=self.get_schema(),
            row_count=len(self._df),
            metadata={
                "version": self._dt.version(),
                "description": metadata.description,
                "partition_columns": metadata.partition_columns
            }
        )

    def get_schema(self) -> Dict[str, str]:
        schema = self._dt.schema()
        return {field.name: str(field.type) for field in schema.fields}

    def get_detailed_schema(self) -> List[Dict[str, Any]]:
        """Get detailed schema"""
        schema = self._dt.schema()
        return [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "metadata": field.metadata
            }
            for field in schema.fields
        ]

    def read_sample(self, n_rows: int = 1000) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        sample = self._df.head(n_rows)
        return sample.to_dict(orient='records')

    def read_all(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return self._df.to_dict(orient='records')

    def get_row_count(self) -> int:
        self._ensure_loaded()
        return len(self._df)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get table history"""
        history = self._dt.history(limit)
        return history.to_pydict()


def create_connector(
    source_type: str,
    path_or_connection: str,
    **kwargs
) -> DataConnector:
    """
    Factory function to create appropriate connector

    Args:
        source_type: One of 'parquet', 'csv', 'sql', 'delta'
        path_or_connection: File path or connection string
        **kwargs: Additional arguments for specific connectors

    Returns:
        DataConnector instance
    """
    connectors = {
        "parquet": ParquetConnector,
        "csv": CSVConnector,
        "sql": SQLConnector,
        "delta": DeltaConnector
    }

    if source_type not in connectors:
        raise ValueError(f"Unknown source type: {source_type}")

    if source_type == "sql":
        if "table_name" not in kwargs:
            raise ValueError("SQL connector requires 'table_name' argument")
        return SQLConnector(
            connection_string=path_or_connection,
            table_name=kwargs.pop("table_name"),
            schema=kwargs.pop("schema", None)
        )
    elif source_type == "csv":
        return CSVConnector(
            file_path=path_or_connection,
            encoding=kwargs.get("encoding", "utf-8"),
            separator=kwargs.get("separator", ",")
        )
    elif source_type == "delta":
        return DeltaConnector(
            table_path=path_or_connection,
            version=kwargs.get("version")
        )
    else:
        return connectors[source_type](path_or_connection)
