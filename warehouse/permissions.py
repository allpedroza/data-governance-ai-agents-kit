"""
Warehouse Permissions Module

Role-based access control for data warehouse operations with:
- Module-level permissions (Lineage, Discovery, Enrichment, Classification, Quality, Asset Value)
- Warehouse-level permissions (Snowflake, Redshift, BigQuery, Synapse)
- Operation-level permissions (read, write, schema, execute)
"""

import json
import secrets
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, Enum, Flag, auto
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import os


class WarehouseAccessLevel(IntEnum):
    """
    Access levels for warehouse operations.
    Higher values = more permissions.
    """
    NONE = 0
    READ_METADATA = 10      # Can read schema and metadata only
    READ_SAMPLE = 20        # Can read sample data (limited rows)
    READ_FULL = 30          # Can read all data
    WRITE = 40              # Can write/modify data
    ADMIN = 50              # Full access including DDL operations


class DataModule(Enum):
    """Data governance modules that can access warehouses."""
    LINEAGE = "lineage"
    DISCOVERY = "discovery"
    ENRICHMENT = "enrichment"
    CLASSIFICATION = "classification"
    QUALITY = "quality"
    ASSET_VALUE = "asset_value"
    ALL = "all"


class WarehouseOperation(Flag):
    """Operations that can be performed on warehouses."""
    NONE = 0
    READ_SCHEMA = auto()           # Read table/column metadata
    READ_SAMPLE = auto()           # Read sample rows
    READ_FULL = auto()             # Read all data
    READ_STATS = auto()            # Read statistics/profiling
    WRITE_DATA = auto()            # Write/insert data
    EXECUTE_QUERY = auto()         # Execute arbitrary queries
    CREATE_OBJECTS = auto()        # Create tables/views
    ALTER_OBJECTS = auto()         # Alter existing objects
    DROP_OBJECTS = auto()          # Drop objects

    # Convenience combinations
    READ_ONLY = READ_SCHEMA | READ_SAMPLE | READ_STATS
    READ_ALL = READ_SCHEMA | READ_SAMPLE | READ_FULL | READ_STATS
    READ_WRITE = READ_ALL | WRITE_DATA
    FULL = READ_ALL | WRITE_DATA | EXECUTE_QUERY | CREATE_OBJECTS | ALTER_OBJECTS | DROP_OBJECTS


@dataclass
class ModuleAccess:
    """Defines access permissions for a specific module to a warehouse."""
    module: DataModule
    warehouse_name: str
    access_level: WarehouseAccessLevel
    operations: WarehouseOperation
    allowed_schemas: Set[str] = field(default_factory=set)  # Empty = all schemas
    allowed_tables: Set[str] = field(default_factory=set)   # Empty = all tables
    max_sample_rows: int = 10000
    max_query_rows: int = 100000
    query_timeout_seconds: int = 300
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if this access grant is still valid."""
        if not self.enabled:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def can_access_schema(self, schema: str) -> bool:
        """Check if access to a specific schema is allowed."""
        if not self.allowed_schemas:
            return True
        return schema in self.allowed_schemas

    def can_access_table(self, table: str, schema: Optional[str] = None) -> bool:
        """Check if access to a specific table is allowed."""
        if schema and not self.can_access_schema(schema):
            return False
        if not self.allowed_tables:
            return True

        # Check table name with or without schema prefix
        full_name = f"{schema}.{table}" if schema else table
        return table in self.allowed_tables or full_name in self.allowed_tables

    def can_perform(self, operation: WarehouseOperation) -> bool:
        """Check if a specific operation is allowed."""
        return bool(self.operations & operation)


@dataclass
class WarehousePermission:
    """
    Represents a permission grant for warehouse access.

    This is the central permission object that combines:
    - User/service identity
    - Warehouse target
    - Module permissions
    - Time-based restrictions
    """
    permission_id: str
    name: str
    description: Optional[str] = None

    # Target warehouse(s)
    warehouse_names: Set[str] = field(default_factory=set)  # Empty = all warehouses

    # Module-specific access
    module_access: Dict[DataModule, ModuleAccess] = field(default_factory=dict)

    # Global settings
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_by: Optional[str] = None

    # Audit
    last_used: Optional[datetime] = None
    usage_count: int = 0

    def is_valid(self) -> bool:
        """Check if this permission is still valid."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def get_module_access(
        self,
        module: DataModule,
        warehouse_name: str
    ) -> Optional[ModuleAccess]:
        """Get the access configuration for a specific module and warehouse."""
        if not self.is_valid():
            return None

        # Check warehouse restriction
        if self.warehouse_names and warehouse_name not in self.warehouse_names:
            return None

        # Check for specific module access
        if module in self.module_access:
            access = self.module_access[module]
            if access.warehouse_name == warehouse_name or not access.warehouse_name:
                return access if access.is_valid() else None

        # Check for ALL modules access
        if DataModule.ALL in self.module_access:
            access = self.module_access[DataModule.ALL]
            return access if access.is_valid() else None

        return None

    def record_usage(self) -> None:
        """Record that this permission was used."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1


class WarehousePermissionManager:
    """
    Manages warehouse permissions for data governance modules.

    Features:
    - Create and manage permissions
    - Validate access requests
    - Audit trail
    - Integration with existing vault access control
    """

    def __init__(self, storage_path: str = ".warehouse_permissions"):
        self._storage_path = Path(storage_path)
        self._permissions: Dict[str, WarehousePermission] = {}
        self._ensure_storage()
        self._load_permissions()

    def _ensure_storage(self) -> None:
        """Create storage with secure permissions."""
        if not self._storage_path.exists():
            self._storage_path.mkdir(parents=True, mode=0o700)

    def create_permission(
        self,
        name: str,
        warehouse_names: Optional[Set[str]] = None,
        description: Optional[str] = None,
        expires_days: Optional[int] = None,
        created_by: Optional[str] = None
    ) -> WarehousePermission:
        """
        Create a new warehouse permission.

        Args:
            name: Unique name for this permission
            warehouse_names: Set of warehouse names (None = all)
            description: Optional description
            expires_days: Days until expiration (None = never)
            created_by: User ID who created this permission

        Returns:
            Created WarehousePermission
        """
        permission_id = f"wh_perm_{secrets.token_hex(8)}"

        permission = WarehousePermission(
            permission_id=permission_id,
            name=name,
            description=description,
            warehouse_names=warehouse_names or set(),
            expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None,
            created_by=created_by
        )

        self._permissions[permission_id] = permission
        self._save_permissions()

        return permission

    def grant_module_access(
        self,
        permission_id: str,
        module: DataModule,
        warehouse_name: str,
        access_level: WarehouseAccessLevel,
        operations: WarehouseOperation,
        allowed_schemas: Optional[Set[str]] = None,
        allowed_tables: Optional[Set[str]] = None,
        max_sample_rows: int = 10000,
        expires_days: Optional[int] = None
    ) -> bool:
        """
        Grant module access to a warehouse.

        Args:
            permission_id: Permission to update
            module: Data module to grant access
            warehouse_name: Target warehouse
            access_level: Access level to grant
            operations: Allowed operations
            allowed_schemas: Restrict to specific schemas
            allowed_tables: Restrict to specific tables
            max_sample_rows: Maximum rows for sampling
            expires_days: Days until this grant expires

        Returns:
            True if successful
        """
        permission = self._permissions.get(permission_id)
        if not permission:
            return False

        module_access = ModuleAccess(
            module=module,
            warehouse_name=warehouse_name,
            access_level=access_level,
            operations=operations,
            allowed_schemas=allowed_schemas or set(),
            allowed_tables=allowed_tables or set(),
            max_sample_rows=max_sample_rows,
            expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
        )

        permission.module_access[module] = module_access
        permission.updated_at = datetime.utcnow()
        self._save_permissions()

        return True

    def revoke_module_access(
        self,
        permission_id: str,
        module: DataModule
    ) -> bool:
        """Revoke module access from a permission."""
        permission = self._permissions.get(permission_id)
        if not permission:
            return False

        if module in permission.module_access:
            del permission.module_access[module]
            permission.updated_at = datetime.utcnow()
            self._save_permissions()
            return True

        return False

    def check_access(
        self,
        permission_id: str,
        module: DataModule,
        warehouse_name: str,
        operation: WarehouseOperation,
        schema: Optional[str] = None,
        table: Optional[str] = None
    ) -> bool:
        """
        Check if an operation is allowed.

        Args:
            permission_id: Permission to check
            module: Data module requesting access
            warehouse_name: Target warehouse
            operation: Operation to perform
            schema: Optional schema name
            table: Optional table name

        Returns:
            True if access is allowed
        """
        permission = self._permissions.get(permission_id)
        if not permission or not permission.is_valid():
            return False

        access = permission.get_module_access(module, warehouse_name)
        if not access:
            return False

        # Check operation
        if not access.can_perform(operation):
            return False

        # Check schema/table restrictions
        if schema and not access.can_access_schema(schema):
            return False

        if table and not access.can_access_table(table, schema):
            return False

        # Record usage
        permission.record_usage()
        self._save_permissions()

        return True

    def get_permission(self, permission_id: str) -> Optional[WarehousePermission]:
        """Get a permission by ID."""
        return self._permissions.get(permission_id)

    def get_permission_by_name(self, name: str) -> Optional[WarehousePermission]:
        """Get a permission by name."""
        for perm in self._permissions.values():
            if perm.name == name:
                return perm
        return None

    def list_permissions(self, active_only: bool = True) -> List[WarehousePermission]:
        """List all permissions."""
        permissions = list(self._permissions.values())
        if active_only:
            permissions = [p for p in permissions if p.is_valid()]
        return permissions

    def deactivate_permission(self, permission_id: str) -> bool:
        """Deactivate a permission."""
        permission = self._permissions.get(permission_id)
        if not permission:
            return False

        permission.is_active = False
        permission.updated_at = datetime.utcnow()
        self._save_permissions()
        return True

    def get_module_permissions(
        self,
        module: DataModule,
        warehouse_name: Optional[str] = None
    ) -> List[WarehousePermission]:
        """Get all permissions that grant access to a specific module."""
        results = []
        for permission in self._permissions.values():
            if not permission.is_valid():
                continue

            for mod, access in permission.module_access.items():
                if mod in (module, DataModule.ALL):
                    if warehouse_name is None or access.warehouse_name == warehouse_name or not access.warehouse_name:
                        results.append(permission)
                        break

        return results

    def create_default_permissions(self) -> Dict[str, WarehousePermission]:
        """
        Create default permissions for each module.

        Returns:
            Dictionary of created permissions by module name
        """
        default_configs = {
            DataModule.LINEAGE: {
                "access_level": WarehouseAccessLevel.READ_METADATA,
                "operations": WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_STATS,
                "description": "Lineage module: read metadata and schema information"
            },
            DataModule.DISCOVERY: {
                "access_level": WarehouseAccessLevel.READ_SAMPLE,
                "operations": WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_SAMPLE | WarehouseOperation.READ_STATS,
                "description": "Discovery module: read schema and sample data for indexing"
            },
            DataModule.ENRICHMENT: {
                "access_level": WarehouseAccessLevel.READ_SAMPLE,
                "operations": WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_SAMPLE | WarehouseOperation.READ_STATS,
                "description": "Enrichment module: read schema and samples for metadata enrichment"
            },
            DataModule.CLASSIFICATION: {
                "access_level": WarehouseAccessLevel.READ_SAMPLE,
                "operations": WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_SAMPLE,
                "description": "Classification module: read samples for data classification"
            },
            DataModule.QUALITY: {
                "access_level": WarehouseAccessLevel.READ_FULL,
                "operations": WarehouseOperation.READ_ALL | WarehouseOperation.EXECUTE_QUERY,
                "description": "Quality module: full read access for quality metrics"
            },
            DataModule.ASSET_VALUE: {
                "access_level": WarehouseAccessLevel.READ_METADATA,
                "operations": WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_STATS,
                "description": "Asset Value module: read metadata for value assessment"
            },
        }

        created = {}
        for module, config in default_configs.items():
            perm = self.create_permission(
                name=f"default_{module.value}",
                description=config["description"]
            )

            # Grant access to all warehouses
            self.grant_module_access(
                permission_id=perm.permission_id,
                module=module,
                warehouse_name="",  # Empty = all warehouses
                access_level=config["access_level"],
                operations=config["operations"]
            )

            created[module.value] = perm

        return created

    def _save_permissions(self) -> None:
        """Persist permissions to storage."""
        data = {}
        for perm_id, perm in self._permissions.items():
            module_access_data = {}
            for module, access in perm.module_access.items():
                module_access_data[module.value] = {
                    "module": access.module.value,
                    "warehouse_name": access.warehouse_name,
                    "access_level": access.access_level.value,
                    "operations": access.operations.value,
                    "allowed_schemas": list(access.allowed_schemas),
                    "allowed_tables": list(access.allowed_tables),
                    "max_sample_rows": access.max_sample_rows,
                    "max_query_rows": access.max_query_rows,
                    "query_timeout_seconds": access.query_timeout_seconds,
                    "enabled": access.enabled,
                    "created_at": access.created_at.isoformat(),
                    "expires_at": access.expires_at.isoformat() if access.expires_at else None,
                }

            data[perm_id] = {
                "permission_id": perm.permission_id,
                "name": perm.name,
                "description": perm.description,
                "warehouse_names": list(perm.warehouse_names),
                "module_access": module_access_data,
                "is_active": perm.is_active,
                "created_at": perm.created_at.isoformat(),
                "updated_at": perm.updated_at.isoformat() if perm.updated_at else None,
                "expires_at": perm.expires_at.isoformat() if perm.expires_at else None,
                "created_by": perm.created_by,
                "last_used": perm.last_used.isoformat() if perm.last_used else None,
                "usage_count": perm.usage_count,
            }

        permissions_file = self._storage_path / "permissions.json"
        with open(permissions_file, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(permissions_file, 0o600)

    def _load_permissions(self) -> None:
        """Load permissions from storage."""
        permissions_file = self._storage_path / "permissions.json"
        if not permissions_file.exists():
            return

        with open(permissions_file, "r") as f:
            data = json.load(f)

        for perm_id, perm_data in data.items():
            module_access = {}
            for mod_name, access_data in perm_data.get("module_access", {}).items():
                module = DataModule(access_data["module"])
                module_access[module] = ModuleAccess(
                    module=module,
                    warehouse_name=access_data["warehouse_name"],
                    access_level=WarehouseAccessLevel(access_data["access_level"]),
                    operations=WarehouseOperation(access_data["operations"]),
                    allowed_schemas=set(access_data.get("allowed_schemas", [])),
                    allowed_tables=set(access_data.get("allowed_tables", [])),
                    max_sample_rows=access_data.get("max_sample_rows", 10000),
                    max_query_rows=access_data.get("max_query_rows", 100000),
                    query_timeout_seconds=access_data.get("query_timeout_seconds", 300),
                    enabled=access_data.get("enabled", True),
                    created_at=datetime.fromisoformat(access_data["created_at"]) if access_data.get("created_at") else datetime.utcnow(),
                    expires_at=datetime.fromisoformat(access_data["expires_at"]) if access_data.get("expires_at") else None,
                )

            self._permissions[perm_id] = WarehousePermission(
                permission_id=perm_data["permission_id"],
                name=perm_data["name"],
                description=perm_data.get("description"),
                warehouse_names=set(perm_data.get("warehouse_names", [])),
                module_access=module_access,
                is_active=perm_data.get("is_active", True),
                created_at=datetime.fromisoformat(perm_data["created_at"]) if perm_data.get("created_at") else datetime.utcnow(),
                updated_at=datetime.fromisoformat(perm_data["updated_at"]) if perm_data.get("updated_at") else None,
                expires_at=datetime.fromisoformat(perm_data["expires_at"]) if perm_data.get("expires_at") else None,
                created_by=perm_data.get("created_by"),
                last_used=datetime.fromisoformat(perm_data["last_used"]) if perm_data.get("last_used") else None,
                usage_count=perm_data.get("usage_count", 0),
            )

    def export_audit_log(self) -> List[Dict[str, Any]]:
        """Export permission usage audit log."""
        audit_entries = []
        for perm in self._permissions.values():
            audit_entries.append({
                "permission_id": perm.permission_id,
                "name": perm.name,
                "created_at": perm.created_at.isoformat(),
                "created_by": perm.created_by,
                "last_used": perm.last_used.isoformat() if perm.last_used else None,
                "usage_count": perm.usage_count,
                "is_active": perm.is_active,
                "modules": [m.value for m in perm.module_access.keys()],
                "warehouses": list(perm.warehouse_names) if perm.warehouse_names else ["all"],
            })

        return sorted(audit_entries, key=lambda x: x["usage_count"], reverse=True)
