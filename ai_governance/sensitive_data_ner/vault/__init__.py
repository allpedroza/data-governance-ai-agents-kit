"""
Secure Vault Module

Provides encrypted storage for sensitive data mappings with:
- Strong encryption (AES-256/Fernet)
- Key management with rotation support
- Role-based access control
- Comprehensive audit logging
- Session-based message tracking
"""

from .storage import (
    VaultStorage,
    SQLiteVaultStorage,
    PostgreSQLVaultStorage,
)
from .key_manager import (
    KeyManager,
    KeyRotationPolicy,
)
from .access_control import (
    AccessLevel,
    VaultUser,
    AccessController,
)
from .audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
)
from .vault import (
    SecureVault,
    VaultSession,
    VaultConfig,
    StoredMapping,
    RetentionPolicy,
)

__all__ = [
    # Storage
    "VaultStorage",
    "SQLiteVaultStorage",
    "PostgreSQLVaultStorage",
    # Key Management
    "KeyManager",
    "KeyRotationPolicy",
    # Access Control
    "AccessLevel",
    "VaultUser",
    "AccessController",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    # Main Vault
    "SecureVault",
    "VaultSession",
    "VaultConfig",
    "StoredMapping",
    "RetentionPolicy",
]
