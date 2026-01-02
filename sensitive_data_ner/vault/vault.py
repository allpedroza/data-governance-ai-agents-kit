"""
Secure Vault Main Module

High-level interface for secure storage and retrieval of
sensitive data with encryption, access control, and audit logging.
"""

import secrets
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from .storage import VaultStorage, SQLiteVaultStorage, StoredMapping, StoredSession
from .key_manager import KeyManager, KeyManagerConfig, KeyRotationPolicy
from .access_control import AccessController, AccessLevel, AuthToken
from .audit import AuditLogger, AuditEventType


@dataclass
class VaultConfig:
    """Configuration for the secure vault."""
    # Storage settings
    storage_path: str = ".vault_data"
    use_postgresql: bool = False
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "vault"
    postgres_user: str = "vault_user"
    postgres_password: str = ""

    # Key management
    key_rotation_policy: KeyRotationPolicy = KeyRotationPolicy.MONTHLY
    master_key_env_var: str = "VAULT_MASTER_KEY"

    # Access control
    require_authentication: bool = True
    session_timeout_hours: int = 8

    # Audit settings
    audit_retention_days: int = 365
    audit_all_operations: bool = True


@dataclass
class VaultSession:
    """Represents an active vault session for a user request."""
    session_id: str
    original_text: str
    anonymized_text: str
    entities: List[Dict[str, Any]]
    risk_score: float
    created_at: datetime
    key_id: str
    is_stored: bool = False


class SecureVault:
    """
    Secure vault for storing and retrieving sensitive data mappings.

    Provides:
    - Encrypted storage of original messages
    - Session-based tracking for later decryption
    - Role-based access control
    - Comprehensive audit logging
    - Key rotation and management

    Usage:
        vault = SecureVault()
        vault.initialize("master_password")

        # Authenticate
        token = vault.authenticate("admin", "password")

        # Store anonymized data
        session = vault.store_session(
            token=token,
            original_text="My CPF is 123.456.789-09",
            anonymized_text="My CPF is [CPF]",
            entities=[...],
            risk_score=0.8
        )

        # Later, decrypt
        original = vault.decrypt_session(token, session.session_id)
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography package required. "
                "Install with: pip install cryptography"
            )

        self.config = config or VaultConfig()
        self._storage: Optional[VaultStorage] = None
        self._key_manager: Optional[KeyManager] = None
        self._access_controller: Optional[AccessController] = None
        self._audit_logger: Optional[AuditLogger] = None
        self._initialized = False

    def initialize(self, master_password: Optional[str] = None) -> str:
        """
        Initialize the vault with all components.

        Args:
            master_password: Master password for key derivation.
                           If None, uses VAULT_MASTER_KEY env var.

        Returns:
            Active encryption key ID
        """
        # Initialize storage
        if self.config.use_postgresql:
            from .storage import PostgreSQLVaultStorage
            self._storage = PostgreSQLVaultStorage(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
        else:
            self._storage = SQLiteVaultStorage(
                db_path=f"{self.config.storage_path}/vault.db"
            )

        self._storage.initialize()

        # Initialize key manager
        key_config = KeyManagerConfig(
            key_storage_path=f"{self.config.storage_path}/keys",
            rotation_policy=self.config.key_rotation_policy,
            master_key_env_var=self.config.master_key_env_var
        )
        self._key_manager = KeyManager(key_config)
        key_id = self._key_manager.initialize(master_password)

        # Initialize access controller
        self._access_controller = AccessController(
            storage_path=f"{self.config.storage_path}/access"
        )

        # Initialize audit logger
        self._audit_logger = AuditLogger(
            storage_path=f"{self.config.storage_path}/audit",
            retention_days=self.config.audit_retention_days
        )

        self._initialized = True

        # Log initialization
        self._audit_logger.log(
            AuditEventType.SYSTEM_STARTUP,
            details={"key_id": key_id}
        )

        return key_id

    def _ensure_initialized(self) -> None:
        """Ensure vault is initialized."""
        if not self._initialized:
            raise RuntimeError("Vault not initialized. Call initialize() first.")

    def _verify_access(
        self,
        token: str,
        required_level: AccessLevel,
        session_id: Optional[str] = None
    ) -> AuthToken:
        """Verify access token and level."""
        if not self.config.require_authentication:
            # Return a dummy token for non-authenticated mode
            return AuthToken(
                token="",
                user_id="system",
                access_level=AccessLevel.ADMIN,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow()
            )

        auth_token = self._access_controller.verify_token(token)
        if not auth_token:
            self._audit_logger.log(
                AuditEventType.SECURITY_ACCESS_DENIED,
                details={"reason": "Invalid token"}
            )
            raise PermissionError("Invalid or expired token")

        if not self._access_controller.check_access(
            token, required_level, session_id
        ):
            self._audit_logger.log(
                AuditEventType.SECURITY_ACCESS_DENIED,
                user_id=auth_token.user_id,
                details={
                    "reason": "Insufficient permissions",
                    "required_level": required_level.name
                }
            )
            raise PermissionError(f"Insufficient permissions. Required: {required_level.name}")

        return auth_token

    # User Management

    def create_user(
        self,
        username: str,
        password: str,
        access_level: AccessLevel,
        admin_token: Optional[str] = None
    ) -> str:
        """
        Create a new vault user.

        Args:
            username: Unique username
            password: User password
            access_level: Access level to grant
            admin_token: Admin token (required except for first user)

        Returns:
            User ID
        """
        self._ensure_initialized()

        user = self._access_controller.create_user(
            username, password, access_level, admin_token
        )

        self._audit_logger.log(
            AuditEventType.USER_CREATED,
            user_id=user.user_id,
            username=username,
            details={"access_level": access_level.name}
        )

        return user.user_id

    def authenticate(self, username: str, password: str) -> str:
        """
        Authenticate a user and get access token.

        Args:
            username: Username
            password: Password

        Returns:
            Access token string
        """
        self._ensure_initialized()

        try:
            token = self._access_controller.authenticate(username, password)

            if token:
                self._audit_logger.log(
                    AuditEventType.AUTH_LOGIN_SUCCESS,
                    user_id=token.user_id,
                    username=username
                )
                return token.token
            else:
                self._audit_logger.log(
                    AuditEventType.AUTH_LOGIN_FAILURE,
                    username=username,
                    details={"reason": "Invalid credentials"}
                )
                raise PermissionError("Invalid credentials")

        except PermissionError as e:
            self._audit_logger.log(
                AuditEventType.AUTH_LOGIN_FAILURE,
                username=username,
                details={"reason": str(e)}
            )
            raise

    def logout(self, token: str) -> None:
        """Logout and revoke token."""
        self._ensure_initialized()

        auth_token = self._access_controller.verify_token(token)
        if auth_token:
            self._access_controller.revoke_token(token)
            self._audit_logger.log(
                AuditEventType.AUTH_LOGOUT,
                user_id=auth_token.user_id
            )

    # Session Storage

    def store_session(
        self,
        token: str,
        original_text: str,
        anonymized_text: str,
        entities: List[Dict[str, Any]],
        risk_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VaultSession:
        """
        Store a session with encrypted original text.

        Args:
            token: Access token
            original_text: Original text to encrypt and store
            anonymized_text: Anonymized version of text
            entities: List of detected entities
            risk_score: Risk score for the session
            metadata: Additional metadata

        Returns:
            VaultSession with session details
        """
        self._ensure_initialized()
        auth_token = self._verify_access(token, AccessLevel.READ_ONLY)

        # Generate session ID
        session_id = f"sess_{secrets.token_hex(16)}"

        # Get encryption key
        key_id, key = self._key_manager.get_encryption_key()
        fernet = Fernet(key)

        # Encrypt original text
        encrypted_text = fernet.encrypt(original_text.encode())

        # Store session
        stored_session = StoredSession(
            session_id=session_id,
            original_message_encrypted=encrypted_text,
            anonymized_message=anonymized_text,
            entity_count=len(entities),
            risk_score=risk_score,
            key_id=key_id,
            created_at=datetime.utcnow(),
            accessed_at=None,
            access_count=0,
            metadata=metadata or {}
        )

        self._storage.store_session(stored_session)

        # Store individual entity mappings
        for i, entity in enumerate(entities):
            original_value = entity.get("value", "")
            encrypted_value = fernet.encrypt(original_value.encode())

            mapping = StoredMapping(
                mapping_id=f"{session_id}_map_{i}",
                session_id=session_id,
                original_encrypted=encrypted_value,
                anonymized_value=entity.get("anonymized", ""),
                entity_type=entity.get("entity_type", "unknown"),
                category=entity.get("category", "unknown"),
                position_start=entity.get("start", 0),
                position_end=entity.get("end", 0),
                key_id=key_id,
                created_at=datetime.utcnow(),
                metadata={
                    "confidence": entity.get("confidence", 0.0),
                    "pattern": entity.get("pattern", "")
                }
            )

            self._storage.store_mapping(mapping)

        # Audit log
        self._audit_logger.log(
            AuditEventType.DATA_STORED,
            user_id=auth_token.user_id,
            session_id=session_id,
            details={
                "entity_count": len(entities),
                "risk_score": risk_score,
                "text_length": len(original_text)
            }
        )

        return VaultSession(
            session_id=session_id,
            original_text=original_text,
            anonymized_text=anonymized_text,
            entities=entities,
            risk_score=risk_score,
            created_at=datetime.utcnow(),
            key_id=key_id,
            is_stored=True
        )

    def decrypt_session(
        self,
        token: str,
        session_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Decrypt and retrieve original session data.

        Args:
            token: Access token with DECRYPT level
            session_id: Session ID to decrypt

        Returns:
            Tuple of (original_text, list of original entities)
        """
        self._ensure_initialized()
        auth_token = self._verify_access(token, AccessLevel.DECRYPT, session_id)

        # Get session
        session = self._storage.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Get decryption key
        key = self._key_manager.get_key_by_id(session.key_id)
        if not key:
            raise RuntimeError(f"Decryption key not found: {session.key_id}")

        fernet = Fernet(key)

        # Decrypt original text
        original_text = fernet.decrypt(session.original_message_encrypted).decode()

        # Get and decrypt mappings
        mappings = self._storage.get_mappings_for_session(session_id)
        entities = []

        for mapping in mappings:
            # Get key for this mapping (might be different if rotated)
            map_key = self._key_manager.get_key_by_id(mapping.key_id)
            if map_key:
                map_fernet = Fernet(map_key)
                original_value = map_fernet.decrypt(mapping.original_encrypted).decode()

                entities.append({
                    "value": original_value,
                    "anonymized": mapping.anonymized_value,
                    "entity_type": mapping.entity_type,
                    "category": mapping.category,
                    "start": mapping.position_start,
                    "end": mapping.position_end,
                    "confidence": mapping.metadata.get("confidence", 0.0)
                })

        # Update access tracking
        self._storage.update_session_access(session_id)

        # Audit log
        self._audit_logger.log(
            AuditEventType.DATA_DECRYPTED,
            user_id=auth_token.user_id,
            session_id=session_id,
            details={"entity_count": len(entities)}
        )

        return original_text, entities

    def get_anonymized_session(
        self,
        token: str,
        session_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get anonymized session data (without decryption).

        Args:
            token: Access token with READ_ONLY level
            session_id: Session ID

        Returns:
            Tuple of (anonymized_text, anonymized entities)
        """
        self._ensure_initialized()
        auth_token = self._verify_access(token, AccessLevel.READ_ONLY, session_id)

        # Get session
        session = self._storage.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Get mappings (anonymized only)
        mappings = self._storage.get_mappings_for_session(session_id)
        entities = [
            {
                "anonymized": m.anonymized_value,
                "entity_type": m.entity_type,
                "category": m.category,
                "start": m.position_start,
                "end": m.position_end
            }
            for m in mappings
        ]

        # Audit log
        self._audit_logger.log(
            AuditEventType.SESSION_ACCESSED,
            user_id=auth_token.user_id,
            session_id=session_id
        )

        return session.anonymized_message, entities

    def delete_session(self, token: str, session_id: str) -> bool:
        """
        Delete a session and all its data.

        Args:
            token: Access token with ADMIN level
            session_id: Session to delete

        Returns:
            True if deleted
        """
        self._ensure_initialized()
        auth_token = self._verify_access(token, AccessLevel.ADMIN)

        deleted = self._storage.delete_session(session_id)

        if deleted:
            self._audit_logger.log(
                AuditEventType.SESSION_DELETED,
                user_id=auth_token.user_id,
                session_id=session_id
            )

        return deleted

    def list_sessions(
        self,
        token: str,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        List sessions (metadata only).

        Args:
            token: Access token
            limit: Max results
            offset: Pagination offset
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of session metadata
        """
        self._ensure_initialized()
        self._verify_access(token, AccessLevel.READ_ONLY)

        sessions = self._storage.list_sessions(limit, offset, start_date, end_date)

        return [
            {
                "session_id": s.session_id,
                "entity_count": s.entity_count,
                "risk_score": s.risk_score,
                "created_at": s.created_at.isoformat(),
                "accessed_at": s.accessed_at.isoformat() if s.accessed_at else None,
                "access_count": s.access_count,
                "anonymized_preview": s.anonymized_message[:100] + "..."
                    if len(s.anonymized_message) > 100 else s.anonymized_message
            }
            for s in sessions
        ]

    # Key Management

    def rotate_key(self, token: str) -> str:
        """
        Rotate encryption key.

        Args:
            token: Access token with ADMIN level

        Returns:
            New key ID
        """
        self._ensure_initialized()
        auth_token = self._verify_access(token, AccessLevel.ADMIN)

        old_key_id = self._key_manager._active_key_id
        new_key_id = self._key_manager.rotate_key()

        self._audit_logger.log(
            AuditEventType.KEY_ROTATED,
            user_id=auth_token.user_id,
            details={
                "old_key_id": old_key_id,
                "new_key_id": new_key_id
            }
        )

        return new_key_id

    # Statistics and Audit

    def get_stats(self, token: str) -> Dict[str, Any]:
        """Get vault statistics."""
        self._ensure_initialized()
        self._verify_access(token, AccessLevel.READ_ONLY)

        storage_stats = self._storage.get_stats()
        key_metadata = self._key_manager.list_keys()

        return {
            **storage_stats,
            "active_key_id": self._key_manager._active_key_id,
            "key_count": len(key_metadata),
            "key_versions": [
                {
                    "key_id": k.key_id,
                    "version": k.version,
                    "is_active": k.is_active,
                    "created_at": k.created_at.isoformat(),
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None
                }
                for k in key_metadata
            ]
        }

    def get_audit_logs(
        self,
        token: str,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs.

        Args:
            token: Access token with ADMIN level
            event_types: Filter by event types
            user_id: Filter by user
            session_id: Filter by session
            start_date: Start of date range
            end_date: End of date range
            limit: Max results

        Returns:
            List of audit events
        """
        self._ensure_initialized()
        self._verify_access(token, AccessLevel.ADMIN)

        events = self._audit_logger.query(
            event_types=event_types,
            user_id=user_id,
            session_id=session_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "timestamp": e.timestamp.isoformat(),
                "user_id": e.user_id,
                "username": e.username,
                "session_id": e.session_id,
                "details": e.details
            }
            for e in events
        ]

    def verify_audit_integrity(self, token: str) -> Tuple[bool, List[str]]:
        """
        Verify integrity of audit logs.

        Args:
            token: Access token with SUPER_ADMIN level

        Returns:
            Tuple of (is_valid, list of errors)
        """
        self._ensure_initialized()
        self._verify_access(token, AccessLevel.SUPER_ADMIN)

        return self._audit_logger.verify_chain_integrity()

    def close(self) -> None:
        """Close vault and release resources."""
        if self._storage:
            self._storage.close()

        if self._audit_logger:
            self._audit_logger.log(AuditEventType.SYSTEM_SHUTDOWN)

        self._initialized = False
