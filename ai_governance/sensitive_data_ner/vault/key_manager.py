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
Key Management Module

Handles encryption keys with secure generation, storage, and rotation.
"""

import os
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from enum import Enum
import base64

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class KeyRotationPolicy(Enum):
    """Key rotation frequency policies."""
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class KeyMetadata:
    """Metadata for an encryption key."""
    key_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    version: int
    rotated_from: Optional[str] = None


@dataclass
class KeyManagerConfig:
    """Configuration for key manager."""
    key_storage_path: str = ".vault_keys"
    rotation_policy: KeyRotationPolicy = KeyRotationPolicy.MONTHLY
    key_derivation_iterations: int = 480000  # OWASP recommended
    master_key_env_var: str = "VAULT_MASTER_KEY"
    auto_rotate: bool = True
    max_key_versions: int = 5  # Keep for decryption of old data


class KeyManager:
    """
    Secure key management with derivation, rotation, and versioning.

    Uses PBKDF2 for key derivation from master password/key.
    Supports automatic key rotation based on policy.
    """

    def __init__(self, config: Optional[KeyManagerConfig] = None):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography package required. "
                "Install with: pip install cryptography"
            )

        self.config = config or KeyManagerConfig()
        self._keys: Dict[str, Tuple[bytes, KeyMetadata]] = {}
        self._active_key_id: Optional[str] = None
        self._master_key: Optional[bytes] = None

        # Ensure storage directory exists with secure permissions
        self._storage_path = Path(self.config.key_storage_path)
        self._ensure_secure_storage()

    def _ensure_secure_storage(self) -> None:
        """Create storage directory with secure permissions."""
        if not self._storage_path.exists():
            self._storage_path.mkdir(parents=True, mode=0o700)
        else:
            # Verify permissions are secure
            current_mode = self._storage_path.stat().st_mode & 0o777
            if current_mode != 0o700:
                os.chmod(self._storage_path, 0o700)

    def initialize(self, master_password: Optional[str] = None) -> str:
        """
        Initialize key manager with master key.

        Args:
            master_password: Master password for key derivation.
                           If None, checks environment variable.

        Returns:
            Active key ID
        """
        # Get or generate master key
        if master_password:
            self._master_key = self._derive_master_key(master_password)
        else:
            env_key = os.environ.get(self.config.master_key_env_var)
            if env_key:
                self._master_key = base64.b64decode(env_key)
            else:
                # Generate new master key
                self._master_key = Fernet.generate_key()
                print(
                    f"WARNING: Generated new master key. "
                    f"Set {self.config.master_key_env_var} environment variable "
                    f"to persist: {base64.b64encode(self._master_key).decode()}"
                )

        # Load existing keys or create first key
        self._load_keys()

        if not self._active_key_id:
            self._active_key_id = self._generate_new_key()

        # Check if rotation is needed
        if self.config.auto_rotate:
            self._check_rotation()

        return self._active_key_id

    def _derive_master_key(self, password: str) -> bytes:
        """Derive master key from password using PBKDF2."""
        # Use fixed salt stored securely (or derive from password)
        salt_path = self._storage_path / ".salt"

        if salt_path.exists():
            with open(salt_path, "rb") as f:
                salt = f.read()
        else:
            salt = secrets.token_bytes(32)
            with open(salt_path, "wb") as f:
                f.write(salt)
            os.chmod(salt_path, 0o600)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.key_derivation_iterations,
            backend=default_backend()
        )

        derived = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(derived)

    def _generate_new_key(self, rotated_from: Optional[str] = None) -> str:
        """Generate a new encryption key."""
        key_id = f"key_{secrets.token_hex(8)}"
        key = Fernet.generate_key()

        # Encrypt key with master key for storage
        master_fernet = Fernet(self._master_key)
        encrypted_key = master_fernet.encrypt(key)

        # Calculate expiry based on rotation policy
        expires_at = self._calculate_expiry()

        # Determine version
        version = 1
        if rotated_from and rotated_from in self._keys:
            version = self._keys[rotated_from][1].version + 1

        metadata = KeyMetadata(
            key_id=key_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_active=True,
            version=version,
            rotated_from=rotated_from
        )

        # Store key
        self._keys[key_id] = (key, metadata)

        # Deactivate old active key
        if self._active_key_id and self._active_key_id in self._keys:
            old_key, old_meta = self._keys[self._active_key_id]
            old_meta.is_active = False
            self._keys[self._active_key_id] = (old_key, old_meta)

        self._active_key_id = key_id

        # Persist to disk
        self._save_keys()

        # Cleanup old versions
        self._cleanup_old_keys()

        return key_id

    def _calculate_expiry(self) -> Optional[datetime]:
        """Calculate key expiry based on rotation policy."""
        now = datetime.utcnow()

        if self.config.rotation_policy == KeyRotationPolicy.NEVER:
            return None
        elif self.config.rotation_policy == KeyRotationPolicy.DAILY:
            return now + timedelta(days=1)
        elif self.config.rotation_policy == KeyRotationPolicy.WEEKLY:
            return now + timedelta(weeks=1)
        elif self.config.rotation_policy == KeyRotationPolicy.MONTHLY:
            return now + timedelta(days=30)
        elif self.config.rotation_policy == KeyRotationPolicy.QUARTERLY:
            return now + timedelta(days=90)

        return None

    def _check_rotation(self) -> None:
        """Check if key rotation is needed and rotate if necessary."""
        if not self._active_key_id:
            return

        _, metadata = self._keys.get(self._active_key_id, (None, None))
        if not metadata or not metadata.expires_at:
            return

        if datetime.utcnow() >= metadata.expires_at:
            self.rotate_key()

    def rotate_key(self) -> str:
        """
        Rotate to a new encryption key.

        Returns:
            New key ID
        """
        old_key_id = self._active_key_id
        new_key_id = self._generate_new_key(rotated_from=old_key_id)
        return new_key_id

    def _cleanup_old_keys(self) -> None:
        """Remove old key versions beyond max_key_versions."""
        if len(self._keys) <= self.config.max_key_versions:
            return

        # Sort by version, keep newest
        sorted_keys = sorted(
            self._keys.items(),
            key=lambda x: x[1][1].version,
            reverse=True
        )

        keys_to_remove = sorted_keys[self.config.max_key_versions:]
        for key_id, _ in keys_to_remove:
            del self._keys[key_id]

        self._save_keys()

    def _save_keys(self) -> None:
        """Persist keys to secure storage."""
        if not self._master_key:
            raise RuntimeError("Key manager not initialized")

        master_fernet = Fernet(self._master_key)

        keys_data = {}
        for key_id, (key, metadata) in self._keys.items():
            encrypted_key = master_fernet.encrypt(key)
            keys_data[key_id] = {
                "encrypted_key": base64.b64encode(encrypted_key).decode(),
                "created_at": metadata.created_at.isoformat(),
                "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                "is_active": metadata.is_active,
                "version": metadata.version,
                "rotated_from": metadata.rotated_from
            }

        keys_file = self._storage_path / "keys.json"
        with open(keys_file, "w") as f:
            json.dump({
                "active_key_id": self._active_key_id,
                "keys": keys_data
            }, f, indent=2)

        os.chmod(keys_file, 0o600)

    def _load_keys(self) -> None:
        """Load keys from secure storage."""
        if not self._master_key:
            raise RuntimeError("Key manager not initialized")

        keys_file = self._storage_path / "keys.json"
        if not keys_file.exists():
            return

        with open(keys_file, "r") as f:
            data = json.load(f)

        master_fernet = Fernet(self._master_key)

        self._active_key_id = data.get("active_key_id")

        for key_id, key_data in data.get("keys", {}).items():
            encrypted_key = base64.b64decode(key_data["encrypted_key"])
            try:
                key = master_fernet.decrypt(encrypted_key)
            except Exception:
                # Key cannot be decrypted with current master key
                continue

            metadata = KeyMetadata(
                key_id=key_id,
                created_at=datetime.fromisoformat(key_data["created_at"]),
                expires_at=datetime.fromisoformat(key_data["expires_at"])
                    if key_data["expires_at"] else None,
                is_active=key_data["is_active"],
                version=key_data["version"],
                rotated_from=key_data.get("rotated_from")
            )

            self._keys[key_id] = (key, metadata)

    def get_encryption_key(self) -> Tuple[str, bytes]:
        """
        Get the active encryption key.

        Returns:
            Tuple of (key_id, key_bytes)
        """
        if not self._active_key_id:
            raise RuntimeError("Key manager not initialized")

        if self._active_key_id not in self._keys:
            raise RuntimeError("Active key not found")

        key, _ = self._keys[self._active_key_id]
        return self._active_key_id, key

    def get_key_by_id(self, key_id: str) -> Optional[bytes]:
        """
        Get a specific key by ID (for decryption of old data).

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        if key_id in self._keys:
            return self._keys[key_id][0]
        return None

    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get metadata for a specific key."""
        if key_id in self._keys:
            return self._keys[key_id][1]
        return None

    def list_keys(self) -> List[KeyMetadata]:
        """List all key metadata."""
        return [meta for _, meta in self._keys.values()]

    def destroy_key(self, key_id: str) -> bool:
        """
        Securely destroy a key.

        WARNING: Data encrypted with this key will be unrecoverable.

        Args:
            key_id: Key to destroy

        Returns:
            True if destroyed, False if not found
        """
        if key_id not in self._keys:
            return False

        if key_id == self._active_key_id:
            raise ValueError("Cannot destroy active key. Rotate first.")

        del self._keys[key_id]
        self._save_keys()
        return True
