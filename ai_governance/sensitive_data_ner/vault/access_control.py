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
Access Control Module

Role-based access control for the secure vault with:
- Multiple access levels
- User authentication
- Permission verification
"""

import hashlib
import secrets
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set
from pathlib import Path
import os


class AccessLevel(IntEnum):
    """
    Access levels for vault operations.

    Higher values = more permissions.
    """
    NONE = 0
    READ_ONLY = 10       # Can view anonymized data only
    DECRYPT = 20         # Can decrypt specific sessions
    FULL_DECRYPT = 30    # Can decrypt all data
    ADMIN = 40           # Full access including key management
    SUPER_ADMIN = 50     # Can manage users and audit logs


@dataclass
class VaultUser:
    """Represents a vault user."""
    user_id: str
    username: str
    access_level: AccessLevel
    password_hash: str
    salt: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    allowed_sessions: Set[str] = field(default_factory=set)  # Empty = all allowed
    mfa_secret: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class AuthToken:
    """Authentication token for vault access."""
    token: str
    user_id: str
    access_level: AccessLevel
    created_at: datetime
    expires_at: datetime
    session_id: Optional[str] = None  # If limited to specific session


class AccessController:
    """
    Manages access control for the secure vault.

    Features:
    - User management with password hashing
    - Token-based authentication
    - Access level verification
    - Account lockout protection
    """

    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=30)
    TOKEN_DURATION = timedelta(hours=8)

    def __init__(self, storage_path: str = ".vault_access"):
        self._storage_path = Path(storage_path)
        self._ensure_storage()
        self._users: Dict[str, VaultUser] = {}
        self._tokens: Dict[str, AuthToken] = {}
        self._load_users()

    def _ensure_storage(self) -> None:
        """Create storage with secure permissions."""
        if not self._storage_path.exists():
            self._storage_path.mkdir(parents=True, mode=0o700)

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations=480000
        ).hex()

    def create_user(
        self,
        username: str,
        password: str,
        access_level: AccessLevel,
        creator_token: Optional[str] = None
    ) -> VaultUser:
        """
        Create a new vault user.

        Args:
            username: Unique username
            password: User password
            access_level: Access level to grant
            creator_token: Token of user creating this user

        Returns:
            Created VaultUser
        """
        # Verify creator has permission
        if creator_token:
            creator = self._verify_token_level(creator_token, AccessLevel.SUPER_ADMIN)
            if not creator:
                raise PermissionError("Insufficient permissions to create users")

        # Check if first user (bootstrap)
        if not self._users and not creator_token:
            # Allow first user creation as super admin
            access_level = AccessLevel.SUPER_ADMIN

        # Validate username
        if any(u.username == username for u in self._users.values()):
            raise ValueError(f"Username '{username}' already exists")

        # Generate user ID and salt
        user_id = f"user_{secrets.token_hex(8)}"
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)

        user = VaultUser(
            user_id=user_id,
            username=username,
            access_level=access_level,
            password_hash=password_hash,
            salt=salt
        )

        self._users[user_id] = user
        self._save_users()

        return user

    def authenticate(self, username: str, password: str) -> Optional[AuthToken]:
        """
        Authenticate user and return access token.

        Args:
            username: Username
            password: Password

        Returns:
            AuthToken if successful, None if failed
        """
        # Find user
        user = None
        for u in self._users.values():
            if u.username == username:
                user = u
                break

        if not user:
            return None

        # Check if locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            raise PermissionError(
                f"Account locked until {user.locked_until.isoformat()}"
            )

        # Check if active
        if not user.is_active:
            raise PermissionError("Account is disabled")

        # Verify password
        password_hash = self._hash_password(password, user.salt)

        if password_hash != user.password_hash:
            # Increment failed attempts
            user.failed_attempts += 1

            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.utcnow() + self.LOCKOUT_DURATION

            self._save_users()
            return None

        # Reset failed attempts
        user.failed_attempts = 0
        user.last_login = datetime.utcnow()
        self._save_users()

        # Generate token
        token = AuthToken(
            token=secrets.token_urlsafe(64),
            user_id=user.user_id,
            access_level=user.access_level,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.TOKEN_DURATION
        )

        self._tokens[token.token] = token
        return token

    def verify_token(self, token_str: str) -> Optional[AuthToken]:
        """
        Verify an access token.

        Args:
            token_str: Token string

        Returns:
            AuthToken if valid, None otherwise
        """
        token = self._tokens.get(token_str)

        if not token:
            return None

        if datetime.utcnow() > token.expires_at:
            del self._tokens[token_str]
            return None

        return token

    def _verify_token_level(
        self,
        token_str: str,
        required_level: AccessLevel
    ) -> Optional[AuthToken]:
        """Verify token has required access level."""
        token = self.verify_token(token_str)

        if not token:
            return None

        if token.access_level < required_level:
            return None

        return token

    def check_access(
        self,
        token_str: str,
        required_level: AccessLevel,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Check if token has access for operation.

        Args:
            token_str: Access token
            required_level: Required access level
            session_id: Optional session ID for session-specific access

        Returns:
            True if access granted
        """
        token = self._verify_token_level(token_str, required_level)

        if not token:
            return False

        # Check session restriction
        if session_id and token.session_id:
            if token.session_id != session_id:
                return False

        # Check user's allowed sessions
        user = self._users.get(token.user_id)
        if user and user.allowed_sessions:
            if session_id and session_id not in user.allowed_sessions:
                return False

        return True

    def revoke_token(self, token_str: str) -> bool:
        """Revoke an access token."""
        if token_str in self._tokens:
            del self._tokens[token_str]
            return True
        return False

    def deactivate_user(
        self,
        user_id: str,
        admin_token: str
    ) -> bool:
        """Deactivate a user account."""
        if not self._verify_token_level(admin_token, AccessLevel.SUPER_ADMIN):
            raise PermissionError("Insufficient permissions")

        if user_id in self._users:
            self._users[user_id].is_active = False
            self._save_users()

            # Revoke all tokens for this user
            tokens_to_revoke = [
                t for t, token in self._tokens.items()
                if token.user_id == user_id
            ]
            for t in tokens_to_revoke:
                del self._tokens[t]

            return True
        return False

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """Change user password."""
        user = self._users.get(user_id)
        if not user:
            return False

        # Verify old password
        old_hash = self._hash_password(old_password, user.salt)
        if old_hash != user.password_hash:
            return False

        # Set new password
        new_salt = secrets.token_hex(32)
        new_hash = self._hash_password(new_password, new_salt)

        user.salt = new_salt
        user.password_hash = new_hash
        self._save_users()

        # Revoke all existing tokens
        tokens_to_revoke = [
            t for t, token in self._tokens.items()
            if token.user_id == user_id
        ]
        for t in tokens_to_revoke:
            del self._tokens[t]

        return True

    def grant_session_access(
        self,
        user_id: str,
        session_id: str,
        admin_token: str
    ) -> bool:
        """Grant a user access to a specific session."""
        if not self._verify_token_level(admin_token, AccessLevel.ADMIN):
            raise PermissionError("Insufficient permissions")

        user = self._users.get(user_id)
        if not user:
            return False

        user.allowed_sessions.add(session_id)
        self._save_users()
        return True

    def list_users(self, admin_token: str) -> List[VaultUser]:
        """List all users (admin only)."""
        if not self._verify_token_level(admin_token, AccessLevel.ADMIN):
            raise PermissionError("Insufficient permissions")

        return list(self._users.values())

    def get_user_by_token(self, token_str: str) -> Optional[VaultUser]:
        """Get user from token."""
        token = self.verify_token(token_str)
        if not token:
            return None
        return self._users.get(token.user_id)

    def _save_users(self) -> None:
        """Persist users to storage."""
        users_data = {}
        for user_id, user in self._users.items():
            users_data[user_id] = {
                "user_id": user.user_id,
                "username": user.username,
                "access_level": user.access_level.value,
                "password_hash": user.password_hash,
                "salt": user.salt,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active,
                "allowed_sessions": list(user.allowed_sessions),
                "mfa_secret": user.mfa_secret,
                "failed_attempts": user.failed_attempts,
                "locked_until": user.locked_until.isoformat() if user.locked_until else None
            }

        users_file = self._storage_path / "users.json"
        with open(users_file, "w") as f:
            json.dump(users_data, f, indent=2)
        os.chmod(users_file, 0o600)

    def _load_users(self) -> None:
        """Load users from storage."""
        users_file = self._storage_path / "users.json"
        if not users_file.exists():
            return

        with open(users_file, "r") as f:
            users_data = json.load(f)

        for user_id, data in users_data.items():
            self._users[user_id] = VaultUser(
                user_id=data["user_id"],
                username=data["username"],
                access_level=AccessLevel(data["access_level"]),
                password_hash=data["password_hash"],
                salt=data["salt"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_login=datetime.fromisoformat(data["last_login"])
                    if data.get("last_login") else None,
                is_active=data.get("is_active", True),
                allowed_sessions=set(data.get("allowed_sessions", [])),
                mfa_secret=data.get("mfa_secret"),
                failed_attempts=data.get("failed_attempts", 0),
                locked_until=datetime.fromisoformat(data["locked_until"])
                    if data.get("locked_until") else None
            )
