"""Centralised authentication and role-based access control for Streamlit UIs.

Usage in ``app.py``::

    from utils.auth_middleware import require_auth, has_role

    require_auth()            # blocks rendering until login succeeds
    if has_role("admin"):
        render_admin_tab()

Credentials come from ``st.secrets`` (recommended) or environment
variables as a fallback.  The expected shape is::

    # .streamlit/secrets.toml
    [auth]
    enabled = true
    users = [
        { username = "admin",        password_hash = "sha256:...", roles = ["admin", "data_steward"] },
        { username = "data_steward", password_hash = "sha256:...", roles = ["data_steward"] },
    ]

When ``auth.enabled`` is ``false`` *or* no ``[auth]`` section is found,
the middleware lets everyone through (useful for local dev).

Password hashes are generated with :func:`generate_password_hash`.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Dict, List, Optional, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def generate_password_hash(password: str) -> str:
    """Create a ``sha256:<hex>`` hash suitable for ``secrets.toml``.

    >>> generate_password_hash("s3cret")  # doctest: +SKIP
    'sha256:2bb80d537b...'
    """
    digest = hashlib.sha256(password.encode()).hexdigest()
    return f"sha256:{digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time comparison of *password* against *stored_hash*."""
    if not stored_hash.startswith("sha256:"):
        return False
    expected = stored_hash.split(":", 1)[1]
    candidate = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(expected, candidate)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def _load_auth_config() -> Dict:
    """Return the ``[auth]`` config from secrets or env vars."""
    # Try st.secrets first
    try:
        auth_section = st.secrets.get("auth", {})
        if auth_section:
            return dict(auth_section)
    except Exception:
        pass

    # Fallback to env-based single-user mode
    env_user = os.environ.get("AUTH_USERNAME", "")
    env_hash = os.environ.get("AUTH_PASSWORD_HASH", "")
    env_enabled = os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")

    if env_user and env_hash and env_enabled:
        return {
            "enabled": True,
            "users": [
                {"username": env_user, "password_hash": env_hash, "roles": ["admin"]},
            ],
        }

    return {"enabled": False}


def _get_user_db(config: Dict) -> Dict[str, Dict]:
    """Build a username → {password_hash, roles} lookup."""
    db: Dict[str, Dict] = {}
    for entry in config.get("users", []):
        username = entry.get("username", "")
        if username:
            db[username] = {
                "password_hash": entry.get("password_hash", ""),
                "roles": list(entry.get("roles", [])),
            }
    return db


# ---------------------------------------------------------------------------
# Session-level state
# ---------------------------------------------------------------------------

_SESSION_KEY_AUTHED = "_auth_authenticated"
_SESSION_KEY_USER = "_auth_username"
_SESSION_KEY_ROLES = "_auth_roles"


def _init_state() -> None:
    if _SESSION_KEY_AUTHED not in st.session_state:
        st.session_state[_SESSION_KEY_AUTHED] = False
        st.session_state[_SESSION_KEY_USER] = ""
        st.session_state[_SESSION_KEY_ROLES] = []


# ---------------------------------------------------------------------------
# Login UI
# ---------------------------------------------------------------------------

def _render_login(user_db: Dict[str, Dict]) -> None:
    """Show a minimal login form and validate credentials."""
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 400px;
            margin: 5rem auto;
            padding: 2rem;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 10px 40px rgba(15, 23, 42, 0.08);
        }
        .login-title {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .login-subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<p class='login-title'>🔐 Data Governance</p>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>Faça login para acessar os agentes</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Usuário", key="login_username_input")
        password = st.text_input("Senha", type="password", key="login_password_input")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

    if submitted:
        user = user_db.get(username)
        if user and verify_password(password, user["password_hash"]):
            st.session_state[_SESSION_KEY_AUTHED] = True
            st.session_state[_SESSION_KEY_USER] = username
            st.session_state[_SESSION_KEY_ROLES] = user["roles"]
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def require_auth() -> None:
    """Gate the Streamlit app behind authentication.

    If auth is disabled (no config / ``enabled = false``), returns immediately.
    Otherwise renders a login form and calls ``st.stop()`` until the user
    authenticates successfully.
    """
    _init_state()

    config = _load_auth_config()
    if not config.get("enabled"):
        # Auth disabled — grant full access
        st.session_state[_SESSION_KEY_AUTHED] = True
        st.session_state[_SESSION_KEY_ROLES] = ["admin"]
        return

    if st.session_state[_SESSION_KEY_AUTHED]:
        return  # already logged in

    user_db = _get_user_db(config)
    if not user_db:
        st.warning("Autenticação habilitada mas nenhum usuário configurado.")
        st.stop()

    _render_login(user_db)
    st.stop()


def logout() -> None:
    """Clear the current session credentials."""
    _init_state()
    st.session_state[_SESSION_KEY_AUTHED] = False
    st.session_state[_SESSION_KEY_USER] = ""
    st.session_state[_SESSION_KEY_ROLES] = []


def current_user() -> str:
    """Return the username of the logged-in user (empty if unauthenticated)."""
    _init_state()
    return st.session_state.get(_SESSION_KEY_USER, "")


def current_roles() -> List[str]:
    """Return the list of roles for the current user."""
    _init_state()
    return list(st.session_state.get(_SESSION_KEY_ROLES, []))


def has_role(role: str) -> bool:
    """Check if the current user has the given *role*."""
    return role in current_roles()


def require_role(role: str, *, message: str = "") -> None:
    """Stop rendering if the current user lacks *role*.

    Useful inside a tab callback to restrict a panel::

        require_role("admin", message="Apenas administradores podem acessar.")
    """
    if not has_role(role):
        st.warning(message or f"Acesso restrito ao papel '{role}'.")
        st.stop()
