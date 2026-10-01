"""
API key loading for SimCheck's network features (page quality, LLM probing).

Keys live OUTSIDE the repo in ~/.config/simcheck/.env (override with the
SIMCHECK_ENV_FILE environment variable). Shell environment variables take
precedence over the file. The core CCS flow never calls this module, so the
app keeps working with no keys configured.

Safety rules enforced here:
- Refuse to read a key file that is group/world readable (must be chmod 600).
- Refuse to read a key file located inside the repository.
- Never expose raw key values via repr/str; use mask_key() for display.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


DEFAULT_ENV_FILE = Path.home() / ".config" / "simcheck" / ".env"
REPO_ROOT = Path(__file__).resolve().parent.parent

KEY_NAMES = ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY")


class ConfigError(Exception):
    """Raised when the key file is unsafe or unreadable."""


@dataclass(frozen=True)
class ApiKeys:
    """
    Loaded API keys. Values are excluded from repr so they never leak into
    logs, tracebacks, or the Streamlit debug panel.
    """
    openrouter: Optional[str] = field(default=None, repr=False)
    typesafe: Optional[str] = field(default=None, repr=False)

    @property
    def has_openrouter(self) -> bool:
        """True if an OpenRouter key is configured."""
        return bool(self.openrouter)

    @property
    def has_typesafe(self) -> bool:
        """True if a TypeSafe (Jev) key is configured."""
        return bool(self.typesafe)

    def status(self) -> dict:
        """Masked, display-safe summary of which keys are configured."""
        return {
            "OPENROUTER_API_KEY": mask_key(self.openrouter),
            "TYPESAFE_API_KEY": mask_key(self.typesafe),
        }


def mask_key(value: Optional[str]) -> str:
    """
    Return a display-safe form of a key: last 4 chars only.

    Args:
        value: Raw key or None

    Returns:
        "not set", "set", or "…abcd"
    """
    if not value:
        return "not set"
    if len(value) < 12:
        return "set"
    return f"…{value[-4:]}"


def _resolve_env_file() -> Path:
    """Key file path: SIMCHECK_ENV_FILE if set, else the default location."""
    override = os.environ.get("SIMCHECK_ENV_FILE")
    return Path(override).expanduser() if override else DEFAULT_ENV_FILE


def _check_file_safety(path: Path) -> None:
    """Raise ConfigError if the key file is inside the repo or too permissive."""
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ConfigError(
            f"Key file {path} is inside the repository. "
            f"Move it to {DEFAULT_ENV_FILE}."
        )
    mode = resolved.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(
            f"Key file {path} is readable by other users. Run: chmod 600 {path}"
        )


def load_api_keys(env_file: Optional[Path] = None) -> ApiKeys:
    """
    Load API keys from the shell environment and the external key file.

    Shell environment variables win over file values. A missing key file is
    not an error (keys are optional); an unsafe one is.

    Args:
        env_file: Explicit key file path (tests); defaults to _resolve_env_file()

    Returns:
        ApiKeys with any configured values

    Raises:
        ConfigError: If the key file is inside the repo or not chmod 600
    """
    return ApiKeys(
        openrouter=load_setting("OPENROUTER_API_KEY", env_file),
        typesafe=load_setting("TYPESAFE_API_KEY", env_file),
    )


def load_setting(name: str, env_file: Optional[Path] = None) -> Optional[str]:
    """
    Read one setting: shell environment first, then the external key file.

    Hosted deployments (Streamlit Cloud, Hugging Face Spaces) expose their
    secrets as environment variables, so the same code works there with no
    key file.

    Args:
        name: Variable name
        env_file: Explicit key file path (tests); defaults to _resolve_env_file()

    Returns:
        Stripped value, or None if unset/blank

    Raises:
        ConfigError: If the key file is inside the repo or not chmod 600
    """
    value = os.environ.get(name)
    if not (value and value.strip()):
        path = env_file if env_file is not None else _resolve_env_file()
        if path.exists():
            _check_file_safety(path)
            value = dotenv_values(path).get(name)
    return value.strip() if value and value.strip() else None
