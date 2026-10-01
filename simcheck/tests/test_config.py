"""Tests for API key loading and safety checks (simcheck.config)."""

import pytest

from simcheck import config
from simcheck.config import ApiKeys, ConfigError, load_api_keys, mask_key


FAKE_OR = "sk-or-v1-0000000000000000wxyz"
FAKE_TS = "ts-000000000000abcd"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate tests from any real keys in the shell environment."""
    for name in config.KEY_NAMES + ("SIMCHECK_ENV_FILE",):
        monkeypatch.delenv(name, raising=False)


def _write_key_file(path, mode=0o600):
    path.write_text(f"OPENROUTER_API_KEY={FAKE_OR}\nTYPESAFE_API_KEY={FAKE_TS}\n")
    path.chmod(mode)
    return path


class TestLoadApiKeys:
    def test_missing_file_returns_empty_keys(self, tmp_path):
        keys = load_api_keys(tmp_path / "nope.env")
        assert not keys.has_openrouter
        assert not keys.has_typesafe

    def test_reads_safe_file(self, tmp_path):
        keys = load_api_keys(_write_key_file(tmp_path / ".env"))
        assert keys.openrouter == FAKE_OR
        assert keys.typesafe == FAKE_TS

    def test_blank_values_treated_as_unset(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("OPENROUTER_API_KEY=\nTYPESAFE_API_KEY=   \n")
        path.chmod(0o600)
        keys = load_api_keys(path)
        assert not keys.has_openrouter
        assert not keys.has_typesafe

    def test_shell_env_overrides_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell-key-1234")
        keys = load_api_keys(_write_key_file(tmp_path / ".env"))
        assert keys.openrouter == "from-shell-key-1234"

    def test_env_file_override_variable(self, tmp_path, monkeypatch):
        path = _write_key_file(tmp_path / "custom.env")
        monkeypatch.setenv("SIMCHECK_ENV_FILE", str(path))
        assert load_api_keys().typesafe == FAKE_TS

    @pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
    def test_rejects_permissive_file(self, tmp_path, mode):
        path = _write_key_file(tmp_path / ".env", mode=mode)
        with pytest.raises(ConfigError, match="chmod 600"):
            load_api_keys(path)

    def test_rejects_file_inside_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
        path = _write_key_file(tmp_path / ".env")
        with pytest.raises(ConfigError, match="inside the repository"):
            load_api_keys(path)


class TestKeyMasking:
    def test_repr_never_contains_keys(self):
        keys = ApiKeys(openrouter=FAKE_OR, typesafe=FAKE_TS)
        assert FAKE_OR not in repr(keys)
        assert FAKE_TS not in repr(keys)
        assert FAKE_OR not in str(keys.status())

    def test_mask_key(self):
        assert mask_key(None) == "not set"
        assert mask_key("") == "not set"
        assert mask_key("short") == "set"
        assert mask_key(FAKE_OR) == "…wxyz"
