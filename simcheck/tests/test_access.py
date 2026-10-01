"""Tests for the login gate and usage limits (simcheck.access)."""

import threading

import pytest

from simcheck import config
from simcheck.access import (
    MIN_CODE_LENGTH,
    AccessConfigError,
    UsageLimiter,
    generate_code,
    load_access_config,
    parse_codes,
)


A = "alice-" + "a" * 20
B = "bob-" + "b" * 20


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for name in ("SIMCHECK_ACCESS_CODES", "SIMCHECK_REQUIRE_LOGIN", "SIMCHECK_DAILY_BUDGET_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SIMCHECK_ENV_FILE", str(tmp_path / "missing.env"))


class TestCodes:
    def test_generate_code_is_long_and_unique(self):
        codes = {generate_code() for _ in range(50)}
        assert len(codes) == 50 and all(len(c) >= MIN_CODE_LENGTH for c in codes)

    def test_parse(self):
        assert parse_codes(f" alice:{A} , bob:{B},") == {A: "alice", B: "bob"}
        assert parse_codes(None) == {} and parse_codes("") == {}

    @pytest.mark.parametrize("raw,match", [
        ("alice", "name:code"), (":" + A, "name:code"), ("alice:", "name:code"),
        ("alice:short", "too short"), (f"a:{A},b:{A}", "Duplicate"),
    ])
    def test_parse_errors(self, raw, match):
        with pytest.raises(AccessConfigError, match=match):
            parse_codes(raw)

    def test_error_never_echoes_a_code(self):
        with pytest.raises(AccessConfigError) as e:
            parse_codes("alice:short-secret")
        assert "short-secret" not in str(e.value)


class TestConfig:
    def test_local_default_is_open(self):
        cfg = load_access_config()
        assert not cfg.required and not cfg.public

    def test_codes_turn_gate_on(self, monkeypatch):
        monkeypatch.setenv("SIMCHECK_ACCESS_CODES", f"alice:{A}")
        cfg = load_access_config()
        assert cfg.required and cfg.check(A) == "alice" and cfg.check(" " + A + " ") == "alice"
        assert cfg.check("wrong") is None and cfg.check("") is None

    def test_require_login_without_codes_fails_closed(self, monkeypatch):
        monkeypatch.setenv("SIMCHECK_REQUIRE_LOGIN", "true")
        cfg = load_access_config()
        assert cfg.required and cfg.check(A) is None

    def test_codes_hidden_from_repr(self, monkeypatch):
        monkeypatch.setenv("SIMCHECK_ACCESS_CODES", f"alice:{A}")
        assert A not in repr(load_access_config())

    def test_budget(self, monkeypatch):
        monkeypatch.setenv("SIMCHECK_DAILY_BUDGET_USD", "12.5")
        assert load_access_config().daily_budget == 12.5
        monkeypatch.setenv("SIMCHECK_DAILY_BUDGET_USD", "lots")
        with pytest.raises(AccessConfigError):
            load_access_config()

    def test_reads_codes_from_key_file(self, monkeypatch, tmp_path):
        path = tmp_path / "k.env"
        path.write_text(f"SIMCHECK_ACCESS_CODES=bob:{B}\n")
        path.chmod(0o600)
        monkeypatch.setenv("SIMCHECK_ENV_FILE", str(path))
        assert load_access_config().check(B) == "bob"


class TestLimiter:
    def _limiter(self, **kw):
        day = {"v": "2026-10-01"}
        lim = UsageLimiter(clock=lambda: day["v"], **kw)
        return lim, day

    def test_per_user_cap(self):
        lim, _ = self._limiter(limits={"probe": 2}, costs={"probe": 0.01})
        assert lim.try_consume("alice", "probe")[0]
        assert lim.try_consume("alice", "probe")[0]
        ok, msg = lim.try_consume("alice", "probe")
        assert not ok and "Daily limit" in msg
        assert lim.try_consume("bob", "probe")[0]
        assert lim.remaining("alice", "probe") == 0 and lim.remaining("bob", "probe") == 1

    def test_global_budget(self):
        lim, _ = self._limiter(daily_budget=0.05, limits={"probe": 100}, costs={"probe": 0.02})
        assert lim.try_consume("a", "probe")[0] and lim.try_consume("b", "probe")[0]
        ok, msg = lim.try_consume("c", "probe")
        assert not ok and "budget" in msg
        assert lim.spend_today == pytest.approx(0.04)

    def test_resets_each_day(self):
        lim, day = self._limiter(limits={"probe": 1}, costs={"probe": 0.01})
        assert lim.try_consume("a", "probe")[0] and not lim.try_consume("a", "probe")[0]
        day["v"] = "2026-10-02"
        assert lim.try_consume("a", "probe")[0] and lim.spend_today == pytest.approx(0.01)

    def test_refusal_does_not_count(self):
        lim, _ = self._limiter(daily_budget=0.01, limits={"explain": 5}, costs={"explain": 0.02})
        assert not lim.try_consume("a", "explain")[0]
        assert lim.remaining("a", "explain") == 5 and lim.spend_today == 0

    def test_thread_safe(self):
        lim, _ = self._limiter(daily_budget=1000, limits={"analyze": 50}, costs={"analyze": 0.001})
        results = []
        threads = [threading.Thread(target=lambda: results.append(lim.try_consume("a", "analyze")[0]))
                   for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(results) == 50

    def test_unknown_action(self):
        lim, _ = self._limiter()
        with pytest.raises(KeyError):
            lim.try_consume("a", "teleport")
