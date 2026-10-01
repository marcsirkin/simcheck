"""
Access control for a hosted SimCheck: login gate and usage limits.

Login: per-person access codes in SIMCHECK_ACCESS_CODES, formatted
"name:code,name:code". Codes should be long and random (see
generate_code); each person gets their own so usage is attributable and
one code can be revoked without touching the others.

The gate is ON when codes are configured or SIMCHECK_REQUIRE_LOGIN is
truthy. Require-login with no codes admits nobody (fails closed), so a
deploy that forgets the codes is locked rather than open. Locally, with
neither set, the gate is off.

Limits: per-person daily caps per paid action plus a global daily spend
cap across everyone. Counters are in memory (reset on restart, UTC day
boundaries); the credit limits set on the API keys themselves are the
hard backstop.
"""

from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from simcheck.config import load_setting


MIN_CODE_LENGTH = 16
MAX_LOGIN_ATTEMPTS = 5

# Per-person daily caps
DAILY_LIMITS = {
    "analyze": 40,
    "explain": 20,
    "probe": 10,        # probe runs (each up to MAX_PROBE_QUERIES questions)
    "site_audit": 3,
}

# Estimated USD per action, for the global budget (spike/live measurements)
ACTION_COST = {
    "analyze": 0.002,
    "explain": 0.012,
    "probe": 0.02,
    "site_audit": 0.06,
}

DEFAULT_DAILY_BUDGET_USD = 5.0
PUBLIC_MAX_SITE_SAMPLE = 25


class AccessConfigError(Exception):
    """Raised when SIMCHECK_ACCESS_CODES is malformed or a code is too weak."""


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def generate_code() -> str:
    """A new random access code (24 url-safe characters)."""
    return secrets.token_urlsafe(18)


def parse_codes(raw: Optional[str]) -> dict:
    """
    Parse "name:code,name:code" into {code: name}.

    Raises:
        AccessConfigError: On a malformed entry, duplicate code, or a code
            shorter than MIN_CODE_LENGTH
    """
    codes = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, code = entry.partition(":")
        name, code = name.strip(), code.strip()
        if not sep or not name or not code:
            raise AccessConfigError(f"Access code entries must be name:code (got {name or entry[:3] + '...'!r}).")
        if len(code) < MIN_CODE_LENGTH:
            raise AccessConfigError(f"Access code for {name!r} is too short (min {MIN_CODE_LENGTH} characters).")
        if code in codes:
            raise AccessConfigError("Duplicate access code.")
        codes[code] = name
    return codes


@dataclass(frozen=True)
class AccessConfig:
    """Login and budget settings."""
    required: bool
    codes: dict = field(repr=False)  # code -> person name; never shown
    daily_budget: float = DEFAULT_DAILY_BUDGET_USD

    @property
    def public(self) -> bool:
        """Hosted mode: gate on, tighter caps."""
        return self.required

    def check(self, code: str) -> Optional[str]:
        """
        Person name for a code, or None. Constant-time comparison per code.
        """
        code = (code or "").strip()
        match = None
        for known, name in self.codes.items():
            if hmac.compare_digest(known.encode(), code.encode()):
                match = name
        return match


def load_access_config() -> AccessConfig:
    """
    Read SIMCHECK_ACCESS_CODES, SIMCHECK_REQUIRE_LOGIN, SIMCHECK_DAILY_BUDGET_USD.

    Raises:
        AccessConfigError: On malformed codes or budget
    """
    codes = parse_codes(load_setting("SIMCHECK_ACCESS_CODES"))
    required = bool(codes) or _truthy(load_setting("SIMCHECK_REQUIRE_LOGIN"))
    budget_raw = load_setting("SIMCHECK_DAILY_BUDGET_USD")
    try:
        budget = float(budget_raw) if budget_raw else DEFAULT_DAILY_BUDGET_USD
    except ValueError as e:
        raise AccessConfigError(f"SIMCHECK_DAILY_BUDGET_USD must be a number (got {budget_raw!r}).") from e
    return AccessConfig(required=required, codes=codes, daily_budget=budget)


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class UsageLimiter:
    """
    Thread-safe daily counters per person and a global spend estimate.

    One instance per server process (the app holds it in st.cache_resource).

    Args:
        daily_budget: Global USD cap per UTC day
        limits: Per-person daily caps per action
        costs: Estimated USD per action
        clock: Returns the current day key (tests inject one)
    """

    def __init__(self, daily_budget: float = DEFAULT_DAILY_BUDGET_USD, limits: Optional[dict] = None,
                 costs: Optional[dict] = None, clock: Callable[[], str] = _utc_day):
        self.daily_budget = daily_budget
        self.limits = dict(limits or DAILY_LIMITS)
        self.costs = dict(costs or ACTION_COST)
        self._clock = clock
        self._lock = threading.Lock()
        self._day = None
        self._counts: dict = {}
        self._spend = 0.0

    def _roll(self) -> None:
        today = self._clock()
        if today != self._day:
            self._day, self._counts, self._spend = today, {}, 0.0

    def try_consume(self, user: str, action: str) -> tuple:
        """
        Record one use if it fits both the person's cap and the global budget.

        Args:
            user: Person name ("local" when the gate is off)
            action: Key in DAILY_LIMITS

        Returns:
            (allowed: bool, message: str); message explains a refusal

        Raises:
            KeyError: On an unknown action (programming error)
        """
        limit = self.limits[action]
        cost = self.costs[action]
        with self._lock:
            self._roll()
            used = self._counts.get((user, action), 0)
            if used >= limit:
                return False, f"Daily limit reached for this action ({limit} per day). Resets at midnight UTC."
            if self._spend + cost > self.daily_budget:
                return False, "SimCheck's shared daily budget is used up. Try again after midnight UTC."
            self._counts[(user, action)] = used + 1
            self._spend += cost
            return True, ""

    def remaining(self, user: str, action: str) -> int:
        """Uses left today for a person and action."""
        with self._lock:
            self._roll()
            return max(0, self.limits[action] - self._counts.get((user, action), 0))

    @property
    def spend_today(self) -> float:
        """Estimated USD spent today across everyone."""
        with self._lock:
            self._roll()
            return self._spend
