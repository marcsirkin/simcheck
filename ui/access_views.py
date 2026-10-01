"""
Login screen and usage guards for a hosted SimCheck.

The limiter lives in st.cache_resource so every session on the server
shares one set of daily counters.
"""

from __future__ import annotations

import time

import streamlit as st

from simcheck.access import (
    MAX_LOGIN_ATTEMPTS,
    PUBLIC_MAX_SITE_SAMPLE,
    AccessConfig,
    AccessConfigError,
    UsageLimiter,
    load_access_config,
)
from simcheck.config import ConfigError


LOCAL_USER = "local"
FAILED_LOGIN_DELAY_SECONDS = 1.0


@st.cache_resource(show_spinner=False)
def get_limiter(daily_budget: float) -> UsageLimiter:
    """One shared limiter per server process (per budget setting)."""
    return UsageLimiter(daily_budget=daily_budget)


def _config() -> AccessConfig:
    try:
        return load_access_config()
    except (AccessConfigError, ConfigError) as e:
        st.error(f"Access configuration error: {e}")
        st.stop()


def require_login() -> str:
    """
    Gate the app. Returns the signed-in person's name (or "local" when the
    gate is off); otherwise renders the sign-in form and stops the run.
    """
    cfg = _config()
    st.session_state.access_public = cfg.public
    st.session_state.access_budget = cfg.daily_budget
    if not cfg.required:
        st.session_state.user = LOCAL_USER
        return LOCAL_USER
    if st.session_state.get("user"):
        return st.session_state.user

    attempts = st.session_state.get("login_attempts", 0)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div class="app-title" style="margin-top:12vh">SimCheck</div>'
            '<p class="app-intro">How Google would rate a page, and whether AI search cites it.</p>',
            unsafe_allow_html=True,
        )
        if attempts >= MAX_LOGIN_ATTEMPTS:
            st.error("Too many attempts. Reload the page to try again.")
            st.stop()
        with st.form("login", border=False):
            code = st.text_input("Access code", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            name = cfg.check(code)
            if name:
                st.session_state.user = name
                st.session_state.login_attempts = 0
                st.rerun()
            st.session_state.login_attempts = attempts + 1
            time.sleep(FAILED_LOGIN_DELAY_SECONDS)  # slows guessing; codes are long anyway
            st.error("That code didn't work.")
        st.caption("Ask Marc for an access code.")
    st.stop()


def sign_out_button() -> None:
    """Small sign-out control, shown only when the gate is on."""
    if st.session_state.get("access_public") and st.session_state.get("user"):
        if st.button("Sign out", key="sign_out"):
            for k in ("user", "analysis", "probes", "explanation", "site_audit"):
                st.session_state.pop(k, None)
            st.rerun()


def guard(action: str) -> bool:
    """
    Consume one use of a paid action; show why if refused.

    Args:
        action: Key in simcheck.access.DAILY_LIMITS

    Returns:
        True if the action may proceed
    """
    limiter = get_limiter(st.session_state.get("access_budget", 5.0))
    ok, message = limiter.try_consume(st.session_state.get("user", LOCAL_USER), action)
    if not ok:
        st.warning(message)
    return ok


def max_site_sample(default_max: int) -> int:
    """Site Audit sample cap: tighter when hosted."""
    return PUBLIC_MAX_SITE_SAMPLE if st.session_state.get("access_public") else default_max
