"""Tests for target-query guardrails and suggestions."""

from simcheck.core.query_quality import assess_target_query, suggest_target_queries


def test_rejects_placeholder_target():
    result = assess_target_query("needs matched")
    assert result.usable is False
    assert "placeholder" in result.error


def test_short_target_is_warning_not_error():
    result = assess_target_query("DKIM")
    assert result.usable is True
    assert "directional" in result.warning


def test_specific_target_is_usable_without_warning():
    result = assess_target_query("how to configure DKIM authentication")
    assert result.usable is True
    assert result.warning is None


def test_suggestions_use_specific_headings_and_skip_generic_ones():
    document = """# Acme manufacturing software

## Learn more
## Reduce factory downtime
## Real-time production dashboards
"""
    assert suggest_target_queries(document) == [
        "Acme manufacturing software",
        "Reduce factory downtime",
        "Real-time production dashboards",
    ]
