"""
SimCheck - Interactive Streamlit Playground

A lightweight UI layer for exploring semantic similarity between
concepts and documents. This app wraps Features 1 (comparison engine)
and Feature 2 (diagnostics) with no additional semantic logic.

Usage:
    streamlit run app.py

Design Principles:
- Thin UI layer: all semantic logic lives in simcheck.core
- Explicit actions: user triggers each step manually
- Inspectable: expose chunk-level details for learning
- No persistence: session resets on reload
"""

import html

import streamlit as st
from markitdown import MarkItDown

# Import backend modules (Features 1 + 2 + 5 + 6)
from simcheck.core.engine import compare_query_to_document, ComparisonError
from simcheck.core.diagnostics import create_diagnostic_report
from simcheck.core.embeddings import DEFAULT_MODEL, get_model_info
from simcheck.core.chunker import ChunkingStrategy
from simcheck.core.models import ChunkLevel, SHORT_QUERY_THRESHOLDS
from simcheck.core.recommendations import (
    generate_recommendations,
    RecommendationPriority,
    RecommendationType,
)
from simcheck.core.geo import generate_geo_next_steps, GeoIntent, GeoPriority


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
<style>
/* ── Layout ─────────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}

/* ── Section label (muted, above cards) ─────────────────────────── */
.section-label {
    font-size: 0.82rem;
    color: #6B778C;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    margin-bottom: 8px;
}

/* ── CCS score display ──────────────────────────────────────────── */
.ccs-score-big {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    color: #172B4D;
}
.ccs-label {
    font-size: 0.85rem;
    color: #6B778C;
    margin-top: 0.2rem;
}

/* ── CCS potential badge ────────────────────────────────────────── */
.ccs-potential {
    display: inline-block;
    background: #E3FCEF;
    border: 1px solid #ABF5D1;
    color: #006644;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 9999px;
}

/* ── Priority badges ────────────────────────────────────────────── */
.badge-high {
    display: inline-block;
    background: #FFEBE6;
    color: #DE350B;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
}
.badge-medium {
    display: inline-block;
    background: #FFF0B3;
    color: #FF991F;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
}
.badge-low {
    display: inline-block;
    background: #E3FCEF;
    color: #00875A;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
}

/* ── Time estimate tag ──────────────────────────────────────────── */
.time-tag {
    display: inline-block;
    background: #F4F5F7;
    color: #6B778C;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 9999px;
    margin-left: 0.5rem;
}

/* ── Step number circle ─────────────────────────────────────────── */
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #1868DB;
    color: #FFFFFF;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
    flex-shrink: 0;
}

/* ── Signal pills ───────────────────────────────────────────────── */
.signal-strip {
    background: #F4F5F7;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 16px;
}
.signal-strip-label {
    font-size: 0.72rem;
    color: #6B778C;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-weight: 600;
    margin-bottom: 6px;
}
.signal-pill {
    display: inline-block;
    background: #FFFFFF;
    color: #505F79;
    font-size: 0.78rem;
    padding: 5px 14px;
    border-radius: 9999px;
    margin: 3px 4px 3px 0;
    border: 1px solid #DFE1E6;
}
.signal-pill-highlight {
    display: inline-block;
    background: #DEEBFF;
    color: #0747A6;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 9999px;
    margin: 3px 4px 3px 0;
    border: 1px solid #B3D4FF;
}

/* ── Input focus state ──────────────────────────────────────────── */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #1868DB !important;
    box-shadow: 0 0 0 2px rgba(24,104,219,0.2) !important;
}

/* ── Tighter Streamlit overrides ────────────────────────────────── */
h3 { margin-top: 0.5rem; margin-bottom: 0.5rem; }
.stExpander { border: 1px solid #DFE1E6 !important; border-radius: 8px !important; }

/* ── Compact footer ─────────────────────────────────────────────── */
.footer-caption {
    text-align: center;
    color: #97A0AF;
    font-size: 0.75rem;
    padding-top: 2rem;
}

/* ── Header subtitle ────────────────────────────────────────────── */
.header-subtitle {
    color: #6B778C;
    font-weight: 400;
    font-size: 1rem;
}
</style>
"""


# =============================================================================
# Example content (demo / first-run experience)
# =============================================================================

EXAMPLE_QUERY = "how to improve email deliverability"

EXAMPLE_DOCUMENT = """# The Complete Guide to Email Deliverability

Email deliverability is the ability of your messages to reach subscriber inboxes instead of spam folders. Improving email deliverability means fixing sender reputation, authenticating your domain, and keeping your list healthy. This guide covers the three areas that matter most.

## Why emails land in spam

Mailbox providers score every incoming message. Poor sender reputation, missing authentication, and low engagement push your email toward the spam folder. High bounce rates and spam complaints above 0.1% are the fastest way to damage deliverability.

## Authentication: SPF, DKIM, and DMARC

Set up all three authentication standards to prove you own your sending domain. SPF lists which servers may send for your domain. DKIM signs each message cryptographically. DMARC tells providers what to do when a message fails both checks. Publish a DMARC policy of at least p=quarantine once SPF and DKIM pass for 100% of your mail. See [Google's sender guidelines](https://support.google.com/mail/answer/81126) for current requirements.

## How our company got started

Back in 2014, three friends met at a hackathon in Austin and bonded over a shared love of clever engineering puzzles. After two failed prototypes and a memorable road trip to a demo day in San Francisco, the team finally rented its first tiny office above a taco shop. The early days were chaotic but joyful.

## Our office culture

We take coffee very seriously — the team runs a rotating barista schedule and a tasting club every Friday. Between the foosball table, the office dog Biscuit, and quarterly hiking retreats, we have built a culture people genuinely enjoy. Remote teammates join our rituals over video.

## List hygiene and monitoring

Remove subscribers who have not opened in 90 days, and use double opt-in to keep invalid addresses off your list. Keep your bounce rate below 2%. Monitor inbox placement weekly with seed testing, and watch Google Postmaster Tools for reputation drops. When deliverability dips, slow your sending volume and re-warm the domain gradually.
"""


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """
    Initialize session state variables.

    Session state tracks:
    - document_text: The pasted document
    - query_text: The concept/query
    - is_indexed: Whether document has been chunked/embedded
    - comparison_result: Output from Feature 1
    - diagnostic_report: Output from Feature 2
    - status_message: Current status for user feedback
    - error_message: Current error message (if any)
    - chunking_strategy: Selected chunking strategy
    - fetched_url: Last fetched URL
    - fetch_status: Status message from URL fetch
    - recommendation_report: Output from Feature 6 (recommendations)
    """
    defaults = {
        "document_text": "",
        "query_text": "",
        "is_indexed": False,
        "comparison_result": None,
        "diagnostic_report": None,
        "recommendation_report": None,
        "status_message": "",
        "error_message": "",
        "chunking_strategy": "flat",
        "fetched_url": "",
        "fetch_status": "",
        "last_analyzed_query": "",
        "last_analyzed_document": "",
        "last_analyzed_strategy": "flat",
        "has_seen_intro": False,
        "geo_intent": "auto",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results():
    """Clear comparison results when document changes."""
    st.session_state.is_indexed = False
    st.session_state.comparison_result = None
    st.session_state.diagnostic_report = None
    st.session_state.recommendation_report = None
    st.session_state.status_message = ""
    st.session_state.error_message = ""


def fetch_url_as_markdown(url: str) -> tuple[bool, str]:
    """
    Fetch a URL and convert it to Markdown using markitdown.

    Args:
        url: The webpage URL to convert

    Returns:
        Tuple of (success: bool, content_or_error: str)
    """
    try:
        md = MarkItDown()
        result = md.convert(url)
        content = result.text_content
        if content and len(content.strip()) > 0:
            return True, content
        else:
            return False, "Conversion returned empty content"
    except Exception as e:
        return False, f"Error: {str(e)}"


# =============================================================================
# Backend Integration (calls Features 1 + 2)
# =============================================================================

@st.cache_resource(show_spinner=False)
def warm_embedding_model() -> dict:
    """
    Load the embedding model once per server process.

    Called at startup so the first "Analyze Document" click doesn't stall
    on model loading (or a ~400MB download on a fresh machine).
    """
    return get_model_info(DEFAULT_MODEL)


def run_comparison(query: str, document: str, strategy: str) -> bool:
    """
    Run the full comparison pipeline.

    Calls Feature 1 (compare_query_to_document) and Feature 2
    (create_diagnostic_report). Updates session state with results.

    Returns True on success, False on error.
    """
    try:
        st.session_state.error_message = ""

        # Convert strategy string to enum
        chunking_strategy = ChunkingStrategy(strategy)

        # Feature 1: Run comparison with selected strategy
        result = compare_query_to_document(
            query,
            document,
            chunking_strategy=chunking_strategy,
        )
        st.session_state.comparison_result = result

        # Feature 2: Create diagnostic report
        report = create_diagnostic_report(result)
        st.session_state.diagnostic_report = report

        # Feature 6: Generate recommendations
        rec_report = generate_recommendations(report)
        st.session_state.recommendation_report = rec_report

        st.session_state.is_indexed = True
        st.session_state.last_analyzed_query = query
        st.session_state.last_analyzed_document = document
        st.session_state.last_analyzed_strategy = strategy

        # Include strategy info in status
        strategy_label = strategy.upper() if strategy != "flat" else "FLAT (default)"
        st.session_state.status_message = (
            f"Analysis complete. {result.chunk_count} chunks processed "
            f"using {strategy_label} strategy."
        )
        return True

    except ComparisonError as e:
        st.session_state.error_message = f"Comparison failed: {e}"
        st.session_state.is_indexed = False
        return False
    except Exception as e:
        st.session_state.error_message = f"Unexpected error: {e}"
        st.session_state.is_indexed = False
        return False


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the app header — minimal title with inline subtitle."""
    st.markdown(
        '# SimCheck <span class="header-subtitle">— semantic coverage analyzer</span>',
        unsafe_allow_html=True,
    )

    expanded = not bool(st.session_state.get("has_seen_intro"))
    with st.expander("How to use", expanded=expanded):
        st.markdown(
            """
- **Enter a target topic** (entity + intent), paste or fetch your content, then hit **Analyze**.
- Read the **Action Plan** for prioritized next steps to improve AI summarization / citation readiness.
- Expand **Detailed Diagnostics** to see per-chunk scores and section analysis.
            """.strip()
        )
    st.session_state.has_seen_intro = True


def render_input_section():
    """
    Render the input section: hero card with URL + topic, document below.

    Returns tuple of (query, document, strategy) strings.
    """
    # Apply pending example load before any widget below is instantiated
    # (Streamlit forbids writing to a widget key after the widget renders).
    if st.session_state.pop("load_example_pending", False):
        st.session_state.query_input = EXAMPLE_QUERY
        st.session_state.document_input = EXAMPLE_DOCUMENT
        st.session_state.strategy_select = "markdown"
        st.session_state.fetch_status = ""
        clear_results()

    # --- Hero input card ---
    st.markdown('<div class="section-label">Analyze a page</div>', unsafe_allow_html=True)

    with st.container(border=True):
        # URL row
        url_col, fetch_col = st.columns([5, 1])

        with url_col:
            url_input = st.text_input(
                "Page URL",
                placeholder="https://example.com/article",
                label_visibility="collapsed",
                key="url_input",
            )

        with fetch_col:
            fetch_clicked = st.button(
                "Fetch",
                type="secondary",
                use_container_width=True,
                disabled=not (url_input and url_input.strip()),
            )

        if fetch_clicked and url_input:
            with st.spinner("Fetching and converting..."):
                success, result = fetch_url_as_markdown(url_input.strip())

                if success:
                    st.session_state.document_input = result
                    st.session_state.fetch_status = f"Fetched {len(result):,} characters"
                    st.session_state.fetched_url = url_input
                    st.session_state.strategy_select = "markdown"
                    st.rerun()
                else:
                    st.error(f"Failed to fetch: {result}")

        if st.session_state.get("fetch_status"):
            st.caption(f"✓ {st.session_state.fetch_status} (via markitdown)")

        # Target topic
        query = st.text_input(
            "Target topic",
            placeholder='Target topic — phrase beats keyword: "DKIM email authentication", not "dkim"',
            label_visibility="collapsed",
            key="query_input",
            help=(
                "Phrase your topic the way a user would search or ask it (entity + intent). "
                "Bare keywords score systematically lower with embedding models; "
                "SimCheck auto-adjusts thresholds for 1–2 word topics, but a full phrase "
                "gives more reliable results."
            ),
        )

        example_col, _ = st.columns([1, 3])
        with example_col:
            if st.button("Load example", type="secondary", use_container_width=True):
                st.session_state.load_example_pending = True
                st.rerun()

    # --- Document text (outside card — it's large) ---
    document = st.text_area(
        "Document text",
        placeholder="Paste your document text here, or fetch from URL above...",
        height=200,
        label_visibility="collapsed",
        key="document_input",
    )

    if document:
        word_count = len(document.split())
        char_count = len(document)
        st.caption(f"{char_count:,} characters · ~{word_count:,} words")

    # --- Settings (compact expander) ---
    with st.expander("Settings", expanded=False):
        s_col, i_col = st.columns(2)

        with s_col:
            strategy = st.selectbox(
                "Chunking strategy",
                options=["flat", "auto", "markdown", "html"],
                format_func=lambda x: {
                    "flat": "Flat (sentence-based)",
                    "auto": "Auto-detect format",
                    "markdown": "Markdown (## / ### headings)",
                    "html": "HTML (<h2> / <h3> tags)",
                }[x],
                help="Choose how to chunk the document. Hierarchical strategies preserve heading structure.",
                key="strategy_select",
            )

        with i_col:
            intent = st.selectbox(
                "GEO intent",
                options=["auto", "informational", "how_to", "commercial"],
                format_func=lambda x: {
                    "auto": "Auto-detect (recommended)",
                    "informational": "Informational (define/explain)",
                    "how_to": "How-to (steps/procedure)",
                    "commercial": "Commercial (compare/choose/buy)",
                }[x],
                help="Override intent detection for tailored action plan suggestions.",
                key="geo_intent",
            )
            _ = intent

    return query, document, strategy


def render_action_buttons(query: str, document: str, strategy: str):
    """Render the analyze button (full-width) and status messages."""
    query_valid = bool(query and query.strip())
    document_valid = bool(document and document.strip())
    can_analyze = query_valid and document_valid

    if st.button(
        "Analyze Document",
        disabled=not can_analyze,
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Chunking, embedding, and scoring..."):
            success = run_comparison(query, document, strategy)

        if success:
            st.rerun()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    elif not can_analyze:
        if not query_valid and not document_valid:
            st.caption("Enter a topic and document to begin")
        elif not query_valid:
            st.caption("Enter a target topic")
        else:
            st.caption("Paste or fetch a document")
    elif st.session_state.status_message:
        st.success(st.session_state.status_message)


# =============================================================================
# Helper functions
# =============================================================================

def get_coverage_color(score: float) -> str:
    """Get color indicator for coverage score."""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 40:
        return "🟠"
    else:
        return "🔴"


def get_interpretation_color(interpretation: str) -> str:
    """Get a color indicator for a chunk's interpretation label."""
    colors = {
        "Strong": "🟢",
        "Moderate": "🟡",
        "Weak": "🟠",
        "Off-topic": "🔴",
    }
    return colors.get(interpretation, "⚪")


def get_level_badge(level: ChunkLevel) -> str:
    """Get a badge/indicator for chunk level."""
    badges = {
        ChunkLevel.MACRO: "🔷 MACRO",
        ChunkLevel.MICRO: "🔹 MICRO",
        ChunkLevel.ATOMIC: "⬜ ATOMIC",
        ChunkLevel.FLAT: "📄 FLAT",
    }
    return badges.get(level, "📄")


def _geo_priority_badge(priority: GeoPriority) -> str:
    """Return HTML badge for a GEO priority level."""
    if priority == GeoPriority.HIGH:
        return '<span class="badge-high">HIGH</span>'
    if priority == GeoPriority.MEDIUM:
        return '<span class="badge-medium">MEDIUM</span>'
    return '<span class="badge-low">LOW</span>'


def _rec_priority_badge(priority: RecommendationPriority) -> str:
    """Return HTML badge for a recommendation priority level."""
    if priority == RecommendationPriority.HIGH:
        return '<span class="badge-high">HIGH</span>'
    if priority == RecommendationPriority.MEDIUM:
        return '<span class="badge-medium">MEDIUM</span>'
    return '<span class="badge-low">LOW</span>'


def _geo_priority_text(priority: GeoPriority) -> str:
    """Return plain-text badge for sorting."""
    if priority == GeoPriority.HIGH:
        return "🔴 HIGH"
    if priority == GeoPriority.MEDIUM:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def get_rec_type_icon(rec_type: RecommendationType) -> str:
    """Get an icon for recommendation type."""
    icons = {
        RecommendationType.REWRITE_OFF_TOPIC: "✏️",
        RecommendationType.STRENGTHEN_WEAK: "💪",
        RecommendationType.EXPAND_STRONG: "📈",
        RecommendationType.RESTRUCTURE_SECTION: "🏗️",
        RecommendationType.REMOVE_DILUTION: "✂️",
    }
    return icons.get(rec_type, "📋")


def _ccs_interpretation_line(score: float) -> str:
    """Contextual interpretation next to the CCS score."""
    if score >= 80:
        return "Strong topical focus"
    elif score >= 60:
        return "Decent focus — fix weak / off-topic sections"
    elif score >= 40:
        return "Weak focus — restructure and improve intro"
    else:
        return "Low focus — content likely doesn't answer the topic"


# =============================================================================
# Results rendering
# =============================================================================

def render_ccs_score():
    """Render the CCS score as a colored-accent banner card."""
    result = st.session_state.comparison_result
    report = st.session_state.diagnostic_report
    rec_report = st.session_state.recommendation_report

    if not result or not report:
        return

    coverage = report.coverage
    interp = _ccs_interpretation_line(coverage.score)

    # Pick accent color for the CCS band
    if coverage.score >= 80:
        accent_color = "#00875A"
    elif coverage.score >= 60:
        accent_color = "#FF991F"
    elif coverage.score >= 40:
        accent_color = "#DE350B"
    else:
        accent_color = "#97A0AF"

    with st.container(border=True):
        # Colored accent bar at top of card
        st.markdown(
            f'<div style="height:4px;background:{accent_color};border-radius:2px;margin:-8px 0 16px 0;"></div>',
            unsafe_allow_html=True,
        )

        score_col, detail_col = st.columns([1, 3])

        with score_col:
            st.markdown(
                f'<div class="ccs-score-big">{coverage.score_rounded}</div>'
                f'<div class="ccs-label">CCS (0–100)</div>',
                unsafe_allow_html=True,
            )

        with detail_col:
            st.markdown(f"**{interp}**")

            # Bucket counts
            bc = coverage.bucket_counts
            st.caption(
                f"Strong: {bc['strong']} · Moderate: {bc['moderate']} · "
                f"Weak: {bc['weak']} · Off-topic: {bc['off_topic']}"
            )

            # CCS potential badge
            if rec_report and rec_report.has_recommendations():
                improvement = rec_report.potential_ccs - rec_report.current_ccs
                if improvement > 0:
                    st.markdown(
                        f'<span class="ccs-potential">'
                        f"CCS Potential: {rec_report.current_ccs:.0f} → {rec_report.potential_ccs:.0f} "
                        f"(+{improvement:.0f})"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

            if coverage.is_single_chunk:
                st.caption("Single-chunk document — score may be less reliable")

            # Score interpretation bands
            st.caption("80+ strong · 60–79 decent · 40–59 weak · <40 low")

            if report.thresholds == SHORT_QUERY_THRESHOLDS:
                st.caption(
                    "Short topic (1–2 words) — thresholds auto-calibrated. "
                    "A full phrase gives more reliable scoring."
                )

            st.caption(
                "CCS is a relative measure: use it to compare drafts and track "
                "improvement for the same topic — it is not comparable to "
                "relevance grades from an LLM."
            )


# Drift map bucket colors — colorblind-safe diverging ladder (cool = aligned,
# warm = drifting). Validated for CVD separation, lightness band, and chroma;
# identity is never color-alone (bar height, hover tooltip, legend, and the
# diagnostics table all carry the same information).
DRIFT_COLORS = {
    "Strong": "#0055CC",
    "Moderate": "#579DFF",
    "Weak": "#E8770C",
    "Off-topic": "#96261B",
}


def render_drift_map():
    """Render the drift map: one bar per chunk, in document order."""
    report = st.session_state.diagnostic_report
    if not report or report.summary.total_chunks < 2:
        return

    bars = []
    for c in report.by_document_order():
        color = DRIFT_COLORS.get(c.interpretation, "#97A0AF")
        bar_height = max(8, round(c.similarity * 64))
        # Collapse whitespace: a newline inside the HTML string would end the
        # markdown HTML block and break the flex row.
        preview = " ".join(c.text_preview.split())
        tooltip = html.escape(
            f"Chunk {c.chunk_index + 1} · {c.similarity:.2f} ({c.interpretation}) — {preview}",
            quote=True,
        )
        bars.append(
            f'<div title="{tooltip}" style="flex:1;max-width:48px;height:{bar_height}px;'
            f'background:{color};border-radius:3px 3px 0 0;"></div>'
        )

    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;'
        f'font-size:0.78rem;color:#505F79;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{color};"></span>'
        f"{label}</span>"
        for label, color in DRIFT_COLORS.items()
    )

    with st.container(border=True):
        st.markdown("### Drift Map")
        st.caption(
            "Each bar is one chunk, in document order. Taller = more aligned with the "
            "target topic. Hover a bar for its score and text."
        )
        st.markdown(
            f'<div style="display:flex;align-items:flex-end;gap:2px;height:68px;'
            f'border-bottom:1px solid #DFE1E6;">{"".join(bars)}</div>'
            f'<div style="margin-top:8px;">{legend}</div>',
            unsafe_allow_html=True,
        )


def render_action_plan():
    """Render the combined action plan: GEO steps + merged recommendations."""
    report = st.session_state.diagnostic_report
    rec_report = st.session_state.recommendation_report

    if not report:
        return

    document = st.session_state.get("last_analyzed_document") or ""
    if not document.strip():
        return

    intent_override = GeoIntent(st.session_state.get("geo_intent", "auto"))
    geo = generate_geo_next_steps(report, document, intent_override=intent_override)

    with st.container(border=True):
        st.markdown("### Action Plan")
        st.write(geo.summary)

        # --- Content signals in a tinted strip ---
        sig = geo.signals
        pills = []
        pills.append(f'<span class="signal-pill-highlight">{geo.intent.value.replace("_", " ").title()} intent</span>')
        pills.append(f'<span class="signal-pill">{sig.word_count:,} words</span>')
        pills.append(f'<span class="signal-pill">{sig.h2_count + sig.h3_count} headings</span>')
        pills.append(f'<span class="signal-pill">{sig.link_count} links</span>')
        if sig.has_faq:
            pills.append('<span class="signal-pill-highlight">FAQ detected</span>')
        if sig.has_tldr:
            pills.append('<span class="signal-pill-highlight">TL;DR detected</span>')
        pills.append(f'<span class="signal-pill">Intro coverage: {sig.intro_query_term_coverage:.0%}</span>')

        st.markdown(
            f'<div class="signal-strip">'
            f'<div class="signal-strip-label">Content signals</div>'
            f'{" ".join(pills)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Collect all steps ---
    all_steps = []

    if geo.steps:
        for step in geo.steps:
            all_steps.append(("geo", step))

    if rec_report and rec_report.has_recommendations():
        for rec in rec_report.recommendations:
            all_steps.append(("rec", rec))

    if not all_steps:
        st.info("No action items detected. Try a stricter target topic, or analyze a longer document.")
        return

    # --- Render steps: single-column, each in a bordered container ---
    for idx, (source, item) in enumerate(all_steps):
        if source == "geo":
            _render_geo_step_card(idx + 1, item)
        else:
            _render_rec_step_card(idx + 1, item)


def _render_geo_step_card(num: int, step):
    """Render a single GEO step inside a bordered container."""
    badge = _geo_priority_badge(step.priority)
    time_html = f'<span class="time-tag">~{step.minutes} min</span>'

    with st.container(border=True):
        st.markdown(
            f'<span class="step-num">{num}</span>'
            f'<strong>{step.title}</strong> {badge} {time_html}',
            unsafe_allow_html=True,
        )

        st.caption(f"**Why:** {step.why}")
        st.write(f"**Do this:** {step.how}")

        if step.examples:
            with st.expander("Template / example"):
                st.code(step.examples, language="markdown")

        if step.target_chunks:
            with st.expander(f"Where to edit ({len(step.target_chunks)})"):
                for chunk in step.target_chunks:
                    color = get_interpretation_color(chunk.interpretation)
                    st.write(
                        f"**#{chunk.chunk_index + 1}** {color} {chunk.similarity:.2f} "
                        f"({chunk.interpretation})"
                    )
                    st.caption(chunk.text_preview)


def _render_rec_step_card(num: int, rec):
    """Render a single recommendation inside a bordered container."""
    icon = get_rec_type_icon(rec.rec_type)
    badge = _rec_priority_badge(rec.priority)

    with st.container(border=True):
        st.markdown(
            f'<span class="step-num">{num}</span>'
            f'<strong>{icon} {rec.what}</strong> {badge}',
            unsafe_allow_html=True,
        )

        st.caption(f"**Impact:** {rec.why}")
        st.write(f"**Fix:** {rec.how}")

        if rec.example_text:
            with st.expander("Example from strong content"):
                st.info(rec.example_text)

        if rec.target_chunks:
            with st.expander(f"Affected chunks ({len(rec.target_chunks)})"):
                for target in rec.target_chunks:
                    color = get_interpretation_color(target.interpretation)
                    st.write(
                        f"**#{target.chunk_index + 1}** {color} {target.similarity:.2f} "
                        f"({target.interpretation})"
                    )
                    st.caption(target.text_preview)


# =============================================================================
# Detailed Diagnostics (collapsed expander)
# =============================================================================

def render_diagnostics_expander():
    """Render all diagnostics inside a single collapsed expander."""
    report = st.session_state.diagnostic_report
    result = st.session_state.comparison_result

    if not report or not result:
        return

    with st.expander("Detailed Diagnostics", expanded=False):
        st.caption("For power users — per-chunk scores, section analysis, and debug data")

        # --- Similarity Metrics ---
        st.write("**Similarity Metrics**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Max Similarity",
                f"{result.max_similarity:.2f}",
                help="Highest similarity score across all chunks",
            )
        with col2:
            st.metric(
                "Avg Similarity",
                f"{result.avg_similarity:.2f}",
                help="Mean similarity across all chunks",
            )
        with col3:
            st.metric(
                "Chunks",
                result.chunk_count,
                help="Number of document chunks analyzed",
            )
        with col4:
            st.metric(
                "On-Topic",
                f"{report.summary.percent_on_topic:.0f}%",
                help=f"Percentage of chunks with similarity >= {report.effective_thresholds['weak']:.2f}",
            )

        # More stats
        summary = report.summary
        coverage = report.coverage

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Similarity Distribution**")
            st.write(f"- Median: {summary.median_similarity:.3f}")
            st.write(f"- Std Dev: {summary.std_similarity:.3f}")
            st.write(f"- Min: {summary.min_similarity:.3f}")
            st.write(f"- Max: {summary.max_similarity:.3f}")
        with col2:
            t = report.effective_thresholds
            st.write("**Chunk Breakdown**")
            st.write(f"- Strong (>= {t['strong']:.2f}): {summary.chunks_strong}")
            st.write(f"- Moderate ({t['moderate']:.2f}-{t['strong']:.2f}): {summary.chunks_moderate}")
            st.write(f"- Weak ({t['weak']:.2f}-{t['moderate']:.2f}): {summary.chunks_weak}")
            st.write(f"- Off-topic (< {t['weak']:.2f}): {summary.chunks_off_topic}")
        with col3:
            st.write("**CCS Calculation**")
            st.write(f"- Weighted Sum: {coverage.weighted_sum:.2f}")
            st.write(f"- Total Chunks: {coverage.total_chunks}")
            st.write(
                f"- Score: ({coverage.weighted_sum:.2f} / {coverage.total_chunks}) "
                f"x 100 = **{coverage.score:.1f}**"
            )

        # --- Best & Worst ---
        if report.summary.total_chunks >= 2:
            st.write("**Best & Worst Chunks**")
            col1, col2 = st.columns(2)
            with col1:
                best = report.get_max_chunk()
                if best:
                    st.write(f"**Highest:** Chunk #{best.chunk_index + 1} — **{best.similarity:.3f}**")
                    st.caption(best.text_preview)
            with col2:
                worst = report.get_min_chunk()
                if worst:
                    st.write(f"**Lowest:** Chunk #{worst.chunk_index + 1} — **{worst.similarity:.3f}**")
                    st.caption(worst.text_preview)

        # --- Section Analysis ---
        if report.is_hierarchical():
            st.write("**Section Analysis**")
            sections = report.get_sections()
            if sections:
                for section in sections:
                    level_badge = get_level_badge(section.level)
                    heading_text = section.heading or "(No heading)"
                    cov_color = get_coverage_color(section.coverage_score)

                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"{level_badge} **{heading_text}**")
                        st.caption(f"{section.chunk_count} chunks, {section.total_words} words")
                    with col2:
                        st.metric("Avg Sim", f"{section.avg_similarity:.2f}")
                    with col3:
                        st.metric("Max", f"{section.max_similarity:.2f}")
                    with col4:
                        st.write(f"{cov_color} **{section.coverage_score:.0f}**")
                        st.caption("Coverage")
            else:
                st.info("No section structure detected in the document.")

        # --- Chunk Analysis ---
        st.write("**Chunk Analysis**")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_option = st.radio(
                "Sort by",
                ["Document Order", "Similarity (High to Low)", "Similarity (Low to High)"],
                horizontal=True,
                key="sort_option",
            )
        with col2:
            show_full_text = st.checkbox("Show full text", value=False, key="show_full")
        with col3:
            if report.is_hierarchical():
                level_filter = st.selectbox(
                    "Filter level",
                    options=["All", "MACRO", "MICRO", "ATOMIC"],
                    key="level_filter",
                )
            else:
                level_filter = "All"

        if sort_option == "Document Order":
            chunks = report.by_document_order()
        elif sort_option == "Similarity (High to Low)":
            chunks = report.by_similarity_descending()
        else:
            chunks = report.by_similarity_ascending()

        if level_filter != "All":
            level_enum = ChunkLevel(level_filter.lower())
            chunks = [c for c in chunks if c.level == level_enum]

        for chunk in chunks:
            with st.container():
                if report.is_hierarchical():
                    c1, c2, c3, c4 = st.columns([0.5, 0.8, 1, 3.7])
                else:
                    c1, c2, c3 = st.columns([0.5, 1, 4])
                    c4 = None

                with c1:
                    st.write(f"**#{chunk.chunk_index + 1}**")
                with c2:
                    color = get_interpretation_color(chunk.interpretation)
                    st.write(f"{color} **{chunk.similarity:.3f}**")
                    st.caption(chunk.interpretation)

                if c4 is not None:
                    with c3:
                        st.write(get_level_badge(chunk.level))
                        if chunk.heading:
                            st.caption(
                                chunk.heading[:20] + "..." if len(chunk.heading) > 20 else chunk.heading
                            )
                    with c4:
                        st.write(chunk.text if show_full_text else chunk.text_preview)
                else:
                    with c3:
                        st.write(chunk.text if show_full_text else chunk.text_preview)

        # --- Debug Info ---
        st.write("**Debug Info**")
        st.write(f"- Query: `{result.query}`")
        try:
            model_info = get_model_info(DEFAULT_MODEL)
            st.write(f"- Model: `{model_info['model_name']}` ({model_info['embedding_dim']} dims)")
        except Exception:
            st.write(f"- Model: `{DEFAULT_MODEL}`")
        st.write(f"- Embedding Dim: {result.embedding_dim}")
        st.write(f"- Document Chars: {result.document_char_count:,}")
        st.write(f"- Document Tokens: {result.document_token_count:,}")

        if report.is_hierarchical():
            st.json(report.hierarchy_heatmap_data())
        else:
            st.json(report.get_heatmap_data())


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="SimCheck",
        page_icon="🔍",
        layout="wide",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize state
    init_session_state()

    # Warm the embedding model so the first analysis doesn't stall
    with st.spinner("Preparing embedding model..."):
        warm_embedding_model()

    # Header
    render_header()

    # Input section
    query, document, strategy = render_input_section()

    # Action button
    render_action_buttons(query, document, strategy)

    # --- Single-page results flow (no tabs) ---
    if st.session_state.is_indexed:
        st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)

        # 1. CCS Score (colored banner card)
        render_ccs_score()

        # 2. Drift map (per-chunk alignment, document order)
        render_drift_map()

        # 3. Action Plan (GEO steps + merged recommendations)
        render_action_plan()

        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)

        # 4. Detailed Diagnostics (collapsed)
        render_diagnostics_expander()

    # Footer
    st.markdown(
        '<div class="footer-caption">SimCheck v1.2.0 · Local-only semantic analysis · No data leaves your machine</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
