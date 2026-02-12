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

import streamlit as st
from markitdown import MarkItDown

# Import backend modules (Features 1 + 2 + 5 + 6)
from simcheck.core.engine import compare_query_to_document, ComparisonError
from simcheck.core.diagnostics import create_diagnostic_report, SortOrder
from simcheck.core.embeddings import DEFAULT_MODEL, get_model_info
from simcheck.core.chunker import ChunkingStrategy
from simcheck.core.models import ChunkLevel
from simcheck.core.recommendations import (
    generate_recommendations,
    RecommendationPriority,
    RecommendationType,
)
from simcheck.core.geo import generate_geo_next_steps, GeoPriority
from simcheck.core.geo import GeoIntent


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
    """Render the app header."""
    st.title("SimCheck")
    st.caption("AI visibility (GEO) focused semantic coverage + next steps")

    # Show model info
    try:
        model_info = get_model_info(DEFAULT_MODEL)
        st.caption(f"Model: `{model_info['model_name']}` ({model_info['embedding_dim']} dims)")
    except Exception:
        st.caption(f"Model: `{DEFAULT_MODEL}`")

    expanded = not bool(st.session_state.get("has_seen_intro"))
    with st.expander("Start here: how to use this (and what a “good” score means)", expanded=expanded):
        st.markdown(
            """
**Goal:** make your page easier for AI systems to *understand, summarize, and cite* for a target topic.

**How to use (fast loop):**
1) Enter a **Target topic** (include the entity + intent; be specific).
2) Paste your content (or **Fetch from URL**) and choose a chunking strategy.
3) Click **Analyze Document**.
4) Read **Action Plan (GEO next steps)** first, then drill into chunk/section details.
5) Edit your content and re-run to compare drafts.

**Score interpretation (CCS 0–100):**
- **80+**: strong topical focus (good baseline)
- **60–79**: decent focus; fix weak/off-topic sections
- **40–59**: weak focus; restructure + improve intro
- **<40**: low focus; content likely doesn’t answer the topic directly

CCS is a *semantic alignment* metric (not a ranking guarantee). Use it to compare versions of your content.
            """.strip()
        )
    st.session_state.has_seen_intro = True


def render_input_section():
    """
    Render the input section with query and document text areas.

    Returns tuple of (query, document, strategy) strings.
    """
    st.subheader("Inputs")

    # Query input
    query = st.text_input(
        "Target topic (keyword/entity + intent)",
        placeholder="e.g., “how to choose a CRM for a small business”",
        help=(
            "Be specific. Include the entity and the user intent. "
            "Examples: “best email warmup tools”, “what is retrieval augmented generation”, "
            "“GEO for local service businesses”."
        ),
        key="query_input",
    )

    # URL Fetcher section
    with st.expander("Fetch from URL", expanded=False):
        st.caption("Convert a webpage to Markdown for analysis")

        col1, col2 = st.columns([4, 1])

        with col1:
            url_input = st.text_input(
                "URL",
                placeholder="https://example.com/article",
                help="Enter a URL to fetch and convert to Markdown",
                key="url_input",
                label_visibility="collapsed",
            )

        with col2:
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
                    # Directly set the document input widget's state
                    st.session_state.document_input = result
                    st.session_state.fetch_status = f"Fetched {len(result):,} characters"
                    st.session_state.fetched_url = url_input
                    # Set strategy to markdown
                    st.session_state.strategy_select = "markdown"
                    st.rerun()
                else:
                    st.error(f"Failed to fetch: {result}")

        if st.session_state.get("fetch_status"):
            st.success(st.session_state.fetch_status)

        st.caption(
            "Powered by markitdown | Content is converted locally"
        )

    # Document input
    document = st.text_area(
        "Document Text",
        placeholder="Paste your document text here, or fetch from URL above...",
        height=200,
        help="Paste the document you want to analyze",
        key="document_input",
    )

    # Show character/word counts
    if document:
        word_count = len(document.split())
        char_count = len(document)
        st.caption(f"{char_count:,} characters, ~{word_count:,} words")

    # Chunking strategy selector
    st.subheader("Chunking Strategy")

    strategy = st.selectbox(
        "Strategy",
        options=["flat", "auto", "markdown", "html"],
        format_func=lambda x: {
            "flat": "Flat (default) - Sentence-based chunking",
            "auto": "Auto-detect - Detect document format",
            "markdown": "Markdown - Parse ## and ### headings",
            "html": "HTML - Parse <h2> and <h3> tags",
        }[x],
        help="Choose how to chunk the document. Hierarchical strategies (markdown, html) "
             "preserve document structure from headings.",
        key="strategy_select",
    )

    with st.expander("GEO settings", expanded=False):
        st.caption("Optional: override intent to tailor next steps.")
        intent = st.selectbox(
            "Intent",
            options=["auto", "informational", "how_to", "commercial"],
            format_func=lambda x: {
                "auto": "Auto-detect (recommended)",
                "informational": "Informational (define/explain)",
                "how_to": "How-to (steps/procedure)",
                "commercial": "Commercial (compare/choose/buy)",
            }[x],
            help="This only changes the Action Plan suggestions; it does not change the similarity score.",
            key="geo_intent",
        )
        _ = intent

    return query, document, strategy


def render_action_buttons(query: str, document: str, strategy: str):
    """
    Render action buttons and handle their logic.

    Button behavior:
    - "Analyze Document": Runs full pipeline (chunk + embed + compare)
    - Disabled when inputs are empty
    """
    st.subheader("Actions")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Validate inputs
        query_valid = bool(query and query.strip())
        document_valid = bool(document and document.strip())
        can_analyze = query_valid and document_valid

        # Show validation hints
        if not query_valid and not document_valid:
            st.caption("Enter a query and document to begin")
        elif not query_valid:
            st.caption("Enter a query to analyze")
        elif not document_valid:
            st.caption("Paste a document to analyze")

        # Analyze button
        if st.button(
            "Analyze Document",
            disabled=not can_analyze,
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Processing..."):
                # Show progress steps
                progress = st.empty()

                progress.text("Chunking document...")
                progress.text("Generating embeddings...")
                progress.text("Computing similarity...")

                success = run_comparison(query, document, strategy)

                progress.empty()

                if success:
                    st.rerun()

    with col2:
        # Status/error messages
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        elif st.session_state.status_message:
            st.success(st.session_state.status_message)


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


def render_coverage_score(*, add_divider: bool = True):
    """Render the Concept Coverage Score (CCS) summary."""
    result = st.session_state.comparison_result
    report = st.session_state.diagnostic_report

    if not result or not report:
        return

    if add_divider:
        st.divider()

    # Concept Coverage Score - prominent display
    coverage = report.coverage
    coverage_color = get_coverage_color(coverage.score)

    st.subheader("Concept Coverage Score (CCS)")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(
            "CCS",
            f"{coverage.score_rounded}",
            help="Concept Coverage Score (0-100): How thoroughly the document expresses the target concept",
        )

    with col2:
        st.write(f"{coverage_color} **{coverage.interpretation}** concept coverage")
        if coverage.is_single_chunk:
            st.caption("⚠️ Single-chunk document - score may be less reliable")
        st.caption(
            f"Strong: {coverage.bucket_counts['strong']} | "
            f"Moderate: {coverage.bucket_counts['moderate']} | "
            f"Weak: {coverage.bucket_counts['weak']} | "
            f"Off-topic: {coverage.bucket_counts['off_topic']}"
        )
        st.progress(min(max(coverage.score / 100, 0.0), 1.0))
        st.caption("Rule of thumb: 80+ strong | 60–79 decent | 40–59 weak | <40 low")

    return


def render_similarity_metrics(*, add_divider: bool = True):
    """Render similarity metrics section."""
    result = st.session_state.comparison_result
    report = st.session_state.diagnostic_report

    if not result or not report:
        return

    coverage = report.coverage

    if add_divider:
        st.divider()

    st.subheader("Similarity Metrics")

    # Key metrics in columns
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
            help="Percentage of chunks with similarity >= 0.45",
        )

    # Additional summary stats in expander
    with st.expander("More Statistics"):
        summary = report.summary

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Similarity Distribution**")
            st.write(f"- Median: {summary.median_similarity:.3f}")
            st.write(f"- Std Dev: {summary.std_similarity:.3f}")
            st.write(f"- Min: {summary.min_similarity:.3f}")
            st.write(f"- Max: {summary.max_similarity:.3f}")

        with col2:
            st.write("**Chunk Breakdown**")
            st.write(f"- Strong (>= 0.80): {summary.chunks_strong}")
            st.write(f"- Moderate (0.65-0.80): {summary.chunks_moderate}")
            st.write(f"- Weak (0.45-0.65): {summary.chunks_weak}")
            st.write(f"- Off-topic (< 0.45): {summary.chunks_off_topic}")

        with col3:
            st.write("**CCS Calculation**")
            st.write(f"- Weighted Sum: {coverage.weighted_sum:.2f}")
            st.write(f"- Total Chunks: {coverage.total_chunks}")
            st.write(f"- Score: ({coverage.weighted_sum:.2f} / {coverage.total_chunks}) × 100 = **{coverage.score:.1f}**")


def get_priority_badge(priority: RecommendationPriority) -> str:
    """Get a badge for recommendation priority."""
    badges = {
        RecommendationPriority.HIGH: "🔴 HIGH",
        RecommendationPriority.MEDIUM: "🟡 MEDIUM",
        RecommendationPriority.LOW: "🟢 LOW",
    }
    return badges.get(priority, "")


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


def render_recommendations():
    """Render the CCS improvement recommendations section."""
    rec_report = st.session_state.recommendation_report

    if not rec_report or not rec_report.has_recommendations():
        return

    st.divider()
    st.subheader("Improvement Recommendations")

    # Summary with potential improvement
    improvement = rec_report.potential_ccs - rec_report.current_ccs
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write(rec_report.summary)

    with col2:
        st.metric(
            "Current CCS",
            f"{rec_report.current_ccs:.0f}",
            help="Current Concept Coverage Score",
        )

    with col3:
        delta = f"+{improvement:.0f}" if improvement > 0 else None
        st.metric(
            "Potential CCS",
            f"{rec_report.potential_ccs:.0f}",
            delta=delta,
            help="Estimated score after implementing recommendations",
        )

    # Quick Wins (HIGH priority)
    high_recs = rec_report.high_priority()
    if high_recs:
        with st.expander("Quick Wins (High Priority)", expanded=True):
            for rec in high_recs:
                _render_recommendation_card(rec)

    # Additional Improvements (MEDIUM/LOW priority)
    other_recs = rec_report.medium_priority() + rec_report.low_priority()
    if other_recs:
        with st.expander("Additional Improvements", expanded=False):
            for rec in other_recs:
                _render_recommendation_card(rec)


def _render_recommendation_card(rec):
    """Render a single recommendation card."""
    icon = get_rec_type_icon(rec.rec_type)
    priority_badge = get_priority_badge(rec.priority)

    with st.container():
        # Header row
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{icon} **{rec.what}**")
        with col2:
            st.write(priority_badge)

        # Why it matters
        st.caption(f"**Impact:** {rec.why}")

        # How to fix
        st.write(f"**Fix:** {rec.how}")

        # Example from strong chunks (if available)
        if rec.example_text:
            with st.expander("Example from strong content"):
                st.info(rec.example_text)

        # Affected chunks (collapsible)
        if rec.target_chunks:
            with st.expander(f"Affected chunks ({len(rec.target_chunks)})"):
                for target in rec.target_chunks:
                    color = get_similarity_color(target.similarity)
                    st.write(
                        f"**#{target.chunk_index + 1}** {color} {target.similarity:.2f} "
                        f"({target.interpretation})"
                    )
                    st.caption(target.text_preview)

        st.divider()


def get_similarity_color(score: float) -> str:
    """
    Get a color indicator for a similarity score.

    Uses simple text indicators instead of custom CSS
    for Streamlit compatibility.
    """
    if score >= 0.80:
        return "🟢"  # Strong
    elif score >= 0.65:
        return "🟡"  # Moderate
    elif score >= 0.45:
        return "🟠"  # Weak
    else:
        return "🔴"  # Off-topic


def _geo_priority_badge(priority: GeoPriority) -> str:
    if priority == GeoPriority.HIGH:
        return "🔴 HIGH"
    if priority == GeoPriority.MEDIUM:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def render_geo_action_plan(*, add_divider: bool = True):
    """Render GEO-focused next steps (actionable, editor-friendly)."""
    report = st.session_state.diagnostic_report
    if not report:
        return

    document = st.session_state.get("last_analyzed_document") or ""
    if not document.strip():
        return

    intent_override = GeoIntent(st.session_state.get("geo_intent", "auto"))
    geo = generate_geo_next_steps(report, document, intent_override=intent_override)

    if add_divider:
        st.divider()
    st.subheader("Action Plan (GEO next steps)")
    st.caption("Start here. These steps are designed for improving AI summarization/citation readiness.")
    st.write(geo.summary)

    # Quick signal chips
    sig = geo.signals
    c0, c1, c2, c3, c4, c5 = st.columns(6)
    c0.metric("Intent", geo.intent.value.replace("_", " ").title())
    c1.metric("Words", f"{sig.word_count:,}")
    c2.metric("Headings", f"{sig.h2_count + sig.h3_count}")
    c3.metric("Links", f"{sig.link_count}")
    c4.metric("FAQ", "Yes" if sig.has_faq else "No")
    c5.metric("Intro aligned", f"{sig.intro_query_term_coverage:.0%}")

    if not geo.steps:
        st.info("No action items detected. Try a stricter target topic, or analyze a longer document.")
        return

    for i, step in enumerate(geo.steps, start=1):
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{i}. {step.title}**")
            with col2:
                st.write(_geo_priority_badge(step.priority))
            with col3:
                st.caption(f"~{step.minutes} min")

            st.caption(f"**Why:** {step.why}")
            st.write(f"**Do this:** {step.how}")

            if step.examples:
                with st.expander("Template / example"):
                    st.code(step.examples, language="markdown")

            if step.target_chunks:
                with st.expander(f"Where to edit ({len(step.target_chunks)})"):
                    for chunk in step.target_chunks:
                        color = get_similarity_color(chunk.similarity)
                        st.write(
                            f"**#{chunk.chunk_index + 1}** {color} {chunk.similarity:.2f} "
                            f"({chunk.interpretation})"
                        )
                        st.caption(chunk.text_preview)

            st.divider()


def render_chunk_diagnostics():
    """Render the chunk-level diagnostics section."""
    report = st.session_state.diagnostic_report

    if not report:
        return

    st.divider()
    st.subheader("Chunk Analysis")
    st.caption("Tip: start with the Action Plan tab, then use this section to find exactly where to edit.")

    # Sorting controls
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
        # Level filter for hierarchical mode
        if report.is_hierarchical():
            level_filter = st.selectbox(
                "Filter level",
                options=["All", "MACRO", "MICRO", "ATOMIC"],
                key="level_filter",
            )
        else:
            level_filter = "All"

    # Get sorted chunks
    if sort_option == "Document Order":
        chunks = report.by_document_order()
    elif sort_option == "Similarity (High to Low)":
        chunks = report.by_similarity_descending()
    else:
        chunks = report.by_similarity_ascending()

    # Apply level filter
    if level_filter != "All":
        level_enum = ChunkLevel(level_filter.lower())
        chunks = [c for c in chunks if c.level == level_enum]

    # Render chunk table
    st.write("")  # Spacing

    for chunk in chunks:
        # Create a container for each chunk
        with st.container():
            # Adjust columns based on hierarchical mode
            if report.is_hierarchical():
                col1, col2, col3, col4 = st.columns([0.5, 0.8, 1, 3.7])
            else:
                col1, col2, col3 = st.columns([0.5, 1, 4])
                col4 = None

            with col1:
                # Chunk index
                st.write(f"**#{chunk.chunk_index + 1}**")

            with col2:
                # Similarity with color indicator
                color = get_similarity_color(chunk.similarity)
                st.write(f"{color} **{chunk.similarity:.3f}**")
                st.caption(chunk.interpretation)

            if col4 is not None:
                with col3:
                    # Hierarchy info
                    level_badge = get_level_badge(chunk.level)
                    st.write(level_badge)
                    if chunk.heading:
                        st.caption(chunk.heading[:20] + "..." if len(chunk.heading) > 20 else chunk.heading)

                with col4:
                    # Chunk text
                    if show_full_text:
                        st.write(chunk.text)
                    else:
                        st.write(chunk.text_preview)
            else:
                with col3:
                    # Chunk text
                    if show_full_text:
                        st.write(chunk.text)
                    else:
                        st.write(chunk.text_preview)

            st.divider()


def render_best_worst_chunks():
    """Render quick-access to best and worst chunks."""
    report = st.session_state.diagnostic_report

    if not report or report.summary.total_chunks < 2:
        return

    with st.expander("Best & Worst Chunks"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Highest Similarity**")
            best = report.get_max_chunk()
            if best:
                st.write(f"Chunk #{best.chunk_index + 1}: **{best.similarity:.3f}**")
                st.caption(best.text_preview)

        with col2:
            st.write("**Lowest Similarity**")
            worst = report.get_min_chunk()
            if worst:
                st.write(f"Chunk #{worst.chunk_index + 1}: **{worst.similarity:.3f}**")
                st.caption(worst.text_preview)


def get_level_badge(level: ChunkLevel) -> str:
    """Get a badge/indicator for chunk level."""
    badges = {
        ChunkLevel.MACRO: "🔷 MACRO",
        ChunkLevel.MICRO: "🔹 MICRO",
        ChunkLevel.ATOMIC: "⬜ ATOMIC",
        ChunkLevel.FLAT: "📄 FLAT",
    }
    return badges.get(level, "📄")


def render_section_analysis():
    """Render section-level analysis for hierarchical chunks."""
    report = st.session_state.diagnostic_report

    if not report or not report.is_hierarchical():
        return

    st.divider()
    st.subheader("Section Analysis")
    st.caption("Aggregated statistics for document sections (hierarchical mode)")

    sections = report.get_sections()

    if not sections:
        st.info("No section structure detected in the document.")
        return

    for section in sections:
        level_badge = get_level_badge(section.level)
        heading_text = section.heading or "(No heading)"
        coverage_color = get_coverage_color(section.coverage_score)

        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.write(f"{level_badge} **{heading_text}**")
                st.caption(f"{section.chunk_count} chunks, {section.total_words} words")

            with col2:
                st.metric(
                    "Avg Sim",
                    f"{section.avg_similarity:.2f}",
                    help="Average similarity across section chunks",
                )

            with col3:
                st.metric(
                    "Max",
                    f"{section.max_similarity:.2f}",
                    help="Highest similarity in section",
                )

            with col4:
                st.write(f"{coverage_color} **{section.coverage_score:.0f}**")
                st.caption("Coverage")

            st.divider()


def render_debug_panel():
    """Render optional debug information."""
    result = st.session_state.comparison_result
    report = st.session_state.diagnostic_report

    if not result:
        return

    with st.expander("Debug Info"):
        st.write("**Comparison Result**")
        st.write(f"- Query: `{result.query}`")
        st.write(f"- Model: `{result.model_name}`")
        st.write(f"- Embedding Dim: {result.embedding_dim}")
        st.write(f"- Document Chars: {result.document_char_count:,}")
        st.write(f"- Document Tokens: {result.document_token_count:,}")

        if report:
            st.write("")
            if report.is_hierarchical():
                st.write("**Hierarchy Heatmap Data**")
                st.json(report.hierarchy_heatmap_data())
            else:
                st.write("**Heatmap Data (for visualization)**")
                st.json(report.get_heatmap_data())


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="SimCheck",
        page_icon="🔍",
        layout="wide",
    )

    # Initialize state
    init_session_state()

    # Render UI sections
    render_header()

    # Input section (now returns strategy too)
    query, document, strategy = render_input_section()

    # Action buttons
    render_action_buttons(query, document, strategy)

    # Results (only shown after analysis)
    if st.session_state.is_indexed:
        action_tab, diagnostics_tab = st.tabs(["Action Plan", "Diagnostics"])

        with action_tab:
            render_coverage_score(add_divider=False)
            render_geo_action_plan()
            render_recommendations()

        with diagnostics_tab:
            render_similarity_metrics(add_divider=False)
            render_best_worst_chunks()
            render_section_analysis()
            render_chunk_diagnostics()
            render_debug_panel()

    # Footer
    st.divider()
    st.caption("SimCheck v0.1.0 | Local-only semantic analysis | No data leaves your machine")


if __name__ == "__main__":
    main()
