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

# Import backend modules (Features 1 + 2)
from simcheck.core.engine import compare_query_to_document, ComparisonError
from simcheck.core.diagnostics import create_diagnostic_report, SortOrder
from simcheck.core.embeddings import DEFAULT_MODEL, get_model_info


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
    """
    defaults = {
        "document_text": "",
        "query_text": "",
        "is_indexed": False,
        "comparison_result": None,
        "diagnostic_report": None,
        "status_message": "",
        "error_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results():
    """Clear comparison results when document changes."""
    st.session_state.is_indexed = False
    st.session_state.comparison_result = None
    st.session_state.diagnostic_report = None
    st.session_state.status_message = ""
    st.session_state.error_message = ""


# =============================================================================
# Backend Integration (calls Features 1 + 2)
# =============================================================================

def run_comparison(query: str, document: str) -> bool:
    """
    Run the full comparison pipeline.

    Calls Feature 1 (compare_query_to_document) and Feature 2
    (create_diagnostic_report). Updates session state with results.

    Returns True on success, False on error.
    """
    try:
        st.session_state.error_message = ""

        # Feature 1: Run comparison
        result = compare_query_to_document(query, document)
        st.session_state.comparison_result = result

        # Feature 2: Create diagnostic report
        report = create_diagnostic_report(result)
        st.session_state.diagnostic_report = report

        st.session_state.is_indexed = True
        st.session_state.status_message = f"Analysis complete. {result.chunk_count} chunks processed."
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
    st.caption("Local Query-to-Document Cosine Similarity Analyzer")

    # Show model info
    try:
        model_info = get_model_info(DEFAULT_MODEL)
        st.caption(f"Model: `{model_info['model_name']}` ({model_info['embedding_dim']} dims)")
    except Exception:
        st.caption(f"Model: `{DEFAULT_MODEL}`")


def render_input_section():
    """
    Render the input section with query and document text areas.

    Returns tuple of (query, document) strings.
    """
    st.subheader("Inputs")

    # Query input
    query = st.text_input(
        "Concept / Query",
        placeholder="e.g., Major League Baseball",
        help="Enter a short concept or phrase to analyze",
        key="query_input",
    )

    # Document input
    document = st.text_area(
        "Document Text",
        placeholder="Paste your document text here...",
        height=200,
        help="Paste the document you want to analyze",
        key="document_input",
    )

    # Show character/word counts
    if document:
        word_count = len(document.split())
        char_count = len(document)
        st.caption(f"{char_count:,} characters, ~{word_count:,} words")

    return query, document


def render_action_buttons(query: str, document: str):
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

                success = run_comparison(query, document)

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


def render_summary_results():
    """Render the summary results section."""
    result = st.session_state.comparison_result
    report = st.session_state.diagnostic_report

    if not result or not report:
        return

    st.divider()

    # Concept Coverage Score - prominent display
    coverage = report.coverage
    coverage_color = get_coverage_color(coverage.score)

    st.subheader("Concept Coverage Score")
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


def render_chunk_diagnostics():
    """Render the chunk-level diagnostics section."""
    report = st.session_state.diagnostic_report

    if not report:
        return

    st.divider()
    st.subheader("Chunk Analysis")

    # Sorting controls
    col1, col2 = st.columns([2, 1])

    with col1:
        sort_option = st.radio(
            "Sort by",
            ["Document Order", "Similarity (High to Low)", "Similarity (Low to High)"],
            horizontal=True,
            key="sort_option",
        )

    with col2:
        show_full_text = st.checkbox("Show full text", value=False, key="show_full")

    # Get sorted chunks
    if sort_option == "Document Order":
        chunks = report.by_document_order()
    elif sort_option == "Similarity (High to Low)":
        chunks = report.by_similarity_descending()
    else:
        chunks = report.by_similarity_ascending()

    # Render chunk table
    st.write("")  # Spacing

    for chunk in chunks:
        # Create a container for each chunk
        with st.container():
            col1, col2, col3 = st.columns([0.5, 1, 4])

            with col1:
                # Chunk index
                st.write(f"**#{chunk.chunk_index + 1}**")

            with col2:
                # Similarity with color indicator
                color = get_similarity_color(chunk.similarity)
                st.write(f"{color} **{chunk.similarity:.3f}**")
                st.caption(chunk.interpretation)

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

    # Input section
    query, document = render_input_section()

    # Action buttons
    render_action_buttons(query, document)

    # Results (only shown after analysis)
    if st.session_state.is_indexed:
        render_summary_results()
        render_best_worst_chunks()
        render_chunk_diagnostics()
        render_debug_panel()

    # Footer
    st.divider()
    st.caption("SimCheck v0.1.0 | Local-only semantic analysis | No data leaves your machine")


if __name__ == "__main__":
    main()
