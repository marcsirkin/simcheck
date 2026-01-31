"""
Chunk-level semantic diagnostics and inspection.

This module transforms Feature 1 outputs (ComparisonResult) into
inspectable, sortable, filterable diagnostic structures. It provides
the interpretability layer that answers "why does the document score
this way?" rather than just "what is the score?"

Feature 4 adds Concept Coverage Score (CCS): a normalized 0-100 score
that quantifies how thoroughly a document expresses a target concept,
based on weighted chunk-level similarity buckets.

Design Principles:
- Pure transformation: no recomputation of embeddings or similarity
- Compositional: builds on top of ComparisonResult
- Inspection-first: expose all data needed to understand behavior
- Heatmap-ready: provide normalized values for visualization

This module does NOT:
- Render UI or visualizations
- Recompute embeddings or similarity scores
- Persist data
- Add new similarity metrics
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum

from simcheck.core.models import (
    ComparisonResult,
    ChunkSimilarity,
    Chunk,
    SIMILARITY_THRESHOLDS,
)


class SortOrder(Enum):
    """Sort order for chunk diagnostics."""
    DOCUMENT_ORDER = "document_order"  # Original position in document
    SIMILARITY_DESC = "similarity_desc"  # Highest similarity first
    SIMILARITY_ASC = "similarity_asc"   # Lowest similarity first


# =============================================================================
# Concept Coverage Score (Feature 4)
# =============================================================================

# Weights for each similarity bucket when computing Concept Coverage Score.
# These weights reflect how much each bucket contributes to "meaningful"
# concept expression. Off-topic chunks contribute nothing.
COVERAGE_WEIGHTS = {
    "strong": 1.0,      # >= 0.80: Full contribution
    "moderate": 0.6,    # 0.65-0.80: Partial contribution
    "weak": 0.2,        # 0.45-0.65: Minimal contribution
    "off_topic": 0.0,   # < 0.45: No contribution
}

# Interpretation bands for Concept Coverage Score (0-100)
COVERAGE_INTERPRETATION = {
    "strong": 80,       # 80-100: Strong concept coverage
    "moderate": 60,     # 60-79: Moderate coverage
    "weak": 40,         # 40-59: Weak coverage
    # < 40: Low coverage
}


@dataclass
class ConceptCoverageScore:
    """
    Document-level score measuring concept coverage (0-100).

    The Concept Coverage Score (CCS) answers:
    "How much of this document meaningfully expresses the target concept?"

    It is computed by weighting chunks based on their similarity bucket:
    - Strong (>= 0.80): weight 1.0
    - Moderate (0.65-0.80): weight 0.6
    - Weak (0.45-0.65): weight 0.2
    - Off-topic (< 0.45): weight 0.0

    Formula: CCS = (sum of weighted chunks / total chunks) * 100

    Attributes:
        score: The coverage score (0-100)
        interpretation: Human-readable interpretation
        bucket_counts: Count of chunks in each bucket
        bucket_weights: Weights used for each bucket
        weighted_sum: Sum of weighted chunk contributions
        is_single_chunk: Warning flag if only one chunk
    """
    score: float
    interpretation: str
    bucket_counts: dict  # {"strong": N, "moderate": N, "weak": N, "off_topic": N}
    bucket_weights: dict  # Reference to weights used
    weighted_sum: float
    total_chunks: int
    is_single_chunk: bool

    @property
    def score_rounded(self) -> int:
        """Score rounded to nearest integer for display."""
        return round(self.score)


def interpret_coverage_score(score: float) -> str:
    """
    Convert a coverage score to a human-readable interpretation.

    Args:
        score: Concept Coverage Score (0-100)

    Returns:
        Interpretation string
    """
    if score >= COVERAGE_INTERPRETATION["strong"]:
        return "Strong"
    elif score >= COVERAGE_INTERPRETATION["moderate"]:
        return "Moderate"
    elif score >= COVERAGE_INTERPRETATION["weak"]:
        return "Weak"
    else:
        return "Low"


def compute_concept_coverage_score(
    chunks_strong: int,
    chunks_moderate: int,
    chunks_weak: int,
    chunks_off_topic: int,
) -> ConceptCoverageScore:
    """
    Compute the Concept Coverage Score from bucket counts.

    The score measures how thoroughly a document expresses a concept
    by weighting chunks based on their semantic alignment:
    - Strong chunks (>= 0.80): full weight (1.0)
    - Moderate chunks (0.65-0.80): partial weight (0.6)
    - Weak chunks (0.45-0.65): minimal weight (0.2)
    - Off-topic chunks (< 0.45): no weight (0.0)

    Formula: CCS = (sum of weighted chunks / total chunks) * 100

    Args:
        chunks_strong: Count of chunks with similarity >= 0.80
        chunks_moderate: Count of chunks with similarity 0.65-0.80
        chunks_weak: Count of chunks with similarity 0.45-0.65
        chunks_off_topic: Count of chunks with similarity < 0.45

    Returns:
        ConceptCoverageScore with score, interpretation, and breakdown
    """
    total = chunks_strong + chunks_moderate + chunks_weak + chunks_off_topic

    if total == 0:
        return ConceptCoverageScore(
            score=0.0,
            interpretation="Low",
            bucket_counts={
                "strong": 0, "moderate": 0, "weak": 0, "off_topic": 0
            },
            bucket_weights=COVERAGE_WEIGHTS,
            weighted_sum=0.0,
            total_chunks=0,
            is_single_chunk=False,
        )

    # Compute weighted sum
    weighted_sum = (
        chunks_strong * COVERAGE_WEIGHTS["strong"] +
        chunks_moderate * COVERAGE_WEIGHTS["moderate"] +
        chunks_weak * COVERAGE_WEIGHTS["weak"] +
        chunks_off_topic * COVERAGE_WEIGHTS["off_topic"]
    )

    # Compute score (0-100)
    score = (weighted_sum / total) * 100

    return ConceptCoverageScore(
        score=score,
        interpretation=interpret_coverage_score(score),
        bucket_counts={
            "strong": chunks_strong,
            "moderate": chunks_moderate,
            "weak": chunks_weak,
            "off_topic": chunks_off_topic,
        },
        bucket_weights=COVERAGE_WEIGHTS,
        weighted_sum=weighted_sum,
        total_chunks=total,
        is_single_chunk=(total == 1),
    )


@dataclass
class ChunkDiagnostic:
    """
    Enhanced chunk information for diagnostic inspection.

    This wraps a ChunkSimilarity with additional metadata useful
    for understanding why a chunk scored the way it did.

    Attributes:
        chunk_index: Original position in document (0-indexed)
        text: The chunk's text content
        text_preview: Truncated text for display (first N chars)
        char_count: Number of characters in chunk
        token_count: Estimated token count
        similarity: Raw cosine similarity score
        interpretation: Human-readable interpretation (Strong/Moderate/Weak/Off-topic)
        similarity_rank: Rank by similarity (1 = highest scoring chunk)
        normalized_score: Score normalized to 0-1 range for heatmap rendering
        position_percent: Position in document as percentage (0.0 = start, 1.0 = end)
        is_max: Whether this is the highest-scoring chunk
        is_min: Whether this is the lowest-scoring chunk
        above_strong_threshold: Score >= strong threshold (0.80)
        above_moderate_threshold: Score >= moderate threshold (0.65)
        below_weak_threshold: Score < weak threshold (0.45)
    """
    # Core identification
    chunk_index: int
    text: str
    text_preview: str
    char_count: int
    token_count: int

    # Similarity data
    similarity: float
    interpretation: str
    similarity_rank: int  # 1 = highest

    # Normalized/derived metrics
    normalized_score: float  # 0-1 for heatmap
    position_percent: float  # 0-1 position in document

    # Flags
    is_max: bool
    is_min: bool
    above_strong_threshold: bool
    above_moderate_threshold: bool
    below_weak_threshold: bool


@dataclass
class DiagnosticSummary:
    """
    Aggregate statistics for the diagnostic report.

    Provides quick-glance metrics without iterating through all chunks.
    """
    total_chunks: int
    max_similarity: float
    min_similarity: float
    avg_similarity: float
    median_similarity: float
    std_similarity: float  # Standard deviation

    # Threshold counts
    chunks_strong: int      # >= 0.80
    chunks_moderate: int    # >= 0.65 and < 0.80
    chunks_weak: int        # >= 0.45 and < 0.65
    chunks_off_topic: int   # < 0.45

    # Percentages
    percent_strong: float
    percent_on_topic: float  # >= weak threshold (0.45)

    # Document coverage
    document_char_count: int
    document_token_count: int


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report for a query-document comparison.

    This is the primary output of Feature 2. It transforms a ComparisonResult
    into an inspectable, sortable, filterable structure for understanding
    semantic alignment at the chunk level.

    Feature 4 adds the Concept Coverage Score, which quantifies how thoroughly
    the document expresses the target concept.

    The report maintains chunks in document order by default but provides
    methods for alternative views (sorted by similarity, filtered by threshold).

    Attributes:
        query: The original query text
        model_name: Embedding model used
        chunks: All chunk diagnostics in document order
        summary: Aggregate statistics
        coverage: Concept Coverage Score (Feature 4)
    """
    query: str
    model_name: str
    chunks: List[ChunkDiagnostic]
    summary: DiagnosticSummary
    coverage: ConceptCoverageScore

    # -------------------------------------------------------------------------
    # Sorting methods - return new lists, don't mutate
    # -------------------------------------------------------------------------

    def by_document_order(self) -> List[ChunkDiagnostic]:
        """Return chunks in original document order."""
        return sorted(self.chunks, key=lambda c: c.chunk_index)

    def by_similarity_descending(self) -> List[ChunkDiagnostic]:
        """Return chunks sorted by similarity, highest first."""
        return sorted(self.chunks, key=lambda c: c.similarity, reverse=True)

    def by_similarity_ascending(self) -> List[ChunkDiagnostic]:
        """Return chunks sorted by similarity, lowest first."""
        return sorted(self.chunks, key=lambda c: c.similarity)

    def sorted_by(self, order: SortOrder) -> List[ChunkDiagnostic]:
        """Return chunks sorted by specified order."""
        if order == SortOrder.DOCUMENT_ORDER:
            return self.by_document_order()
        elif order == SortOrder.SIMILARITY_DESC:
            return self.by_similarity_descending()
        elif order == SortOrder.SIMILARITY_ASC:
            return self.by_similarity_ascending()
        else:
            raise ValueError(f"Unknown sort order: {order}")

    # -------------------------------------------------------------------------
    # Filtering methods - return new lists, don't mutate
    # -------------------------------------------------------------------------

    def above_threshold(self, threshold: float) -> List[ChunkDiagnostic]:
        """Return chunks with similarity >= threshold."""
        return [c for c in self.chunks if c.similarity >= threshold]

    def below_threshold(self, threshold: float) -> List[ChunkDiagnostic]:
        """Return chunks with similarity < threshold."""
        return [c for c in self.chunks if c.similarity < threshold]

    def in_range(self, min_sim: float, max_sim: float) -> List[ChunkDiagnostic]:
        """Return chunks with similarity in [min_sim, max_sim]."""
        return [c for c in self.chunks if min_sim <= c.similarity <= max_sim]

    def strong_chunks(self) -> List[ChunkDiagnostic]:
        """Return chunks with Strong interpretation (>= 0.80)."""
        return self.above_threshold(SIMILARITY_THRESHOLDS["strong"])

    def weak_or_off_topic_chunks(self) -> List[ChunkDiagnostic]:
        """Return chunks below moderate threshold (< 0.65)."""
        return self.below_threshold(SIMILARITY_THRESHOLDS["moderate"])

    def off_topic_chunks(self) -> List[ChunkDiagnostic]:
        """Return chunks below weak threshold (< 0.45)."""
        return self.below_threshold(SIMILARITY_THRESHOLDS["weak"])

    def filter_by(
        self,
        predicate: Callable[[ChunkDiagnostic], bool]
    ) -> List[ChunkDiagnostic]:
        """Return chunks matching a custom predicate function."""
        return [c for c in self.chunks if predicate(c)]

    # -------------------------------------------------------------------------
    # Convenience accessors
    # -------------------------------------------------------------------------

    def get_max_chunk(self) -> Optional[ChunkDiagnostic]:
        """Return the highest-scoring chunk."""
        for chunk in self.chunks:
            if chunk.is_max:
                return chunk
        return None

    def get_min_chunk(self) -> Optional[ChunkDiagnostic]:
        """Return the lowest-scoring chunk."""
        for chunk in self.chunks:
            if chunk.is_min:
                return chunk
        return None

    def get_chunk(self, index: int) -> Optional[ChunkDiagnostic]:
        """Return chunk by document index, or None if not found."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None

    def top_n(self, n: int) -> List[ChunkDiagnostic]:
        """Return top N chunks by similarity."""
        return self.by_similarity_descending()[:n]

    def bottom_n(self, n: int) -> List[ChunkDiagnostic]:
        """Return bottom N chunks by similarity."""
        return self.by_similarity_ascending()[:n]

    # -------------------------------------------------------------------------
    # Heatmap data extraction
    # -------------------------------------------------------------------------

    def get_heatmap_data(self) -> List[dict]:
        """
        Return chunk data formatted for heatmap rendering.

        Returns list of dicts in document order with:
        - index: chunk position
        - score: normalized 0-1 score
        - raw_score: original similarity
        - interpretation: text label
        """
        return [
            {
                "index": c.chunk_index,
                "score": c.normalized_score,
                "raw_score": c.similarity,
                "interpretation": c.interpretation,
            }
            for c in self.by_document_order()
        ]

    def get_similarity_sequence(self) -> List[float]:
        """Return list of similarity scores in document order."""
        return [c.similarity for c in self.by_document_order()]

    def get_normalized_sequence(self) -> List[float]:
        """Return list of normalized scores (0-1) in document order."""
        return [c.normalized_score for c in self.by_document_order()]


# -----------------------------------------------------------------------------
# Factory function - main entry point
# -----------------------------------------------------------------------------

def create_diagnostic_report(
    result: ComparisonResult,
    preview_length: int = 80,
) -> DiagnosticReport:
    """
    Transform a ComparisonResult into a DiagnosticReport.

    This is the main entry point for Feature 2. It takes the output
    from compare_query_to_document() and creates an inspectable
    diagnostic structure.

    Args:
        result: ComparisonResult from Feature 1
        preview_length: Max characters for text_preview field

    Returns:
        DiagnosticReport with full diagnostic data

    Example:
        >>> from simcheck import compare_query_to_document
        >>> from simcheck.core.diagnostics import create_diagnostic_report
        >>>
        >>> result = compare_query_to_document("baseball", document)
        >>> report = create_diagnostic_report(result)
        >>>
        >>> # Inspect highest-scoring chunks
        >>> for chunk in report.top_n(3):
        ...     print(f"Chunk {chunk.chunk_index}: {chunk.similarity:.2f}")
        >>>
        >>> # Find off-topic sections
        >>> for chunk in report.off_topic_chunks():
        ...     print(f"Off-topic: {chunk.text_preview}")
    """
    if not result.chunk_similarities:
        raise ValueError("ComparisonResult has no chunk similarities")

    # Extract raw similarity scores for statistics
    scores = [cs.similarity for cs in result.chunk_similarities]

    # Compute ranks (1 = highest similarity)
    # Create list of (index, score) tuples, sort by score descending
    indexed_scores = [(i, s) for i, s in enumerate(scores)]
    sorted_by_score = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    # Map chunk index -> rank
    rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_by_score)}

    # Compute normalized scores for heatmap (0-1 range)
    # Normalize relative to min/max in this document
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    def normalize_score(score: float) -> float:
        """Normalize score to 0-1 range within this document."""
        if score_range == 0:
            # All scores are identical
            return 0.5
        return (score - min_score) / score_range

    # Build chunk diagnostics
    total_chunks = len(result.chunk_similarities)
    chunk_diagnostics: List[ChunkDiagnostic] = []

    for cs in result.chunk_similarities:
        chunk = cs.chunk
        similarity = cs.similarity

        # Create text preview
        text_preview = chunk.text[:preview_length]
        if len(chunk.text) > preview_length:
            text_preview = text_preview.rstrip() + "..."

        diagnostic = ChunkDiagnostic(
            chunk_index=chunk.index,
            text=chunk.text,
            text_preview=text_preview,
            char_count=chunk.char_count,
            token_count=chunk.token_count,
            similarity=similarity,
            interpretation=cs.interpretation,
            similarity_rank=rank_map[chunk.index],
            normalized_score=normalize_score(similarity),
            position_percent=chunk.index / max(1, total_chunks - 1) if total_chunks > 1 else 0.0,
            is_max=(similarity == max_score),
            is_min=(similarity == min_score),
            above_strong_threshold=(similarity >= SIMILARITY_THRESHOLDS["strong"]),
            above_moderate_threshold=(similarity >= SIMILARITY_THRESHOLDS["moderate"]),
            below_weak_threshold=(similarity < SIMILARITY_THRESHOLDS["weak"]),
        )
        chunk_diagnostics.append(diagnostic)

    # Build summary statistics
    summary = _compute_summary(result, scores)

    # Compute Concept Coverage Score (Feature 4)
    coverage = compute_concept_coverage_score(
        chunks_strong=summary.chunks_strong,
        chunks_moderate=summary.chunks_moderate,
        chunks_weak=summary.chunks_weak,
        chunks_off_topic=summary.chunks_off_topic,
    )

    return DiagnosticReport(
        query=result.query,
        model_name=result.model_name,
        chunks=chunk_diagnostics,
        summary=summary,
        coverage=coverage,
    )


def _compute_summary(result: ComparisonResult, scores: List[float]) -> DiagnosticSummary:
    """Compute aggregate statistics for the diagnostic summary."""
    import statistics

    total = len(scores)

    # Basic stats
    max_sim = max(scores)
    min_sim = min(scores)
    avg_sim = statistics.mean(scores)
    median_sim = statistics.median(scores)
    std_sim = statistics.stdev(scores) if total > 1 else 0.0

    # Threshold counts
    strong = SIMILARITY_THRESHOLDS["strong"]
    moderate = SIMILARITY_THRESHOLDS["moderate"]
    weak = SIMILARITY_THRESHOLDS["weak"]

    chunks_strong = sum(1 for s in scores if s >= strong)
    chunks_moderate = sum(1 for s in scores if moderate <= s < strong)
    chunks_weak = sum(1 for s in scores if weak <= s < moderate)
    chunks_off_topic = sum(1 for s in scores if s < weak)

    # Percentages
    percent_strong = (chunks_strong / total) * 100 if total > 0 else 0
    percent_on_topic = ((total - chunks_off_topic) / total) * 100 if total > 0 else 0

    return DiagnosticSummary(
        total_chunks=total,
        max_similarity=max_sim,
        min_similarity=min_sim,
        avg_similarity=avg_sim,
        median_similarity=median_sim,
        std_similarity=std_sim,
        chunks_strong=chunks_strong,
        chunks_moderate=chunks_moderate,
        chunks_weak=chunks_weak,
        chunks_off_topic=chunks_off_topic,
        percent_strong=percent_strong,
        percent_on_topic=percent_on_topic,
        document_char_count=result.document_char_count,
        document_token_count=result.document_token_count,
    )
