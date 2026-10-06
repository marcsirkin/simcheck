"""
CCS Improvement Recommendations.

This module generates actionable recommendations for improving a document's
Concept Coverage Score (CCS). It analyzes a DiagnosticReport and produces
prioritized, specific suggestions for content improvement.

Design Principles:
- Recommendation-focused: answers "how to improve?" vs diagnostics' "what is the score?"
- Actionable: every recommendation includes specific guidance
- Prioritized: recommendations ranked by impact and effort
- Chunk-specific: ties recommendations to specific content

This module does NOT:
- Modify documents
- Recompute similarity scores
- Generate actual replacement text
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from simcheck.core.diagnostics import (
    DiagnosticReport,
    ChunkDiagnostic,
    COVERAGE_WEIGHTS,
)
from simcheck.core.models import ChunkLevel, SIMILARITY_THRESHOLDS


class RecommendationType(Enum):
    """Types of recommendations for improving CCS."""
    REWRITE_OFF_TOPIC = "rewrite_off_topic"       # Fix chunks < 0.45
    STRENGTHEN_WEAK = "strengthen_weak"            # Improve chunks 0.45-0.65
    EXPAND_STRONG = "expand_strong"                # Leverage strong patterns
    RESTRUCTURE_SECTION = "restructure_section"    # Section-level fixes (hierarchical)
    REMOVE_DILUTION = "remove_dilution"            # Remove score-diluting content


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    HIGH = "high"       # Quick win, high impact
    MEDIUM = "medium"   # Moderate effort
    LOW = "low"         # Nice-to-have


@dataclass
class TargetChunk:
    """
    Reference to a specific chunk that needs improvement.

    Attributes:
        chunk_index: Index of the chunk in the document
        text_preview: Truncated preview of the chunk text
        similarity: Current similarity score
        interpretation: Score interpretation (Strong/Moderate/Weak/Off-topic)
    """
    chunk_index: int
    text_preview: str
    similarity: float
    interpretation: str


@dataclass
class Recommendation:
    """
    A single actionable recommendation for improving CCS.

    Attributes:
        rec_type: Category of recommendation
        priority: Impact/effort priority level
        what: Description of the problem
        why: Explanation of CCS impact
        how: Actionable fix guidance
        target_chunks: Specific chunks to address
        estimated_impact: Expected improvement ("High", "Medium", "Low")
        example_text: Example from strong chunks to emulate (if available)
    """
    rec_type: RecommendationType
    priority: RecommendationPriority
    what: str
    why: str
    how: str
    target_chunks: List[TargetChunk]
    estimated_impact: str
    example_text: Optional[str] = None


@dataclass
class RecommendationReport:
    """
    Complete recommendation report for improving CCS.

    Attributes:
        recommendations: Top prioritized recommendations (up to 5)
        current_ccs: Current Concept Coverage Score
        potential_ccs: Estimated CCS after implementing recommendations
        summary: Quick summary of findings
    """
    recommendations: List[Recommendation]
    current_ccs: float
    potential_ccs: float
    summary: str

    def high_priority(self) -> List[Recommendation]:
        """Return HIGH priority recommendations."""
        return [r for r in self.recommendations
                if r.priority == RecommendationPriority.HIGH]

    def medium_priority(self) -> List[Recommendation]:
        """Return MEDIUM priority recommendations."""
        return [r for r in self.recommendations
                if r.priority == RecommendationPriority.MEDIUM]

    def low_priority(self) -> List[Recommendation]:
        """Return LOW priority recommendations."""
        return [r for r in self.recommendations
                if r.priority == RecommendationPriority.LOW]

    def has_recommendations(self) -> bool:
        """Check if there are any recommendations."""
        return len(self.recommendations) > 0


# =============================================================================
# Thresholds and Configuration
# =============================================================================

# Off-topic percentage threshold for HIGH priority rewrite recommendation
OFF_TOPIC_HIGH_PRIORITY_THRESHOLD = 0.15  # 15% off-topic triggers HIGH priority

# Section coverage threshold for restructure recommendation
SECTION_COVERAGE_THRESHOLD = 40  # Sections below 40 CCS need restructuring

# Maximum recommendations to include in final report
MAX_RECOMMENDATIONS = 5


# =============================================================================
# Analysis Functions
# =============================================================================

def _create_target_chunk(chunk: ChunkDiagnostic, preview_length: int = 60) -> TargetChunk:
    """Create a TargetChunk reference from a ChunkDiagnostic."""
    preview = chunk.text[:preview_length]
    if len(chunk.text) > preview_length:
        preview = preview.rstrip() + "..."

    return TargetChunk(
        chunk_index=chunk.chunk_index,
        text_preview=preview,
        similarity=chunk.similarity,
        interpretation=chunk.interpretation,
    )


def _analyze_off_topic_chunks(report: DiagnosticReport) -> List[Recommendation]:
    """
    Analyze off-topic chunks (similarity < 0.45) and generate recommendations.

    Off-topic chunks contribute nothing to CCS. Rewriting them to be on-topic
    can dramatically improve scores.
    """
    recommendations = []
    off_topic = report.off_topic_chunks()

    if not off_topic:
        return recommendations

    total_chunks = report.summary.total_chunks
    off_topic_percent = len(off_topic) / total_chunks if total_chunks > 0 else 0

    # Determine priority based on percentage of off-topic content
    if off_topic_percent > OFF_TOPIC_HIGH_PRIORITY_THRESHOLD:
        priority = RecommendationPriority.HIGH
        impact = "High"
    else:
        priority = RecommendationPriority.MEDIUM
        impact = "Medium"

    # Create target chunk references
    target_chunks = [_create_target_chunk(c) for c in off_topic]

    # Build recommendation
    if len(off_topic) == 1:
        what = "Review 1 lower-alignment chunk"
    else:
        what = f"Review {len(off_topic)} lower-alignment chunks"

    why = (
        f"The provisional raw-cosine bands place {off_topic_percent:.0%} of the page below "
        "the current weak threshold. That may reflect useful context, a target mismatch, or genuine drift."
    )

    how = (
        "Decide whether each section serves the page's audience and purpose. Keep useful context; "
        "otherwise clarify the connection, condense it, move it, or remove it. Do not add query "
        "terms solely to change the score."
    )

    recommendations.append(Recommendation(
        rec_type=RecommendationType.REWRITE_OFF_TOPIC,
        priority=priority,
        what=what,
        why=why,
        how=how,
        target_chunks=target_chunks,
        estimated_impact=impact,
        example_text=None,  # Will be filled in later if strong chunks exist
    ))

    return recommendations


def _analyze_weak_chunks(report: DiagnosticReport) -> List[Recommendation]:
    """
    Analyze weak chunks (similarity 0.45-0.65) and generate recommendations.

    Weak chunks only contribute 20% of their potential weight to CCS.
    Strengthening them can provide moderate improvements.
    """
    recommendations = []

    # Get weak chunks (between the weak and moderate thresholds)
    thresholds = report.effective_thresholds
    weak_threshold = thresholds["weak"]
    moderate_threshold = thresholds["moderate"]
    weak_chunks = [c for c in report.chunks
                   if weak_threshold <= c.similarity < moderate_threshold]

    if not weak_chunks:
        return recommendations

    total_chunks = report.summary.total_chunks
    weak_percent = len(weak_chunks) / total_chunks if total_chunks > 0 else 0

    # Weak chunks are MEDIUM priority by default
    priority = RecommendationPriority.MEDIUM
    impact = "Medium"

    target_chunks = [_create_target_chunk(c) for c in weak_chunks]

    band = f"{weak_threshold:.2f}-{moderate_threshold:.2f}"
    if len(weak_chunks) == 1:
        what = f"Review 1 partially aligned chunk ({band}, provisional)"
    else:
        what = f"Review {len(weak_chunks)} partially aligned chunks ({band}, provisional)"

    why = (
        "These sections sit in a provisional middle band. The score can flag where to look, "
        "but it cannot tell whether breadth or indirect language is editorially appropriate."
    )

    how = (
        "Where the meaning is genuinely vague, consider adding concrete entities, constraints, evidence, "
        "or examples. Preserve natural language and voice; leave the section alone when it already does its job."
    )

    recommendations.append(Recommendation(
        rec_type=RecommendationType.STRENGTHEN_WEAK,
        priority=priority,
        what=what,
        why=why,
        how=how,
        target_chunks=target_chunks,
        estimated_impact=impact,
        example_text=None,
    ))

    return recommendations


def _analyze_strong_patterns(report: DiagnosticReport) -> List[Recommendation]:
    """
    Analyze strong chunks and provide them as templates for improvement.

    When strong chunks exist, they serve as examples of effective content
    that can guide improvements to weaker sections.
    """
    recommendations = []
    strong_chunks = report.strong_chunks()

    if not strong_chunks:
        return recommendations

    # Only provide this recommendation if there are also weak/off-topic chunks to improve
    weak_or_off = report.weak_or_off_topic_chunks()
    if not weak_or_off:
        return recommendations

    # This is a LOW-MEDIUM priority "template" recommendation
    priority = RecommendationPriority.LOW
    impact = "Low"

    # Get the best strong chunk as an example
    best_strong = max(strong_chunks, key=lambda c: c.similarity)
    example_preview = best_strong.text[:200]
    if len(best_strong.text) > 200:
        example_preview = example_preview.rstrip() + "..."

    # Target chunks are the weak ones that could learn from strong patterns
    target_chunks = [_create_target_chunk(c) for c in weak_or_off[:5]]  # Limit to 5

    what = f"Compare with the page's {len(strong_chunks)} higher-alignment chunks"

    why = (
        "The page's higher-scoring passages can reveal which concrete details or framing distinguish them, "
        "without treating their wording as a template."
    )

    how = (
        "Compare the passages for specificity, scope, and reader usefulness. Borrow an underlying technique "
        "only when it improves the lower-alignment section and still sounds like the same author."
    )

    recommendations.append(Recommendation(
        rec_type=RecommendationType.EXPAND_STRONG,
        priority=priority,
        what=what,
        why=why,
        how=how,
        target_chunks=target_chunks,
        estimated_impact=impact,
        example_text=example_preview,
    ))

    return recommendations


def _analyze_sections(report: DiagnosticReport) -> List[Recommendation]:
    """
    Analyze section-level coverage for hierarchical documents.

    Identifies sections with low coverage that may need restructuring
    rather than just individual chunk improvements.
    """
    recommendations = []

    if not report.is_hierarchical():
        return recommendations

    sections = report.get_sections()
    if not sections:
        return recommendations

    # Find sections with low coverage
    low_sections = [s for s in sections if s.coverage_score < SECTION_COVERAGE_THRESHOLD]

    if not low_sections:
        return recommendations

    priority = RecommendationPriority.MEDIUM
    impact = "Medium"

    # Create target chunks for the section headers
    target_chunks = []
    for section in low_sections:
        section_chunk = report.get_chunk(section.section_index)
        if section_chunk:
            target_chunks.append(_create_target_chunk(section_chunk))

    if len(low_sections) == 1:
        section = low_sections[0]
        heading = section.heading or f"Section {section.section_index + 1}"
        what = f"Section '{heading}' has low concept coverage ({section.coverage_score:.0f})"
    else:
        what = f"{len(low_sections)} sections have coverage below {SECTION_COVERAGE_THRESHOLD}"

    why = (
        "The section has lower weighted alignment with the supplied target. It may be intentionally broader, "
        "or the target may not describe the whole page."
    )

    how = (
        "Review the section as a unit. Keep it if it serves the page's purpose; otherwise clarify its role, "
        "tighten it, or move it rather than editing isolated sentences to chase the score."
    )

    recommendations.append(Recommendation(
        rec_type=RecommendationType.RESTRUCTURE_SECTION,
        priority=priority,
        what=what,
        why=why,
        how=how,
        target_chunks=target_chunks,
        estimated_impact=impact,
        example_text=None,
    ))

    return recommendations


def _analyze_dilution(report: DiagnosticReport) -> List[Recommendation]:
    """
    Analyze score dilution from off-topic content despite strong chunks.

    When a document has strong content but also significant off-topic content,
    the off-topic sections dilute the overall score.
    """
    recommendations = []

    strong_count = report.summary.chunks_strong
    off_topic_count = report.summary.chunks_off_topic
    total_chunks = report.summary.total_chunks

    # Only relevant if there are both strong and off-topic chunks
    if strong_count == 0 or off_topic_count == 0:
        return recommendations

    # Calculate dilution: significant off-topic despite good strong content
    strong_percent = strong_count / total_chunks if total_chunks > 0 else 0
    off_topic_percent = off_topic_count / total_chunks if total_chunks > 0 else 0

    # Dilution is significant if there's decent strong content but also lots of off-topic
    if strong_percent < 0.20 or off_topic_percent < 0.20:
        return recommendations

    priority = RecommendationPriority.LOW
    impact = "Low"

    off_topic = report.off_topic_chunks()
    target_chunks = [_create_target_chunk(c) for c in off_topic[:5]]

    what = (
        f"The page mixes {strong_count} higher-alignment chunks with "
        f"{off_topic_count} lower-alignment chunks"
    )

    why = (
        f"The provisional bands show a split pattern: {strong_percent:.0%} in the highest band and "
        f"{off_topic_percent:.0%} below the weak threshold. That is a prompt to inspect page scope, not "
        "automatic evidence that content should be removed."
    )

    how = (
        "Check whether the page is intentionally serving more than one reader need. If the scope feels unfocused, "
        "consider clearer section framing, a separate page, or selective condensation."
    )

    recommendations.append(Recommendation(
        rec_type=RecommendationType.REMOVE_DILUTION,
        priority=priority,
        what=what,
        why=why,
        how=how,
        target_chunks=target_chunks,
        estimated_impact=impact,
        example_text=None,
    ))

    return recommendations


def _rank_recommendations(recommendations: List[Recommendation]) -> List[Recommendation]:
    """
    Rank and filter recommendations to the top N by priority and impact.

    Priority order: HIGH > MEDIUM > LOW
    Within same priority, order by number of target chunks (more = higher impact)
    """
    # Define priority order
    priority_order = {
        RecommendationPriority.HIGH: 0,
        RecommendationPriority.MEDIUM: 1,
        RecommendationPriority.LOW: 2,
    }

    # Sort by priority, then by number of target chunks (descending)
    sorted_recs = sorted(
        recommendations,
        key=lambda r: (priority_order[r.priority], -len(r.target_chunks))
    )

    return sorted_recs[:MAX_RECOMMENDATIONS]


def _estimate_potential_ccs(
    report: DiagnosticReport,
    recommendations: List[Recommendation]
) -> float:
    """
    Estimate potential CCS after implementing recommendations.

    This is a rough estimate based on:
    - Off-topic chunks moving to weak (0.45-0.65)
    - Weak chunks moving to moderate (0.65-0.80)

    The estimate is intentionally conservative.
    """
    current_counts = report.coverage.bucket_counts.copy()

    # Estimate improvements from recommendations
    for rec in recommendations:
        if rec.rec_type == RecommendationType.REWRITE_OFF_TOPIC:
            # Assume off-topic chunks become weak (conservative)
            moved = min(len(rec.target_chunks), current_counts["off_topic"])
            current_counts["off_topic"] -= moved
            current_counts["weak"] += moved

        elif rec.rec_type == RecommendationType.STRENGTHEN_WEAK:
            # Assume weak chunks become moderate (conservative)
            moved = min(len(rec.target_chunks), current_counts["weak"])
            current_counts["weak"] -= moved
            current_counts["moderate"] += moved

    # Recalculate CCS with estimated improvements
    total = sum(current_counts.values())
    if total == 0:
        return report.coverage.score

    weighted_sum = (
        current_counts["strong"] * COVERAGE_WEIGHTS["strong"] +
        current_counts["moderate"] * COVERAGE_WEIGHTS["moderate"] +
        current_counts["weak"] * COVERAGE_WEIGHTS["weak"] +
        current_counts["off_topic"] * COVERAGE_WEIGHTS["off_topic"]
    )

    return (weighted_sum / total) * 100


def _generate_summary(
    report: DiagnosticReport,
    recommendations: List[Recommendation],
    potential_ccs: float
) -> str:
    """Generate a human-readable summary of the recommendation report."""
    current_ccs = report.coverage.score
    improvement = potential_ccs - current_ccs

    if not recommendations:
        if current_ccs >= 80:
            return "The provisional bands show consistently high alignment with the target."
        elif current_ccs >= 60:
            return "The provisional bands show generally consistent alignment with the target."
        else:
            return "Review the target and the lower-alignment passages before drawing a conclusion."

    high_count = len([r for r in recommendations if r.priority == RecommendationPriority.HIGH])
    total_count = len(recommendations)

    summary_parts = []

    if high_count > 0:
        summary_parts.append(f"{high_count} high-priority review{'s' if high_count > 1 else ''} identified")

    if improvement > 5:
        summary_parts.append(f"illustrative +{improvement:.0f} CCS scenario if flagged bands changed")

    if report.summary.chunks_off_topic > 0:
        summary_parts.append(
            f"{report.summary.chunks_off_topic} lower-alignment chunk"
            f"{'s' if report.summary.chunks_off_topic > 1 else ''} to review"
        )

    if summary_parts:
        return f"{total_count} editorial options: " + ", ".join(summary_parts) + "."
    else:
        return f"{total_count} editorial options for reviewing concept coverage."


def _add_examples_to_recommendations(
    recommendations: List[Recommendation],
    report: DiagnosticReport
) -> None:
    """Add example text from strong chunks to applicable recommendations."""
    strong_chunks = report.strong_chunks()
    if not strong_chunks:
        return

    # Get best strong chunk as example
    best_strong = max(strong_chunks, key=lambda c: c.similarity)
    example_preview = best_strong.text[:200]
    if len(best_strong.text) > 200:
        example_preview = example_preview.rstrip() + "..."

    # Add to off-topic and weak recommendations
    for rec in recommendations:
        if rec.rec_type in (RecommendationType.REWRITE_OFF_TOPIC,
                            RecommendationType.STRENGTHEN_WEAK):
            if rec.example_text is None:
                rec.example_text = example_preview


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_recommendations(report: DiagnosticReport) -> RecommendationReport:
    """
    Generate prioritized recommendations for improving CCS.

    Analyzes the diagnostic report and produces actionable recommendations
    for improving the document's Concept Coverage Score.

    Args:
        report: DiagnosticReport from create_diagnostic_report()

    Returns:
        RecommendationReport with prioritized recommendations

    Example:
        >>> from simcheck import compare_query_to_document
        >>> from simcheck.core.diagnostics import create_diagnostic_report
        >>> from simcheck.core.recommendations import generate_recommendations
        >>>
        >>> result = compare_query_to_document("baseball", document)
        >>> report = create_diagnostic_report(result)
        >>> recs = generate_recommendations(report)
        >>>
        >>> # View high-priority fixes
        >>> for rec in recs.high_priority():
        ...     print(f"{rec.what}")
        ...     print(f"  Fix: {rec.how}")
    """
    all_recommendations: List[Recommendation] = []

    # Analyze different aspects
    all_recommendations.extend(_analyze_off_topic_chunks(report))
    all_recommendations.extend(_analyze_weak_chunks(report))
    all_recommendations.extend(_analyze_strong_patterns(report))
    all_recommendations.extend(_analyze_sections(report))
    all_recommendations.extend(_analyze_dilution(report))

    # Rank and limit recommendations
    ranked = _rank_recommendations(all_recommendations)

    # Add examples from strong chunks
    _add_examples_to_recommendations(ranked, report)

    # Estimate potential improvement
    potential_ccs = _estimate_potential_ccs(report, ranked)

    # Generate summary
    summary = _generate_summary(report, ranked, potential_ccs)

    return RecommendationReport(
        recommendations=ranked,
        current_ccs=report.coverage.score,
        potential_ccs=potential_ccs,
        summary=summary,
    )
