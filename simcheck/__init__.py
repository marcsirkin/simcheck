"""
SimCheck - Local Query-to-Document Cosine Similarity Analyzer

A tool for analyzing semantic alignment between concepts and documents
using local embeddings and cosine similarity.
"""

from simcheck.core.engine import compare_query_to_document
from simcheck.core.models import Chunk, ChunkSimilarity, ComparisonResult, ChunkLevel
from simcheck.core.chunker import ChunkingStrategy, ChunkingConfig
from simcheck.core.diagnostics import (
    create_diagnostic_report,
    ChunkDiagnostic,
    DiagnosticReport,
    DiagnosticSummary,
    SortOrder,
    SectionSummary,
    ConceptCoverageScore,
    compute_concept_coverage_score,
    COVERAGE_WEIGHTS,
)
from simcheck.core.recommendations import (
    generate_recommendations,
    Recommendation,
    RecommendationReport,
    RecommendationType,
    RecommendationPriority,
    TargetChunk,
)

__version__ = "1.1.0"
__all__ = [
    # Feature 1: Core comparison
    "compare_query_to_document",
    "Chunk",
    "ChunkSimilarity",
    "ComparisonResult",
    # Feature 2: Diagnostics
    "create_diagnostic_report",
    "ChunkDiagnostic",
    "DiagnosticReport",
    "DiagnosticSummary",
    "SortOrder",
    # Feature 4: Concept Coverage Score
    "ConceptCoverageScore",
    "compute_concept_coverage_score",
    "COVERAGE_WEIGHTS",
    # Feature 5: Hierarchical Chunking
    "ChunkLevel",
    "ChunkingStrategy",
    "ChunkingConfig",
    "SectionSummary",
    # Feature 6: Recommendations
    "generate_recommendations",
    "Recommendation",
    "RecommendationReport",
    "RecommendationType",
    "RecommendationPriority",
    "TargetChunk",
]
