"""
SimCheck - Local Query-to-Document Cosine Similarity Analyzer

A tool for analyzing semantic alignment between concepts and documents
using local embeddings and cosine similarity.
"""

from simcheck.core.engine import compare_query_to_document
from simcheck.core.models import Chunk, ChunkSimilarity, ComparisonResult
from simcheck.core.diagnostics import (
    create_diagnostic_report,
    ChunkDiagnostic,
    DiagnosticReport,
    DiagnosticSummary,
    SortOrder,
    ConceptCoverageScore,
    compute_concept_coverage_score,
    COVERAGE_WEIGHTS,
)

__version__ = "0.1.0"
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
]
