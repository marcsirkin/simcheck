"""
Core semantic comparison engine.

This module provides the foundational logic for:
- Document chunking
- Embedding generation
- Cosine similarity calculation
- Query-to-document comparison
- Chunk-level diagnostics and inspection (Feature 2)
"""

from simcheck.core.chunker import chunk_document
from simcheck.core.embeddings import embed_text, embed_texts
from simcheck.core.similarity import cosine_similarity
from simcheck.core.engine import compare_query_to_document
from simcheck.core.models import Chunk, ChunkSimilarity, ComparisonResult
from simcheck.core.diagnostics import (
    create_diagnostic_report,
    ChunkDiagnostic,
    DiagnosticReport,
    DiagnosticSummary,
    SortOrder,
)

__all__ = [
    # Feature 1: Core comparison
    "chunk_document",
    "embed_text",
    "embed_texts",
    "cosine_similarity",
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
]
