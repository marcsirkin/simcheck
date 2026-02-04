"""Tests for GEO / AI-SEO next steps heuristics."""

from simcheck.core.models import Chunk, ChunkSimilarity, ComparisonResult, ChunkLevel
from simcheck.core.diagnostics import create_diagnostic_report
from simcheck.core.geo import extract_content_signals, generate_geo_next_steps, GeoPriority, GeoIntent, infer_intent


def make_chunk(index: int, text: str, token_count: int = 10) -> Chunk:
    return Chunk(
        index=index,
        text=text,
        char_start=index * 100,
        char_end=(index * 100) + len(text),
        token_count=token_count,
        level=ChunkLevel.FLAT,
    )


def make_chunk_similarity(index: int, text: str, similarity: float, interpretation: str) -> ChunkSimilarity:
    return ChunkSimilarity(
        chunk=make_chunk(index, text),
        similarity=similarity,
        interpretation=interpretation,
    )


def make_comparison_result(chunk_similarities: list[ChunkSimilarity], query: str) -> ComparisonResult:
    scores = [cs.similarity for cs in chunk_similarities]
    max_sim = max(scores)
    max_idx = scores.index(max_sim)
    avg_sim = sum(scores) / len(scores)
    total_chars = sum(cs.chunk.char_count for cs in chunk_similarities)
    total_tokens = sum(cs.chunk.token_count for cs in chunk_similarities)

    return ComparisonResult(
        query=query,
        document_char_count=total_chars,
        document_token_count=total_tokens,
        chunk_count=len(chunk_similarities),
        max_similarity=max_sim,
        max_similarity_chunk_index=max_idx,
        avg_similarity=avg_sim,
        chunk_similarities=chunk_similarities,
        model_name="test-model",
        embedding_dim=384,
    )


def test_extract_content_signals_basic_markdown():
    doc = """# Title

TL;DR: summary here

## What is Retrieval Augmented Generation?
RAG is a technique...

## Sources
- [Docs](https://example.com)

## FAQ
### What is RAG?
Answer.
"""
    sig = extract_content_signals(doc, query="retrieval augmented generation")
    assert sig.has_tldr is True
    assert sig.h2_count >= 3
    assert sig.link_count == 1
    assert sig.has_faq is True
    assert sig.has_sources_section is True
    assert sig.has_freshness_signals is False
    assert 0.0 <= sig.intro_query_term_coverage <= 1.0


def test_generate_geo_next_steps_front_load_and_off_topic():
    # Best chunk appears late + off-topic exists => should include HIGH front-load and rewrite steps.
    chunk_sims = [
        make_chunk_similarity(0, "Intro that never names the topic.", 0.30, "Off-topic"),
        make_chunk_similarity(1, "More filler content.", 0.40, "Off-topic"),
        make_chunk_similarity(2, "Finally: retrieval augmented generation (RAG) explained.", 0.85, "Strong"),
        make_chunk_similarity(3, "Off-topic tangent.", 0.20, "Off-topic"),
    ]
    result = make_comparison_result(chunk_sims, query="retrieval augmented generation")
    report = create_diagnostic_report(result)
    doc = " ".join(cs.chunk.text for cs in chunk_sims)

    geo = generate_geo_next_steps(report, doc, intent_override=GeoIntent.AUTO)
    assert len(geo.steps) > 0

    titles = [s.title.lower() for s in geo.steps]
    assert any("front-load" in t for t in titles)
    assert any("off-topic" in t or "drift" in t for t in titles)

    # At least one HIGH priority item should be present in this case.
    assert any(s.priority == GeoPriority.HIGH for s in geo.steps)


def test_infer_intent():
    assert infer_intent("how to choose a crm") == GeoIntent.HOW_TO
    assert infer_intent("best crm software pricing") == GeoIntent.COMMERCIAL
    assert infer_intent("what is retrieval augmented generation") == GeoIntent.INFORMATIONAL
