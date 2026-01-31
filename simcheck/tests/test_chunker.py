"""Tests for document chunking logic."""

import pytest
from simcheck.core.chunker import (
    chunk_document,
    get_total_tokens,
    ChunkingError,
    _estimate_tokens,
    _split_into_sentences,
    DEFAULT_CHUNK_TOKENS,
    CHARS_PER_TOKEN,
)


class TestEstimateTokens:
    """Tests for token estimation heuristic."""

    def test_empty_string(self):
        """Empty string should return 1 (minimum)."""
        assert _estimate_tokens("") == 1

    def test_short_text(self):
        """Short text token estimation."""
        # 4 chars / 4 chars_per_token = 1 token
        assert _estimate_tokens("word") == 1

    def test_longer_text(self):
        """Longer text should estimate proportionally."""
        # 20 chars / 4 = 5 tokens
        text = "a" * 20
        assert _estimate_tokens(text) == 5

    def test_chars_per_token_constant(self):
        """CHARS_PER_TOKEN should be 4."""
        assert CHARS_PER_TOKEN == 4


class TestSplitIntoSentences:
    """Tests for sentence splitting."""

    def test_single_sentence(self):
        """Single sentence should return one item."""
        sentences = _split_into_sentences("Hello world.")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    def test_multiple_sentences(self):
        """Multiple sentences should split correctly."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_question_marks(self):
        """Should split on question marks."""
        text = "Is this a question? Yes it is."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2

    def test_exclamation_marks(self):
        """Should split on exclamation marks."""
        text = "Wow! That's amazing."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2

    def test_no_punctuation(self):
        """Text without sentence-ending punctuation."""
        text = "No punctuation here"
        sentences = _split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "No punctuation here"

    def test_empty_string(self):
        """Empty string should return empty list."""
        sentences = _split_into_sentences("")
        assert sentences == []

    def test_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        sentences = _split_into_sentences("   \n\t  ")
        assert sentences == []


class TestChunkDocument:
    """Tests for the main chunk_document function."""

    def test_empty_document_raises(self):
        """Empty document should raise ChunkingError."""
        with pytest.raises(ChunkingError, match="empty"):
            chunk_document("")

    def test_whitespace_only_raises(self):
        """Whitespace-only document should raise ChunkingError."""
        with pytest.raises(ChunkingError, match="empty"):
            chunk_document("   \n\t  ")

    def test_none_document_raises(self):
        """None document should raise error."""
        with pytest.raises((ChunkingError, TypeError)):
            chunk_document(None)

    def test_short_document_single_chunk(self):
        """Short document should produce single chunk."""
        doc = "This is a short document."
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == doc

    def test_chunks_have_sequential_indices(self):
        """Chunk indices should be sequential starting from 0."""
        # Create a document that will produce multiple chunks
        # Each sentence is ~40 chars = ~10 tokens
        # With 350 token target, need ~35 sentences to get 2 chunks
        sentences = ["This is sentence number {:03d}.".format(i) for i in range(40)]
        doc = " ".join(sentences)
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunks_preserve_order(self):
        """Chunks should be in document order."""
        sentences = ["First sentence here.", "Second sentence here.", "Third sentence here."]
        # Make each sentence appear in a different chunk by using small target
        doc = " ".join(sentences)
        chunks = chunk_document(doc, target_tokens=10)

        # First chunk should contain "First"
        assert "First" in chunks[0].text
        # Last chunk should contain "Third"
        assert "Third" in chunks[-1].text

    def test_chunk_has_token_count(self):
        """Each chunk should have a token count estimate."""
        doc = "This is a test document with some words."
        chunks = chunk_document(doc)
        assert chunks[0].token_count > 0

    def test_chunk_has_char_positions(self):
        """Each chunk should have char_start and char_end."""
        doc = "This is a test document."
        chunks = chunk_document(doc)
        assert chunks[0].char_start == 0
        assert chunks[0].char_end > 0

    def test_default_chunk_tokens(self):
        """DEFAULT_CHUNK_TOKENS should be 150 for granular analysis."""
        assert DEFAULT_CHUNK_TOKENS == 150

    def test_custom_target_tokens(self):
        """Should respect custom target_tokens parameter."""
        # Long document
        doc = "Word. " * 500  # ~500 sentences

        # Small chunks
        small_chunks = chunk_document(doc, target_tokens=50)
        # Large chunks
        large_chunks = chunk_document(doc, target_tokens=500)

        # Small target should produce more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_long_single_sentence(self):
        """Very long single sentence should become one chunk."""
        # A single sentence with no periods until the end
        doc = "word " * 200 + "end."
        chunks = chunk_document(doc)
        # Should be at least 1 chunk
        assert len(chunks) >= 1


class TestGetTotalTokens:
    """Tests for get_total_tokens utility."""

    def test_empty_list(self):
        """Empty chunk list should return 0."""
        assert get_total_tokens([]) == 0

    def test_single_chunk(self):
        """Single chunk should return its token count."""
        doc = "Test document."
        chunks = chunk_document(doc)
        total = get_total_tokens(chunks)
        assert total == chunks[0].token_count

    def test_multiple_chunks(self):
        """Should sum token counts across chunks."""
        # Create multi-chunk document
        doc = "Sentence one. " * 50 + "Sentence two. " * 50
        chunks = chunk_document(doc, target_tokens=100)

        expected = sum(c.token_count for c in chunks)
        assert get_total_tokens(chunks) == expected
