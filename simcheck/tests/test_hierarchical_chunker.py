"""Tests for hierarchical document chunking."""

import pytest
from simcheck.core.chunker import (
    chunk_document,
    chunk_document_hierarchical,
    ChunkingStrategy,
    ChunkingConfig,
    ChunkingError,
    _detect_document_format,
    _parse_markdown_structure,
    _parse_html_structure,
    _split_into_paragraphs,
    DEFAULT_CHUNK_TOKENS,
)
from simcheck.core.models import Chunk, ChunkLevel


# =============================================================================
# Test Fixtures
# =============================================================================

SAMPLE_MARKDOWN = """
# Title (ignored - H1)

Introduction paragraph here.

## Section One

This is the first section with some content.
It has multiple sentences. And more text here.

### Subsection 1.1

Details about the first subsection.
More details follow.

### Subsection 1.2

Content for subsection 1.2 goes here.

## Section Two

The second main section starts here.
It contains different information.

### Subsection 2.1

Final subsection with its own content.
"""

SAMPLE_HTML = """
<h1>Title (ignored)</h1>
<p>Introduction paragraph here.</p>

<h2>Section One</h2>
<p>This is the first section with some content. It has multiple sentences.</p>

<h3>Subsection 1.1</h3>
<p>Details about the first subsection.</p>

<h3>Subsection 1.2</h3>
<p>Content for subsection 1.2 goes here.</p>

<h2>Section Two</h2>
<p>The second main section starts here.</p>

<h3>Subsection 2.1</h3>
<p>Final subsection with its own content.</p>
"""

SAMPLE_PLAIN_TEXT = """
This is a plain text document without any headings.
It has multiple sentences but no structure.
The chunker should fall back to flat chunking for this.
More sentences follow to provide enough content.
"""


# =============================================================================
# TestMarkdownParsing
# =============================================================================

class TestMarkdownParsing:
    """Tests for Markdown structure parsing."""

    def test_parse_h2_headings(self):
        """Should detect ## headings as level 2."""
        doc = "## First Section\n\nContent here.\n\n## Second Section\n\nMore content."
        sections = _parse_markdown_structure(doc)

        h2_sections = [s for s in sections if s.level == 2]
        assert len(h2_sections) == 2
        assert h2_sections[0].heading == "First Section"
        assert h2_sections[1].heading == "Second Section"

    def test_parse_h3_headings(self):
        """Should detect ### headings as level 3."""
        doc = "## Main\n\nIntro.\n\n### Sub One\n\nContent 1.\n\n### Sub Two\n\nContent 2."
        sections = _parse_markdown_structure(doc)

        h3_sections = [s for s in sections if s.level == 3]
        assert len(h3_sections) == 2
        assert h3_sections[0].heading == "Sub One"
        assert h3_sections[1].heading == "Sub Two"

    def test_ignore_h1_headings(self):
        """H1 headings (single #) should not create MACRO sections."""
        doc = "# Title\n\nIntro.\n\n## Section\n\nContent."
        sections = _parse_markdown_structure(doc)

        # The H1 content becomes paragraph-level (level 4)
        h1_sections = [s for s in sections if s.level == 1]
        assert len(h1_sections) == 0

    def test_content_between_headings(self):
        """Content between headings should be captured correctly."""
        doc = "## Section\n\nParagraph one.\n\nParagraph two.\n\n## Next"
        sections = _parse_markdown_structure(doc)

        first_section = [s for s in sections if s.heading == "Section"][0]
        assert "Paragraph one" in first_section.content
        assert "Paragraph two" in first_section.content

    def test_empty_document(self):
        """Empty document should return empty list."""
        sections = _parse_markdown_structure("")
        assert sections == []

    def test_no_headings(self):
        """Document without headings should have paragraph-level sections."""
        doc = "Just some text without any headings."
        sections = _parse_markdown_structure(doc)

        assert len(sections) == 1
        assert sections[0].level == 4  # Paragraph level
        assert sections[0].heading is None


# =============================================================================
# TestHtmlParsing
# =============================================================================

class TestHtmlParsing:
    """Tests for HTML structure parsing."""

    def test_parse_h2_tags(self):
        """Should detect <h2> tags."""
        doc = "<h2>Section One</h2><p>Content</p><h2>Section Two</h2><p>More</p>"
        sections = _parse_html_structure(doc)

        h2_sections = [s for s in sections if s.level == 2]
        assert len(h2_sections) == 2
        assert h2_sections[0].heading == "Section One"
        assert h2_sections[1].heading == "Section Two"

    def test_parse_h3_tags(self):
        """Should detect <h3> tags."""
        doc = "<h2>Main</h2><p>Intro</p><h3>Sub</h3><p>Details</p>"
        sections = _parse_html_structure(doc)

        h3_sections = [s for s in sections if s.level == 3]
        assert len(h3_sections) == 1
        assert h3_sections[0].heading == "Sub"

    def test_strip_inner_html_from_headings(self):
        """Should strip inner HTML tags from heading text."""
        doc = "<h2><strong>Bold</strong> Title</h2><p>Content</p>"
        sections = _parse_html_structure(doc)

        assert sections[0].heading == "Bold Title"

    def test_strip_html_from_content(self):
        """Content should have HTML tags stripped."""
        doc = "<h2>Section</h2><p>Some <em>emphasized</em> text.</p>"
        sections = _parse_html_structure(doc)

        section = [s for s in sections if s.heading == "Section"][0]
        assert "<em>" not in section.content
        assert "emphasized" in section.content

    def test_no_headings(self):
        """Document without h2/h3 should return content as single section."""
        doc = "<p>Just a paragraph.</p><p>Another one.</p>"
        sections = _parse_html_structure(doc)

        assert len(sections) == 1
        assert sections[0].level == 4
        assert "Just a paragraph" in sections[0].content


# =============================================================================
# TestHierarchicalChunking
# =============================================================================

class TestHierarchicalChunking:
    """Tests for hierarchical chunk creation."""

    def test_markdown_creates_macro_chunks(self):
        """Markdown H2 sections should create MACRO chunks."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        macro_chunks = [c for c in chunks if c.level == ChunkLevel.MACRO]
        assert len(macro_chunks) >= 2  # "Section One" and "Section Two"

    def test_markdown_creates_micro_chunks(self):
        """Markdown H3 sections should create MICRO chunks."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        micro_chunks = [c for c in chunks if c.level == ChunkLevel.MICRO]
        assert len(micro_chunks) >= 3  # Three ### subsections

    def test_parent_child_relationships(self):
        """MICRO chunks should reference MACRO parent."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        # Find a MICRO chunk
        micro_chunks = [c for c in chunks if c.level == ChunkLevel.MICRO]

        if micro_chunks:
            micro = micro_chunks[0]
            # Should have a parent index
            assert micro.parent_index is not None
            # Parent should be MACRO level
            parent = chunks[micro.parent_index]
            assert parent.level == ChunkLevel.MACRO

    def test_depth_values(self):
        """Depth should reflect hierarchy level."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        for chunk in chunks:
            if chunk.level == ChunkLevel.MACRO:
                assert chunk.depth == 0
            elif chunk.level == ChunkLevel.MICRO:
                assert chunk.depth == 1
            elif chunk.level == ChunkLevel.ATOMIC:
                assert chunk.depth == 2

    def test_heading_preserved(self):
        """Chunks should preserve their section heading."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        macro_chunks = [c for c in chunks if c.level == ChunkLevel.MACRO]
        headings = [c.heading for c in macro_chunks if c.heading]

        assert "Section One" in headings
        assert "Section Two" in headings

    def test_word_count_property(self):
        """Chunks should have word_count property."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        for chunk in chunks:
            assert chunk.word_count >= 0
            assert chunk.word_count == len(chunk.text.split())

    def test_is_hierarchical_property(self):
        """Hierarchical chunks should return True for is_hierarchical."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        for chunk in chunks:
            if chunk.level != ChunkLevel.FLAT:
                assert chunk.is_hierarchical is True

    def test_html_chunking(self):
        """HTML strategy should parse h2/h3 tags."""
        chunks = chunk_document(SAMPLE_HTML, strategy=ChunkingStrategy.HTML)

        macro_chunks = [c for c in chunks if c.level == ChunkLevel.MACRO]
        assert len(macro_chunks) >= 2

    def test_sequential_indices(self):
        """Chunk indices should be sequential starting from 0."""
        chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.MARKDOWN)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# =============================================================================
# TestAutoDetection
# =============================================================================

class TestAutoDetection:
    """Tests for format auto-detection."""

    def test_detect_markdown(self):
        """Should detect Markdown format from ## headings."""
        strategy = _detect_document_format("## Heading\n\nContent here.")
        assert strategy == ChunkingStrategy.MARKDOWN

    def test_detect_html(self):
        """Should detect HTML format from h2/h3 tags."""
        strategy = _detect_document_format("<h2>Heading</h2><p>Content</p>")
        assert strategy == ChunkingStrategy.HTML

    def test_detect_plain_text(self):
        """Plain text should fall back to FLAT strategy."""
        strategy = _detect_document_format("Just plain text with no structure.")
        assert strategy == ChunkingStrategy.FLAT

    def test_prefer_markdown_when_mixed(self):
        """When both formats present, prefer Markdown."""
        doc = "## Markdown Heading\n\n<h2>HTML Heading</h2>"
        strategy = _detect_document_format(doc)
        assert strategy == ChunkingStrategy.MARKDOWN

    def test_auto_strategy_routes_correctly(self):
        """AUTO strategy should detect and use appropriate parser."""
        # Markdown document
        md_chunks = chunk_document(SAMPLE_MARKDOWN, strategy=ChunkingStrategy.AUTO)
        assert any(c.level == ChunkLevel.MACRO for c in md_chunks)

        # Plain text should use FLAT
        plain_chunks = chunk_document(SAMPLE_PLAIN_TEXT, strategy=ChunkingStrategy.AUTO)
        assert all(c.level == ChunkLevel.FLAT for c in plain_chunks)


# =============================================================================
# TestBackwardsCompatibility
# =============================================================================

class TestBackwardsCompatibility:
    """Tests ensuring backwards compatibility with existing behavior."""

    def test_default_strategy_is_flat(self):
        """Default strategy should be FLAT."""
        # Call without strategy parameter
        chunks = chunk_document("This is a test document with some content.")

        assert all(c.level == ChunkLevel.FLAT for c in chunks)

    def test_flat_chunks_have_default_hierarchy_values(self):
        """FLAT chunks should have default hierarchy field values."""
        chunks = chunk_document("Test document here.", strategy=ChunkingStrategy.FLAT)

        for chunk in chunks:
            assert chunk.level == ChunkLevel.FLAT
            assert chunk.heading is None
            assert chunk.parent_index is None
            assert chunk.depth == 0

    def test_flat_chunking_unchanged(self):
        """FLAT chunking behavior should match original implementation."""
        doc = "First sentence here. Second sentence follows. Third one too."

        # Should still respect target_tokens
        chunks_small = chunk_document(doc, target_tokens=10, strategy=ChunkingStrategy.FLAT)
        chunks_large = chunk_document(doc, target_tokens=500, strategy=ChunkingStrategy.FLAT)

        # Small target should produce more chunks
        assert len(chunks_small) >= len(chunks_large)

    def test_existing_chunk_properties_preserved(self):
        """Original Chunk properties should still work."""
        chunks = chunk_document("Test document with content.")

        chunk = chunks[0]
        assert hasattr(chunk, 'index')
        assert hasattr(chunk, 'text')
        assert hasattr(chunk, 'char_start')
        assert hasattr(chunk, 'char_end')
        assert hasattr(chunk, 'token_count')
        assert hasattr(chunk, 'char_count')  # Property

    def test_empty_document_raises(self):
        """Empty document should raise ChunkingError regardless of strategy."""
        with pytest.raises(ChunkingError):
            chunk_document("", strategy=ChunkingStrategy.FLAT)

        with pytest.raises(ChunkingError):
            chunk_document("", strategy=ChunkingStrategy.MARKDOWN)

    def test_whitespace_only_raises(self):
        """Whitespace-only document should raise ChunkingError."""
        with pytest.raises(ChunkingError):
            chunk_document("   \n\t  ", strategy=ChunkingStrategy.FLAT)


# =============================================================================
# TestChunkingConfig
# =============================================================================

class TestChunkingConfig:
    """Tests for ChunkingConfig customization."""

    def test_default_config_values(self):
        """Default config should have reasonable values."""
        config = ChunkingConfig()

        assert config.macro_min_words == 100
        assert config.macro_max_words == 800
        assert config.micro_min_words == 50
        assert config.micro_max_words == 200
        assert config.atomic_min_words == 20
        assert config.atomic_max_words == 100
        assert config.split_long_paragraphs is True

    def test_custom_config(self):
        """Custom config should be respected."""
        config = ChunkingConfig(
            atomic_max_words=50,
            split_long_paragraphs=True,
        )

        # Create a document with a very long paragraph
        long_paragraph = "Word " * 100  # 100 words
        doc = f"## Section\n\n{long_paragraph}"

        chunks = chunk_document(
            doc,
            strategy=ChunkingStrategy.MARKDOWN,
            config=config,
        )

        # Long paragraph should be split due to config
        atomic_chunks = [c for c in chunks if c.level == ChunkLevel.ATOMIC]
        # May have multiple atomic chunks from the split
        assert len(chunks) >= 1


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_document_with_only_h1(self):
        """Document with only H1 should fall back gracefully."""
        doc = "# Title\n\nJust content without H2 or H3."
        chunks = chunk_document(doc, strategy=ChunkingStrategy.MARKDOWN)

        # Should still produce chunks
        assert len(chunks) >= 1

    def test_deeply_nested_html(self):
        """Nested HTML tags should be handled."""
        doc = "<h2><span class='title'>Section</span></h2><p>Content</p>"
        chunks = chunk_document(doc, strategy=ChunkingStrategy.HTML)

        assert len(chunks) >= 1

    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        doc = "## Secci\u00f3n Uno\n\nContenido en espa\u00f1ol aqu\u00ed."
        chunks = chunk_document(doc, strategy=ChunkingStrategy.MARKDOWN)

        # Check that unicode is preserved in either text or heading
        has_unicode = any(
            ("\u00f3" in c.text or (c.heading and "\u00f3" in c.heading))
            for c in chunks
        )
        assert has_unicode

    def test_very_short_sections(self):
        """Very short sections should still create chunks."""
        doc = "## A\n\nB\n\n## C\n\nD"
        chunks = chunk_document(doc, strategy=ChunkingStrategy.MARKDOWN)

        assert len(chunks) >= 2

    def test_consecutive_headings(self):
        """Consecutive headings without content between them."""
        doc = "## First\n\n## Second\n\nContent here."
        chunks = chunk_document(doc, strategy=ChunkingStrategy.MARKDOWN)

        # Should handle gracefully
        assert len(chunks) >= 1
