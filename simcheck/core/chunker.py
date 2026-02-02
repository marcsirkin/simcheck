"""
Document chunking for semantic analysis.

This module splits documents into chunks suitable for embedding comparison.
The chunking strategy is intentionally simple and opinionated:

Strategy:
1. Split document into sentences
2. Group sentences into chunks of approximately TARGET_TOKENS tokens
3. Never split mid-sentence (semantic boundary preservation)
4. If a single sentence exceeds max size, keep it as its own chunk

Hierarchical Chunking (v2):
The module also supports hierarchical chunking based on document structure:
- MARKDOWN strategy: Parse ## (H2) and ### (H3) headings
- HTML strategy: Parse <h2> and <h3> tags
- AUTO strategy: Auto-detect format

Three-tier hierarchy:
- MACRO: H2-level sections (typically 300-800 words)
- MICRO: H3-level subsections (typically 100-200 words)
- ATOMIC: Paragraph-level content (typically 20-50 words)

Token Estimation:
We use a simple heuristic: ~4 characters per token for English text.
This avoids the overhead of loading a tokenizer and is accurate enough
for chunking purposes. The actual embedding model handles precise tokenization.

Design Decisions:
- No overlap between chunks (simpler analysis, cleaner boundaries)
- Sentence-boundary preservation (maintains semantic coherence)
- Fixed defaults (opinionated, not configurable in v1)
- Hierarchical mode preserves semantic structure from headings
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from simcheck.core.models import Chunk, ChunkLevel


# Default chunk size in approximate tokens.
# 150 tokens (~600 chars) provides granular analysis for diagnostic purposes.
# Smaller chunks make it easier to identify exactly where content drifts
# off-topic, which is the primary use case for this tool.
# Most sentence-transformer models handle 512 tokens max.
DEFAULT_CHUNK_TOKENS = 150

# Approximate characters per token for English text.
# This is a rough heuristic. Actual tokenization varies by model,
# but for chunking purposes this is sufficient.
CHARS_PER_TOKEN = 4


class ChunkingError(Exception):
    """Raised when document chunking fails."""
    pass


class ChunkingStrategy(Enum):
    """
    Strategy for document chunking.

    FLAT: Original sentence-boundary chunking (default, backwards compatible)
    MARKDOWN: Parse Markdown headings (## H2, ### H3) for hierarchy
    HTML: Parse HTML heading tags (<h2>, <h3>) for hierarchy
    AUTO: Auto-detect document format and choose appropriate strategy
    """
    FLAT = "flat"           # Current behavior (default)
    MARKDOWN = "markdown"   # Parse Markdown headings
    HTML = "html"           # Parse HTML heading tags
    AUTO = "auto"           # Auto-detect format


@dataclass
class ChunkingConfig:
    """
    Configuration for hierarchical chunking.

    Attributes:
        macro_min_words: Minimum words for MACRO chunks (default: 100)
        macro_max_words: Maximum words for MACRO chunks (default: 800)
        micro_min_words: Minimum words for MICRO chunks (default: 50)
        micro_max_words: Maximum words for MICRO chunks (default: 200)
        atomic_min_words: Minimum words for ATOMIC chunks (default: 20)
        atomic_max_words: Maximum words for ATOMIC chunks (default: 100)
        split_long_paragraphs: Split paragraphs exceeding max words (default: True)
    """
    macro_min_words: int = 100
    macro_max_words: int = 800
    micro_min_words: int = 50
    micro_max_words: int = 200
    atomic_min_words: int = 20
    atomic_max_words: int = 100
    split_long_paragraphs: bool = True


# Default configuration
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using character-based heuristic.

    This is intentionally approximate. For English text, ~4 chars/token
    is reasonable. The embedding model will handle actual tokenization.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.

    This handles common sentence boundaries: . ! ?
    It attempts to avoid splitting on abbreviations like "Dr." or "U.S."
    by requiring whitespace or end-of-string after the punctuation.

    Args:
        text: Input text

    Returns:
        List of sentence strings (may include trailing whitespace)
    """
    # Pattern: split after .!? followed by whitespace or end
    # This is imperfect but handles most cases without NLP dependencies
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)

    # Filter out empty strings
    return [s for s in sentences if s.strip()]


def chunk_document(
    document: str,
    target_tokens: int = DEFAULT_CHUNK_TOKENS,
    strategy: ChunkingStrategy = ChunkingStrategy.FLAT,
    config: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """
    Split a document into semantic chunks for embedding comparison.

    By default (FLAT strategy), the document is split into sentences,
    then sentences are grouped into chunks of approximately target_tokens size.
    Sentence boundaries are always preserved to maintain semantic coherence.

    With hierarchical strategies (MARKDOWN, HTML, AUTO), the document is
    parsed for heading structure and chunked according to semantic boundaries
    defined by headings (H2 -> MACRO, H3 -> MICRO, paragraphs -> ATOMIC).

    Args:
        document: The document text to chunk
        target_tokens: Target size for each chunk in tokens (for FLAT strategy)
        strategy: Chunking strategy (FLAT, MARKDOWN, HTML, AUTO)
        config: Optional configuration for hierarchical chunking

    Returns:
        List of Chunk objects in document order

    Raises:
        ChunkingError: If document is empty or chunking fails

    Example:
        >>> # Default flat chunking
        >>> chunks = chunk_document(document)
        >>>
        >>> # Markdown hierarchical chunking
        >>> chunks = chunk_document(document, strategy=ChunkingStrategy.MARKDOWN)
        >>>
        >>> # Auto-detect format
        >>> chunks = chunk_document(document, strategy=ChunkingStrategy.AUTO)
    """
    # Validate input
    if not document or not document.strip():
        raise ChunkingError("Document is empty or contains only whitespace")

    # Handle AUTO detection
    effective_strategy = strategy
    if strategy == ChunkingStrategy.AUTO:
        effective_strategy = _detect_document_format(document)

    # Route to appropriate chunking implementation
    if effective_strategy == ChunkingStrategy.FLAT:
        return _chunk_document_flat(document, target_tokens)
    else:
        return chunk_document_hierarchical(document, effective_strategy, config)


def get_total_tokens(chunks: List[Chunk]) -> int:
    """
    Calculate total estimated tokens across all chunks.

    Args:
        chunks: List of Chunk objects

    Returns:
        Sum of token counts
    """
    return sum(chunk.token_count for chunk in chunks)


# =============================================================================
# Hierarchical Chunking (v2)
# =============================================================================

@dataclass
class _ParsedSection:
    """Internal structure for parsed document sections."""
    level: int  # 2 for H2, 3 for H3, 4 for paragraph
    heading: Optional[str]
    content: str
    char_start: int
    char_end: int


def _detect_document_format(document: str) -> ChunkingStrategy:
    """
    Auto-detect document format based on content patterns.

    Checks for Markdown heading patterns (## and ###) and HTML heading
    tags (<h2>, <h3>). If neither is found, falls back to FLAT strategy.

    Args:
        document: The document text to analyze

    Returns:
        Detected ChunkingStrategy
    """
    # Check for Markdown headings (## or ###)
    markdown_pattern = r'^#{2,3}\s+.+$'
    has_markdown = bool(re.search(markdown_pattern, document, re.MULTILINE))

    # Check for HTML headings (<h2> or <h3>)
    html_pattern = r'<h[23][^>]*>.*?</h[23]>'
    has_html = bool(re.search(html_pattern, document, re.IGNORECASE | re.DOTALL))

    if has_markdown and not has_html:
        return ChunkingStrategy.MARKDOWN
    elif has_html and not has_markdown:
        return ChunkingStrategy.HTML
    elif has_markdown and has_html:
        # Prefer Markdown if both present (more common in mixed docs)
        return ChunkingStrategy.MARKDOWN
    else:
        return ChunkingStrategy.FLAT


def _parse_markdown_structure(document: str) -> List[_ParsedSection]:
    """
    Parse Markdown document into hierarchical sections.

    Extracts H2 (##) and H3 (###) headings, treating content between
    headings as belonging to the preceding heading. Paragraphs within
    sections are identified by blank lines.

    Args:
        document: Markdown document text

    Returns:
        List of _ParsedSection objects representing document structure
    """
    sections: List[_ParsedSection] = []

    # Pattern to match ## or ### headings
    heading_pattern = r'^(#{2,3})\s+(.+)$'

    lines = document.split('\n')
    current_heading: Optional[str] = None
    current_level: int = 4  # Default to paragraph level
    current_content_lines: List[str] = []
    current_start: int = 0
    char_pos: int = 0

    for line in lines:
        match = re.match(heading_pattern, line)

        if match:
            # Save previous section if it has content
            if current_content_lines:
                content = '\n'.join(current_content_lines).strip()
                if content:
                    sections.append(_ParsedSection(
                        level=current_level,
                        heading=current_heading,
                        content=content,
                        char_start=current_start,
                        char_end=char_pos - 1,
                    ))

            # Start new section
            hashes, heading_text = match.groups()
            current_level = len(hashes)  # 2 for ##, 3 for ###
            current_heading = heading_text.strip()
            current_content_lines = []
            current_start = char_pos

        else:
            current_content_lines.append(line)

        char_pos += len(line) + 1  # +1 for newline

    # Don't forget the last section
    if current_content_lines:
        content = '\n'.join(current_content_lines).strip()
        if content:
            sections.append(_ParsedSection(
                level=current_level,
                heading=current_heading,
                content=content,
                char_start=current_start,
                char_end=char_pos,
            ))

    return sections


def _parse_html_structure(document: str) -> List[_ParsedSection]:
    """
    Parse HTML document into hierarchical sections.

    Extracts <h2> and <h3> tags, treating content between headings
    as belonging to the preceding heading.

    Args:
        document: HTML document text

    Returns:
        List of _ParsedSection objects representing document structure
    """
    sections: List[_ParsedSection] = []

    # Pattern to match h2 or h3 tags with content
    heading_pattern = r'<(h[23])[^>]*>(.*?)</\1>'

    # Find all headings with their positions
    headings: List[Tuple[int, int, int, str]] = []  # (start, end, level, text)
    for match in re.finditer(heading_pattern, document, re.IGNORECASE | re.DOTALL):
        tag = match.group(1).lower()
        level = int(tag[1])  # 'h2' -> 2, 'h3' -> 3
        heading_text = re.sub(r'<[^>]+>', '', match.group(2)).strip()  # Strip inner tags
        headings.append((match.start(), match.end(), level, heading_text))

    if not headings:
        # No headings found, treat entire document as one section
        content = re.sub(r'<[^>]+>', ' ', document).strip()  # Strip all tags
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        if content:
            sections.append(_ParsedSection(
                level=4,
                heading=None,
                content=content,
                char_start=0,
                char_end=len(document),
            ))
        return sections

    # Process content between headings
    prev_end = 0
    for i, (start, end, level, heading_text) in enumerate(headings):
        # Content before this heading (belongs to previous section or is orphan)
        if start > prev_end:
            content_before = document[prev_end:start]
            content_text = re.sub(r'<[^>]+>', ' ', content_before).strip()
            content_text = re.sub(r'\s+', ' ', content_text)
            if content_text and sections:
                # Append to previous section
                sections[-1] = _ParsedSection(
                    level=sections[-1].level,
                    heading=sections[-1].heading,
                    content=sections[-1].content + ' ' + content_text if sections[-1].content else content_text,
                    char_start=sections[-1].char_start,
                    char_end=start,
                )
            elif content_text:
                # Orphan content before first heading
                sections.append(_ParsedSection(
                    level=4,
                    heading=None,
                    content=content_text,
                    char_start=prev_end,
                    char_end=start,
                ))

        # Content after this heading (until next heading or end)
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(document)
        content_after = document[end:next_start]
        content_text = re.sub(r'<[^>]+>', ' ', content_after).strip()
        content_text = re.sub(r'\s+', ' ', content_text)

        sections.append(_ParsedSection(
            level=level,
            heading=heading_text,
            content=content_text,
            char_start=start,
            char_end=next_start,
        ))

        prev_end = next_start

    return sections


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on blank lines.

    Args:
        text: Input text

    Returns:
        List of paragraph strings
    """
    # Split on one or more blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    # Filter out empty paragraphs and strip whitespace
    return [p.strip() for p in paragraphs if p.strip()]


def _chunk_document_flat(
    document: str,
    target_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> List[Chunk]:
    """
    Original flat chunking logic (internal function).

    This is the backwards-compatible chunking that splits by sentences.

    Args:
        document: The document text to chunk
        target_tokens: Target size for each chunk in tokens

    Returns:
        List of Chunk objects in document order
    """
    # Normalize whitespace
    document = document.strip()

    # Split into sentences
    sentences = _split_into_sentences(document)

    if not sentences:
        raise ChunkingError("Could not extract any sentences from document")

    # Group sentences into chunks
    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_token_count = 0
    char_position = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        if current_sentences and (current_token_count + sentence_tokens > target_tokens):
            chunk_text = " ".join(current_sentences)
            chunk_start = char_position
            chunk_end = chunk_start + len(chunk_text)

            chunks.append(Chunk(
                index=len(chunks),
                text=chunk_text,
                char_start=chunk_start,
                char_end=chunk_end,
                token_count=current_token_count,
                level=ChunkLevel.FLAT,
            ))

            char_position = chunk_end
            while char_position < len(document) and document[char_position] in ' \n\t':
                char_position += 1

            current_sentences = [sentence]
            current_token_count = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += sentence_tokens

    # Last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_start = char_position
        chunk_end = min(chunk_start + len(chunk_text), len(document))

        chunks.append(Chunk(
            index=len(chunks),
            text=chunk_text,
            char_start=chunk_start,
            char_end=chunk_end,
            token_count=current_token_count,
            level=ChunkLevel.FLAT,
        ))

    return chunks


def _level_from_parsed_level(parsed_level: int) -> ChunkLevel:
    """Convert parsed level (2, 3, 4) to ChunkLevel enum."""
    if parsed_level == 2:
        return ChunkLevel.MACRO
    elif parsed_level == 3:
        return ChunkLevel.MICRO
    else:
        return ChunkLevel.ATOMIC


def chunk_document_hierarchical(
    document: str,
    strategy: ChunkingStrategy,
    config: Optional[ChunkingConfig] = None,
) -> List[Chunk]:
    """
    Chunk document using hierarchical structure from headings.

    Creates a three-tier hierarchy based on document headings:
    - MACRO chunks from H2 sections
    - MICRO chunks from H3 subsections
    - ATOMIC chunks from paragraphs

    Args:
        document: The document text to chunk
        strategy: MARKDOWN or HTML parsing strategy
        config: Optional configuration for chunk sizes

    Returns:
        List of Chunk objects with hierarchy metadata

    Raises:
        ChunkingError: If document is empty or parsing fails
    """
    if not document or not document.strip():
        raise ChunkingError("Document is empty or contains only whitespace")

    cfg = config or DEFAULT_CHUNKING_CONFIG

    # Parse document structure based on strategy
    if strategy == ChunkingStrategy.MARKDOWN:
        sections = _parse_markdown_structure(document)
    elif strategy == ChunkingStrategy.HTML:
        sections = _parse_html_structure(document)
    else:
        raise ChunkingError(f"Unsupported strategy for hierarchical chunking: {strategy}")

    if not sections:
        # No structure found, fall back to flat chunking with ATOMIC level
        return _chunk_document_flat(document)

    chunks: List[Chunk] = []
    parent_indices: dict = {}  # Map level -> most recent chunk index at that level

    for section in sections:
        level = _level_from_parsed_level(section.level)
        content = section.content

        if not content:
            continue

        # Determine parent
        parent_index: Optional[int] = None
        if level == ChunkLevel.MICRO:
            # MICRO's parent is the most recent MACRO
            parent_index = parent_indices.get(ChunkLevel.MACRO)
        elif level == ChunkLevel.ATOMIC:
            # ATOMIC's parent is the most recent MICRO, or MACRO if no MICRO
            parent_index = parent_indices.get(ChunkLevel.MICRO)
            if parent_index is None:
                parent_index = parent_indices.get(ChunkLevel.MACRO)

        # Determine depth
        depth = 0 if level == ChunkLevel.MACRO else (1 if level == ChunkLevel.MICRO else 2)

        # Check if we need to split long content
        word_count = len(content.split())
        max_words = (cfg.macro_max_words if level == ChunkLevel.MACRO else
                     cfg.micro_max_words if level == ChunkLevel.MICRO else
                     cfg.atomic_max_words)

        if cfg.split_long_paragraphs and word_count > max_words and level == ChunkLevel.ATOMIC:
            # Split long atomic chunks by sentences
            sentences = _split_into_sentences(content)
            current_words: List[str] = []
            current_word_count = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())

                if current_words and current_word_count + sentence_words > max_words:
                    # Create chunk from accumulated sentences
                    chunk_text = ' '.join(current_words)
                    chunks.append(Chunk(
                        index=len(chunks),
                        text=chunk_text,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        token_count=_estimate_tokens(chunk_text),
                        level=level,
                        heading=section.heading,
                        parent_index=parent_index,
                        depth=depth,
                    ))
                    current_words = [sentence]
                    current_word_count = sentence_words
                else:
                    current_words.append(sentence)
                    current_word_count += sentence_words

            # Last chunk from remaining sentences
            if current_words:
                chunk_text = ' '.join(current_words)
                chunks.append(Chunk(
                    index=len(chunks),
                    text=chunk_text,
                    char_start=section.char_start,
                    char_end=section.char_end,
                    token_count=_estimate_tokens(chunk_text),
                    level=level,
                    heading=section.heading,
                    parent_index=parent_index,
                    depth=depth,
                ))
        else:
            # Create single chunk for this section
            chunk = Chunk(
                index=len(chunks),
                text=content,
                char_start=section.char_start,
                char_end=section.char_end,
                token_count=_estimate_tokens(content),
                level=level,
                heading=section.heading,
                parent_index=parent_index,
                depth=depth,
            )
            chunks.append(chunk)

            # Update parent tracking for MACRO and MICRO levels
            if level in (ChunkLevel.MACRO, ChunkLevel.MICRO):
                parent_indices[level] = chunk.index

    if not chunks:
        raise ChunkingError("Could not extract any chunks from document")

    return chunks
