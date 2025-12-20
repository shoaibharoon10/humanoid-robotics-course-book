"""Markdown document chunking for RAG ingestion."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import frontmatter
import tiktoken


@dataclass
class Chunk:
    """Represents a document chunk for embedding."""
    text: str
    chunk_id: str
    chunk_index: int
    doc_id: str
    title: str
    module: str
    heading: str
    heading_level: int
    section_hierarchy: List[str]
    has_code: bool
    code_languages: List[str]
    has_mermaid: bool
    has_table: bool
    has_callout: bool
    callout_type: Optional[str]
    url_path: str
    file_path: str
    line_start: int
    line_end: int
    token_count: int
    keywords: List[str] = field(default_factory=list)
    description: str = ""


class MarkdownChunker:
    """Chunk markdown documents intelligently by headers."""

    def __init__(self, max_tokens: int = 1500, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_file(self, file_path: Path, base_url: str = "/docs") -> List[Chunk]:
        """Chunk a markdown file into smaller pieces."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse frontmatter
        post = frontmatter.loads(content)
        metadata = post.metadata
        body = post.content

        # Extract document info
        doc_id = self._extract_doc_id(file_path)
        module = self._extract_module(file_path)
        title = metadata.get("title", file_path.stem)
        url_path = f"{base_url}/{'/'.join(file_path.parts[-2:]).replace('.md', '')}"
        keywords = metadata.get("keywords", [])
        description = metadata.get("description", "")

        # Split by headers
        sections = self._split_by_headers(body)

        chunks = []
        for i, section in enumerate(sections):
            section_chunks = self._process_section(
                section=section,
                chunk_base_index=len(chunks),
                doc_id=doc_id,
                title=title,
                module=module,
                url_path=url_path,
                file_path=str(file_path),
                keywords=keywords,
                description=description
            )
            chunks.extend(section_chunks)

        return chunks

    def _split_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Split content by H2 and H3 headers."""
        lines = content.split("\n")
        sections = []
        current_section = {
            "heading": "Introduction",
            "heading_level": 1,
            "hierarchy": [],
            "lines": [],
            "line_start": 1
        }

        hierarchy = []

        for i, line in enumerate(lines, 1):
            header_match = re.match(r"^(#{2,4})\s+(.+)$", line)

            if header_match:
                # Save current section if it has content
                if current_section["lines"]:
                    current_section["line_end"] = i - 1
                    sections.append(current_section)

                level = len(header_match.group(1))
                heading = header_match.group(2).strip()

                # Update hierarchy
                while hierarchy and hierarchy[-1][0] >= level:
                    hierarchy.pop()
                hierarchy.append((level, heading))

                current_section = {
                    "heading": heading,
                    "heading_level": level,
                    "hierarchy": [h[1] for h in hierarchy],
                    "lines": [],
                    "line_start": i
                }
            else:
                current_section["lines"].append(line)

        # Add final section
        if current_section["lines"]:
            current_section["line_end"] = len(lines)
            sections.append(current_section)

        return sections

    def _process_section(
        self,
        section: Dict[str, Any],
        chunk_base_index: int,
        doc_id: str,
        title: str,
        module: str,
        url_path: str,
        file_path: str,
        keywords: List[str],
        description: str
    ) -> List[Chunk]:
        """Process a section, splitting if too long."""
        text = "\n".join(section["lines"]).strip()

        if not text:
            return []

        tokens = self._count_tokens(text)
        content_info = self._detect_content_types(text)

        if tokens <= self.max_tokens:
            # Single chunk
            return [Chunk(
                text=text,
                chunk_id=f"{doc_id}#chunk-{chunk_base_index}",
                chunk_index=chunk_base_index,
                doc_id=doc_id,
                title=title,
                module=module,
                heading=section["heading"],
                heading_level=section["heading_level"],
                section_hierarchy=section["hierarchy"],
                url_path=url_path,
                file_path=file_path,
                line_start=section["line_start"],
                line_end=section.get("line_end", section["line_start"]),
                token_count=tokens,
                keywords=keywords,
                description=description,
                **content_info
            )]
        else:
            # Split into multiple chunks
            return self._split_with_overlap(
                text=text,
                section=section,
                chunk_base_index=chunk_base_index,
                doc_id=doc_id,
                title=title,
                module=module,
                url_path=url_path,
                file_path=file_path,
                keywords=keywords,
                description=description
            )

    def _split_with_overlap(
        self,
        text: str,
        section: Dict[str, Any],
        chunk_base_index: int,
        doc_id: str,
        title: str,
        module: str,
        url_path: str,
        file_path: str,
        keywords: List[str],
        description: str
    ) -> List[Chunk]:
        """Split long text into overlapping chunks."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                content_info = self._detect_content_types(chunk_text)

                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}#chunk-{chunk_base_index + len(chunks)}",
                    chunk_index=chunk_base_index + len(chunks),
                    doc_id=doc_id,
                    title=title,
                    module=module,
                    heading=section["heading"],
                    heading_level=section["heading_level"],
                    section_hierarchy=section["hierarchy"],
                    url_path=url_path,
                    file_path=file_path,
                    line_start=section["line_start"],
                    line_end=section.get("line_end", section["line_start"]),
                    token_count=self._count_tokens(chunk_text),
                    keywords=keywords,
                    description=description,
                    **content_info
                ))

                # Start new chunk with overlap
                overlap_paras = []
                overlap_tokens = 0
                for p in reversed(current_chunk):
                    p_tokens = self._count_tokens(p)
                    if overlap_tokens + p_tokens <= self.overlap_tokens:
                        overlap_paras.insert(0, p)
                        overlap_tokens += p_tokens
                    else:
                        break

                current_chunk = overlap_paras
                current_tokens = overlap_tokens

            current_chunk.append(para)
            current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            content_info = self._detect_content_types(chunk_text)

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}#chunk-{chunk_base_index + len(chunks)}",
                chunk_index=chunk_base_index + len(chunks),
                doc_id=doc_id,
                title=title,
                module=module,
                heading=section["heading"],
                heading_level=section["heading_level"],
                section_hierarchy=section["hierarchy"],
                url_path=url_path,
                file_path=file_path,
                line_start=section["line_start"],
                line_end=section.get("line_end", section["line_start"]),
                token_count=self._count_tokens(chunk_text),
                keywords=keywords,
                description=description,
                **content_info
            ))

        return chunks

    def _detect_content_types(self, text: str) -> Dict[str, Any]:
        """Detect special content types in text."""
        code_languages = re.findall(r"```(\w+)", text)

        return {
            "has_code": "```" in text,
            "code_languages": list(set(code_languages)),
            "has_mermaid": "```mermaid" in text,
            "has_table": bool(re.search(r"\|.+\|", text)),
            "has_callout": bool(re.search(r":::\w+", text)),
            "callout_type": self._extract_callout_type(text)
        }

    def _extract_callout_type(self, text: str) -> Optional[str]:
        """Extract callout type if present."""
        match = re.search(r":::(\w+)", text)
        return match.group(1) if match else None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _extract_doc_id(self, file_path: Path) -> str:
        """Extract document ID from file path."""
        parts = file_path.parts
        # Get module and filename
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1].replace('.md', '')}"
        return file_path.stem

    def _extract_module(self, file_path: Path) -> str:
        """Extract module name from file path."""
        parts = file_path.parts
        for part in parts:
            if part.startswith("module-"):
                return part
        return "general"


def chunk_to_payload(chunk: Chunk) -> Dict[str, Any]:
    """Convert a Chunk to a Qdrant payload dictionary."""
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "title": chunk.title,
        "module": chunk.module,
        "heading": chunk.heading,
        "heading_level": chunk.heading_level,
        "section_hierarchy": chunk.section_hierarchy,
        "text": chunk.text,
        "text_length": len(chunk.text),
        "has_code": chunk.has_code,
        "code_languages": chunk.code_languages,
        "has_mermaid": chunk.has_mermaid,
        "has_table": chunk.has_table,
        "has_callout": chunk.has_callout,
        "callout_type": chunk.callout_type,
        "url_path": chunk.url_path,
        "file_path": chunk.file_path,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "token_count": chunk.token_count,
        "keywords": chunk.keywords,
        "description": chunk.description,
    }
