from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class ChildChunk:
    source_path: str
    law_name: str
    domain: str
    parent_id: str
    parent_index: int
    child_index: int
    parent_text: str
    child_text: str

# iterate through root and select only .txt files
def iter_source_files(data_root: Path) -> Iterable[Path]:
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in {".txt", ""}:
            yield path

# seperate title and body by custom delimiter; else first line is title and rest is body; else title is filename and body is empty
def parse_source(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    if "/()/()/" in raw:
        title, body = raw.split("/()/()/", 1)
        return title.strip() or path.stem, body.strip()

    lines = raw.splitlines()
    if not lines:
        return path.stem, ""
    title = lines[0].strip() or path.stem
    body = "\n".join(lines[1:]).strip()
    return title, body

# standardise newlines 
def _clean_text(text: str) -> str:
    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()

# split text into parent chunks based on markdown headers
def _markdown_parent_chunks(text: str) -> list[str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
    )
    docs = splitter.split_text(text)
    chunks = [doc.page_content.strip() for doc in docs if doc.page_content and doc.page_content.strip()]
    return chunks

# fallback chunking method using recursive character splitting
# [NOTE] this probably could be used as main chunking method unless we can produce md / hierachical chunking is a lot better 
def _fallback_parent_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def build_parent_child_chunks(
    *,
    source_path: Path,
    law_name: str,
    domain: str,
    body: str,
    parent_chunk_size: int = 1400,
    parent_chunk_overlap: int = 120,
    child_chunk_size: int = 380,
    child_chunk_overlap: int = 60,
    parent_snippet_chars: int = 900,
) -> list[ChildChunk]:
    cleaned = _clean_text(body)
    if not cleaned:
        return []

    parent_chunks = _markdown_parent_chunks(cleaned)
    if len(parent_chunks) <= 1 and len(cleaned) > parent_chunk_size:
        parent_chunks = _fallback_parent_chunks(cleaned, parent_chunk_size, parent_chunk_overlap)
    elif not parent_chunks:
        parent_chunks = _fallback_parent_chunks(cleaned, parent_chunk_size, parent_chunk_overlap)

    # split the child chunks 
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    records: list[ChildChunk] = []
    seen_child_text: set[str] = set()

    # create child chunk for each parent chunk; also create a "snippet" of the parent text to be used as context for the child chunk 
    for parent_index, parent_text in enumerate(parent_chunks):
        if not parent_text.strip():
            continue
        parent_id = f"{source_path.stem}:p{parent_index}"
        parent_snippet = parent_text.strip()[:parent_snippet_chars]

        children = [chunk.strip() for chunk in child_splitter.split_text(parent_text) if chunk.strip()]
        for child_index, child_text in enumerate(children):
            child_key = child_text.lower().strip()
            if child_key in seen_child_text:
                continue
            seen_child_text.add(child_key)
            records.append(
                ChildChunk(
                    source_path=str(source_path),
                    law_name=law_name,
                    domain=domain,
                    parent_id=parent_id,
                    parent_index=parent_index,
                    child_index=child_index,
                    parent_text=parent_snippet,
                    child_text=child_text,
                )
            )

    return records
