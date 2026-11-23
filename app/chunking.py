from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdown_it import MarkdownIt

from .parsing import NeedsReview, SourcePart

TARGET_TOKENS = 600
MAX_TOKENS = 800
OVERLAP_RATIO = 0.15
MARKDOWN = MarkdownIt("commonmark").enable("table")


@dataclass(frozen=True)
class Chunk:
    content: str
    token_count: int
    metadata: dict


@cache
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer().encode_ordinary(text))


PROSE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=520,
    chunk_overlap=0,
    length_function=count_tokens,
    keep_separator="end",
    separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
)


def split_prose(text: str) -> Iterator[str]:
    pieces = PROSE_SPLITTER.split_text(text)
    if len(pieces) >= 2 and count_tokens(pieces[-1]) < 300:
        merged = "\n\n".join(pieces[-2:])
        if count_tokens(merged) <= 680:
            pieces[-2:] = [merged]

    previous = ""
    for piece in pieces:
        tokens = tokenizer().encode_ordinary(previous)
        overlap_count = round(len(tokens) * OVERLAP_RATIO)
        overlap = tokenizer().decode(tokens[-overlap_count:], errors="ignore").strip() if overlap_count else ""
        yield "\n\n".join(block for block in (overlap, piece) if block)
        previous = piece


def split_table(lines: list[str]) -> Iterator[str]:
    table = "\n".join(lines)
    if count_tokens(table) <= MAX_TOKENS:
        yield table
        return

    header = "\n".join(lines[:2])
    if count_tokens(header) >= MAX_TOKENS:
        raise NeedsReview("表头超出单个 chunk 的 800 token 上限")
    rows: list[str] = []
    for row in lines[2:]:
        if count_tokens(f"{header}\n{row}") > MAX_TOKENS:
            raise NeedsReview("表格单行连同表头超出 800 token 上限")
        if rows and count_tokens("\n".join([header, *rows, row])) > TARGET_TOKENS:
            yield "\n".join([header, *rows])
            rows = []
        rows.append(row)
    if rows:
        yield "\n".join([header, *rows])


def split_part(part: SourcePart) -> Iterator[tuple[str, dict]]:
    lines = part.text.split("\n")
    tokens = MARKDOWN.parse(part.text)
    headings: list[tuple[int, str]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token.level != 0 or token.type not in {"heading_open", "table_open"}:
            continue
        first, end = token.map
        metadata = {"page": part.page, "section": [title for _, title in headings], "kind": "text"}
        for text in split_prose("\n".join(lines[start:first])):
            yield text, metadata

        if token.type == "heading_open":
            level = int(token.tag[1:])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, tokens[index + 1].content))
            start = first
        else:
            for table in split_table(lines[first:end]):
                yield table, {**metadata, "kind": "table"}
            start = end

    metadata = {"page": part.page, "section": [title for _, title in headings], "kind": "text"}
    for text in split_prose("\n".join(lines[start:])):
        yield text, metadata


def chunk_document(parts: list[SourcePart]) -> list[Chunk]:
    chunks = [
        Chunk(text, count_tokens(text), metadata)
        for part in parts
        for text, metadata in split_part(part)
    ]
    if not chunks or any(chunk.token_count > MAX_TOKENS for chunk in chunks):
        raise NeedsReview("无法生成符合 token 上限的文本块")
    return chunks
