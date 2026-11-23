from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from unicodedata import normalize

import pdfplumber
from docx import Document
from docx.text.paragraph import Paragraph
from pdfplumber.utils import extract_text


class NeedsReview(Exception):
    pass


@dataclass(frozen=True)
class SourcePart:
    text: str
    page: int | None = None


def markdown_table(rows: list[list[str | None]]) -> str:
    width = max((len(row) for row in rows), default=0)
    if not width or not any(cell and cell.strip() for row in rows for cell in row):
        return ""

    def render(row: list[str | None]) -> str:
        escaped = [
            " ".join((cell or "").split()).replace("\\", "\\\\").replace("|", "\\|")
            for cell in row
        ]
        return "| " + " | ".join(escaped + [""] * (width - len(row))) + " |"

    return "\n".join([render(rows[0]), render(["---"] * width), *map(render, rows[1:])])


def parse_docx(path: Path) -> str:
    blocks = []
    for item in Document(path).iter_inner_content():
        if isinstance(item, Paragraph):
            style_id = item.style.style_id if item.style else ""
            level = style_id.removeprefix("Heading")
            text = item.text.strip()
            if text and style_id.startswith("Heading") and level in {"1", "2", "3", "4", "5", "6"}:
                text = f"{'#' * int(level)} {text}"
            blocks.append(text)
        else:
            blocks.append(markdown_table([[cell.text for cell in row.cells] for row in item.rows]))
    return "\n\n".join(block for block in blocks if block)


def pdf_page_text(page: pdfplumber.page.Page) -> str:
    tables = []
    for table in page.find_tables():
        markdown = markdown_table(table.extract())
        if markdown:
            tables.append((table.bbox, markdown))
    if not tables:
        return page.extract_text() or ""

    tables.sort(key=lambda table: (table[0][1], table[0][0]))
    table_starts = [bbox[1] for bbox, _ in tables]
    fragments = [[] for _ in range(len(tables) + 1)]
    for char in page.chars:
        x = (char["x0"] + char["x1"]) / 2
        y = (char["top"] + char["bottom"]) / 2
        if not any(left <= x < right and top <= y < bottom for (left, top, right, bottom), _ in tables):
            fragments[bisect_right(table_starts, y)].append(char)

    blocks = []
    for index, (_, table) in enumerate(tables):
        blocks.extend((extract_text(fragments[index]), table))
    blocks.append(extract_text(fragments[-1]))
    return "\n\n".join(block for block in blocks if block)


def parse_pdf(path: Path) -> list[SourcePart]:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(SourcePart(pdf_page_text(page), page.page_number))
            page.close()
    return parts


def read_text(path: Path) -> str:
    content = path.read_bytes()
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = content.decode("utf-16")
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("gb18030")
    if "\x00" in text:
        raise NeedsReview("文本包含 NUL 字符，请检查文件格式或编码")
    return text


def clean_text(text: str) -> str:
    text = normalize("NFC", text).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def parse_document(path: Path, file_type: str) -> list[SourcePart]:
    try:
        if file_type == "pdf":
            parts = parse_pdf(path)
        elif file_type == "docx":
            parts = [SourcePart(parse_docx(path))]
        else:
            parts = [SourcePart(read_text(path))]
    except (NeedsReview, OSError):
        raise
    except Exception as error:
        raise NeedsReview(f"{file_type.upper()} 解析失败 ({type(error).__name__})") from error

    cleaned = [SourcePart(clean_text(part.text), part.page) for part in parts]
    if not any(part.text for part in cleaned):
        raise NeedsReview("未提取到文本，请检查是否为空文档、扫描件或图片版 PDF")
    return [part for part in cleaned if part.text]
