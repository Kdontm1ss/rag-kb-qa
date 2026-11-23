import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Generic, Literal, TypeVar
from uuid import UUID, uuid4

import psycopg
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .db import connect
from .embedding import EmbeddingError, embed_texts, vector_parameter
from .storage import DOCUMENT_QUEUE, UPLOAD_DIR, queue_connection

logger = logging.getLogger(__name__)
app = FastAPI(title="RAG 知识库文档入库")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FILE_TYPES = {"pdf", "docx", "md", "txt"}
DOCUMENT_FIELDS = """
    SELECT d.id AS document_id, d.title, d.file_type, d.status, d.content_hash,
           d.duplicate_of, d.error_message, d.created_at,
           (SELECT count(*) FROM chunks WHERE document_id = COALESCE(d.duplicate_of, d.id)) AS chunk_count
    FROM documents AS d
"""
CHUNK_FIELDS = """
    c.id AS chunk_id, c.document_id, d.title, c.content, c.chunk_index, c.token_count, c.metadata
"""

DocumentStatus = Literal["processing", "ready", "failed", "needs_review"]
T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "ok"
    data: T


class UploadResult(BaseModel):
    document_id: UUID
    status: Literal["processing"] = "processing"


class DocumentInfo(BaseModel):
    document_id: UUID
    title: str
    file_type: str
    status: DocumentStatus
    content_hash: str | None
    duplicate_of: UUID | None
    error_message: str | None
    created_at: datetime
    chunk_count: int


class ChunkHit(BaseModel):
    chunk_id: int
    document_id: UUID
    title: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict
    score: float | None


@app.exception_handler(StarletteHTTPException)
def handle_http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"code": error.status_code, "message": error.detail, "data": None},
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
def handle_validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    details = [{"location": item["loc"], "message": item["msg"]} for item in error.errors()]
    return JSONResponse(
        status_code=422,
        content={"code": 422, "message": "请求参数无效", "data": details},
    )


@app.exception_handler(psycopg.OperationalError)
def handle_database_error(_request: Request, error: psycopg.OperationalError) -> JSONResponse:
    logger.warning("Database unavailable (%s)", type(error).__name__)
    return JSONResponse(status_code=503, content={"code": 503, "message": "数据库暂不可用", "data": None})


@app.exception_handler(Exception)
def handle_server_error(_request: Request, _error: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"code": 500, "message": "服务内部错误", "data": None})


@app.get("/health")
def health() -> ApiResponse[dict[str, str]]:
    with connect() as conn:
        conn.execute("SELECT 1 FROM documents LIMIT 1")
    return ApiResponse(data={"status": "ok"})


@app.post("/api/documents/upload", status_code=202)
def upload_document(file: Annotated[UploadFile, File()]) -> ApiResponse[UploadResult]:
    title = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    file_type = Path(title).suffix.lower().lstrip(".")
    if file_type not in FILE_TYPES:
        raise HTTPException(415, "支持的文件类型为 PDF、DOCX、Markdown 和 TXT")
    if len(title) > 255 or "\x00" in title:
        raise HTTPException(400, "文件名无效")

    document_id = uuid4()
    path = UPLOAD_DIR / f"{document_id}.{file_type}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = path.open("xb")
    try:
        with target:
            total = 0
            while block := file.file.read(1024 * 1024):
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "文件大小上限为 20 MiB")
                target.write(block)
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, title, file_type, file_path, status)
                VALUES (%s, %s, %s, %s, 'processing')
                """,
                (document_id, title, file_type, path.name),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise

    try:
        queue_connection().rpush(DOCUMENT_QUEUE, str(document_id))
    except RedisError:
        logger.warning("Redis notification failed for document %s. Worker will recover it from PostgreSQL.", document_id)

    return ApiResponse(data=UploadResult(document_id=document_id))


@app.get("/api/documents")
def list_documents(
    status: DocumentStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[list[DocumentInfo]]:
    with connect() as conn:
        rows = conn.execute(
            DOCUMENT_FIELDS + " WHERE (%s::text IS NULL OR d.status = %s)"
            " ORDER BY d.created_at DESC, d.id DESC LIMIT %s OFFSET %s",
            (status, status, limit, offset),
        ).fetchall()
    return ApiResponse(data=[DocumentInfo.model_validate(row) for row in rows])


@app.get("/api/documents/{document_id}")
def get_document(document_id: UUID) -> ApiResponse[DocumentInfo]:
    with connect() as conn:
        document = conn.execute(DOCUMENT_FIELDS + " WHERE d.id = %s", (document_id,)).fetchone()
    if document is None:
        raise HTTPException(404, "文档不存在")
    return ApiResponse(data=DocumentInfo.model_validate(document))


@app.get("/api/chunks/search")
def search_chunks(
    q: Annotated[str, Query(min_length=1, max_length=1000)],
    mode: Literal["keyword", "vector"] = "keyword",
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
) -> ApiResponse[list[ChunkHit]]:
    query = q.strip()
    if not query or "\x00" in query:
        raise HTTPException(422, "请输入有效的检索内容")

    if mode == "vector":
        try:
            vector = embed_texts([query])[0]
        except EmbeddingError as error:
            raise HTTPException(503, str(error)) from error
        sql = f"""
            SELECT {CHUNK_FIELDS}, 1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready' ORDER BY score DESC, c.id LIMIT %s
        """
        parameters = (vector_parameter(vector), limit)
    else:
        sql = f"""
            SELECT {CHUNK_FIELDS}, NULL AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready' AND strpos(lower(c.content), lower(%s)) > 0
            ORDER BY c.id LIMIT %s
        """
        parameters = (query, limit)

    with connect() as conn:
        rows = conn.execute(sql, parameters).fetchall()
    return ApiResponse(data=[ChunkHit.model_validate(row) for row in rows])
