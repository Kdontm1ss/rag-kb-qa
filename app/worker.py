import logging
import time
from hashlib import sha256

import psycopg
from psycopg import Connection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from redis.exceptions import RedisError

from .chunking import chunk_document
from .db import connect
from .embedding import EmbeddingError, embed_texts, vector_parameter
from .parsing import NeedsReview, parse_document
from .storage import DOCUMENT_QUEUE, UPLOAD_DIR, queue_connection

logger = logging.getLogger(__name__)


def ingest_document(conn: Connection[DictRow], document: DictRow) -> None:
    document_id = document["id"]
    parts = parse_document(UPLOAD_DIR / document["file_path"], document["file_type"])
    content_hash = sha256("\n\n".join(part.text for part in parts).encode("utf-8")).hexdigest()

    # Serialize identical content across workers before calling the embedding service.
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (content_hash,))
    original = conn.execute(
        """
        SELECT id FROM documents
        WHERE content_hash = %s AND status = 'ready' AND duplicate_of IS NULL
        """,
        (content_hash,),
    ).fetchone()
    duplicate_of = original["id"] if original else None
    if not original:
        chunks = chunk_document(parts)
        vectors = embed_texts([chunk.content for chunk in chunks])
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (document_id, content, chunk_index, token_count, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                """,
                [
                    (document_id, chunk.content, index, chunk.token_count, vector_parameter(vector), Jsonb(chunk.metadata))
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    conn.execute(
        "UPDATE documents SET status = 'ready', content_hash = %s, duplicate_of = %s WHERE id = %s",
        (content_hash, duplicate_of, document_id),
    )


def process_next_document() -> None:
    with connect() as conn:
        document = conn.execute(
            """
            SELECT id, file_path, file_type FROM documents
            WHERE status = 'processing' ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED
            """
        ).fetchone()
        if document is None:
            return

        status = "ready"
        error_message = None
        try:
            with conn.transaction():
                ingest_document(conn, document)
        except NeedsReview as error:
            status, error_message = "needs_review", str(error)
        except EmbeddingError as error:
            status, error_message = "failed", str(error)
        except OSError as error:
            status, error_message = "failed", f"无法读取原始文件 ({type(error).__name__})"
        except psycopg.OperationalError:
            raise
        except Exception:
            logger.exception("Document processing failed for %s", document["id"])
            status, error_message = "failed", "文档处理失败，请查看 worker 日志"

        if error_message:
            conn.execute(
                "UPDATE documents SET status = %s, error_message = %s WHERE id = %s",
                (status, error_message, document["id"]),
            )
    logger.info("Document %s: %s%s", document["id"], status, f" ({error_message})" if error_message else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Document worker started")
    while True:
        try:
            # A Redis notification wakes the worker early. On timeout we still check durable jobs.
            queue_connection().blpop(DOCUMENT_QUEUE, timeout=2)
        except RedisError:
            time.sleep(2)
        try:
            process_next_document()
        except psycopg.Error as error:
            logger.warning("Database unavailable for document worker (%s)", type(error).__name__)
            time.sleep(2)


if __name__ == "__main__":
    main()
