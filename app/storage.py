import os
from functools import cache
from pathlib import Path

from redis import Redis

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))
DOCUMENT_QUEUE = "rag:documents:ingest"


@cache
def queue_connection() -> Redis:
    return Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        socket_connect_timeout=2,
        socket_timeout=5,
    )
