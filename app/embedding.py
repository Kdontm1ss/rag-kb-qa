import json
import math
import os
from operator import itemgetter

import httpx

VECTOR_DIMENSIONS = 1536
BATCH_SIZE = 16


class EmbeddingError(Exception):
    pass


def vector_parameter(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def unpack_vectors(payload: dict, expected: int) -> list[list[float]]:
    try:
        rows = sorted(payload["data"], key=itemgetter("index"))
        if [row["index"] for row in rows] != list(range(expected)):
            raise ValueError("invalid response indices")
        vectors = [row["embedding"] for row in rows]
        for vector in vectors:
            if len(vector) != VECTOR_DIMENSIONS or not all(
                type(value) in (int, float) and math.isfinite(value) for value in vector
            ) or not any(vector):
                raise ValueError("invalid vector")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise EmbeddingError("Embedding 响应缺失或不是有效的 1536 维向量") from error
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EmbeddingError("请配置 DASHSCOPE_API_KEY")
    if os.getenv("EMBEDDING_DIMENSIONS", str(VECTOR_DIMENSIONS)) != str(VECTOR_DIMENSIONS):
        raise EmbeddingError("EMBEDDING_DIMENSIONS 必须为 1536，以匹配 pgvector 列")

    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    vectors = []
    try:
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30, connect=5),
        ) as client:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]
                response = client.post("embeddings", json={
                    "model": model,
                    "input": batch,
                    "dimensions": VECTOR_DIMENSIONS,
                    "encoding_format": "float",
                })
                response.raise_for_status()
                vectors.extend(unpack_vectors(response.json(), len(batch)))
    except httpx.HTTPStatusError as error:
        raise EmbeddingError(f"Embedding 服务返回 HTTP {error.response.status_code}") from error
    except (httpx.RequestError, httpx.InvalidURL) as error:
        raise EmbeddingError(f"无法调用 Embedding 服务 ({type(error).__name__})") from error
    except ValueError as error:
        raise EmbeddingError("Embedding 服务响应格式无效") from error
    return vectors
