from uuid import UUID

from psycopg import Connection
from psycopg.rows import DictRow
from pydantic import BaseModel

from .db import connect
from .embedding import embed_texts, vector_parameter

HYBRID_TOP_K = 8
RRF_K = 60

CHUNK_FIELDS = """
    c.id AS chunk_id, c.document_id, d.title, c.content, c.chunk_index, c.token_count, c.metadata
"""
READY_CHUNKS = """
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE d.status = 'ready'
"""

# Okapi BM25, with each ready chunk counted as a document. The GIN index identifies matches
# for any query term, while the stored token array supplies exact frequencies and lengths.
BM25_QUERY = f"""
    WITH query_terms AS (
        SELECT DISTINCT term FROM unnest(rag_search_terms(%s, false)) AS terms(term)
    ), query_match AS (
        SELECT string_agg(quote_literal(term), ' | ')::tsquery AS query FROM query_terms
    ), corpus AS (
        SELECT count(*) AS size, avg(cardinality(c.search_terms)) AS avg_length
        {READY_CHUNKS}
    ), term_counts AS MATERIALIZED (
        SELECT c.id, tokens.term, cardinality(c.search_terms) AS length, count(*) AS frequency
        FROM chunks c JOIN documents d ON d.id = c.document_id
        CROSS JOIN query_match
        CROSS JOIN LATERAL unnest(c.search_terms) AS tokens(term)
        JOIN query_terms USING (term)
        WHERE d.status = 'ready' AND rag_search_vector(c.search_terms) @@ query_match.query
        GROUP BY c.id, tokens.term, cardinality(c.search_terms)
    ), document_counts AS (
        SELECT term, count(*) AS document_frequency FROM term_counts GROUP BY term
    ), ranked AS (
        SELECT id, sum(
            ln(1 + (corpus.size - document_frequency + 0.5) / (document_frequency + 0.5))
            * (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * length / corpus.avg_length))
        )::double precision AS score
        FROM term_counts JOIN document_counts USING (term) CROSS JOIN corpus
        GROUP BY id
        ORDER BY score DESC, id
        LIMIT %s
    )
    SELECT {CHUNK_FIELDS}, ranked.score
    FROM ranked JOIN chunks c ON c.id = ranked.id JOIN documents d ON d.id = c.document_id
    ORDER BY ranked.score DESC, c.id
"""


class ChunkHit(BaseModel):
    chunk_id: int
    document_id: UUID
    title: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict
    score: float | None


def _keyword_hits(conn: Connection[DictRow], question: str, limit: int) -> list[ChunkHit]:
    rows = conn.execute(BM25_QUERY, (question, limit)).fetchall()
    return [ChunkHit.model_validate(row) for row in rows]


def retrieve_keywords(question: str, limit: int) -> list[ChunkHit]:
    with connect() as conn:
        return _keyword_hits(conn, question, limit)


def retrieve_vectors(question: str, limit: int) -> list[ChunkHit]:
    vector = embed_texts([question])[0]
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {CHUNK_FIELDS}, 1 - (c.embedding <=> %s::vector) AS score
            {READY_CHUNKS}
            ORDER BY score DESC, c.id LIMIT %s
            """,
            (vector_parameter(vector), limit),
        ).fetchall()
    return [ChunkHit.model_validate(row) for row in rows]


def retrieve_hybrid(question: str) -> list[ChunkHit]:
    with connect() as conn:
        keywords = _keyword_hits(conn, question, HYBRID_TOP_K)
        if not keywords and conn.execute(f"SELECT 1 {READY_CHUNKS} LIMIT 1").fetchone() is None:
            return []

    vectors = retrieve_vectors(question, HYBRID_TOP_K)
    hits: dict[int, ChunkHit] = {}
    scores: dict[int, float] = {}
    for ranking in (keywords, vectors):
        for rank, hit in enumerate(ranking, start=1):
            hits[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (RRF_K + rank)

    top_ids = sorted(hits, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:HYBRID_TOP_K]
    return [hits[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in top_ids]
