from pathlib import Path

from .db import connect


def ensure_lexical_schema() -> None:
    with connect() as conn:
        # Take a transaction-scoped lock so simultaneous API starts apply the migration once.
        conn.execute("SELECT pg_advisory_xact_lock(20261123, 2)")
        installed = conn.execute(
            "SELECT to_regclass('chunks_search_terms_gin') IS NOT NULL AS installed"
        ).fetchone()
        if not installed["installed"]:
            script = Path(__file__).resolve().parents[1] / "sql/migrations/002_chunk_lexical_search.sql"
            conn.execute(script.read_text(encoding="utf-8"), prepare=False)
