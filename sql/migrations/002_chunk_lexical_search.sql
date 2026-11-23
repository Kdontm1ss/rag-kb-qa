-- Split Han runs into overlapping bigrams. Index unigrams as well for single-character queries.
CREATE OR REPLACE FUNCTION rag_search_terms(source text, include_han_unigrams boolean)
RETURNS text[] LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT ARRAY(
        SELECT term
        FROM regexp_matches(lower(source), '[㐀-䶿一-鿿豈-﫿]+|[^㐀-䶿一-鿿豈-﫿]+', 'g') AS runs(parts)
        CROSS JOIN LATERAL (
            SELECT substr(parts[1], start_at, width) AS term
            FROM generate_series(1, char_length(parts[1])) AS positions(start_at)
            CROSS JOIN generate_series(1, 2) AS widths(width)
            WHERE parts[1] ~ '^[㐀-䶿一-鿿豈-﫿]'
              AND start_at + width - 1 <= char_length(parts[1])
              AND (width = 2 OR include_han_unigrams OR char_length(parts[1]) = 1)
            UNION ALL
            SELECT words[1]
            FROM regexp_matches(parts[1], '[[:alnum:]]+', 'g') AS tokens(words)
            WHERE parts[1] !~ '^[㐀-䶿一-鿿豈-﫿]' AND char_length(words[1]) <= 128
        ) AS tokens
    );
$$;

CREATE OR REPLACE FUNCTION rag_search_vector(terms text[])
RETURNS tsvector LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT to_tsvector('pg_catalog.simple', array_to_string(terms, ' '));
$$;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_terms text[]
    GENERATED ALWAYS AS (rag_search_terms(content, true)) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_terms_gin ON chunks USING gin (rag_search_vector(search_terms));
