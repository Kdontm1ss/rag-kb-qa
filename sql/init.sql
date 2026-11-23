BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    file_type text NOT NULL CHECK (file_type IN ('pdf', 'docx', 'md', 'txt')),
    file_path text NOT NULL,
    status text NOT NULL CHECK (status IN ('processing', 'ready', 'failed', 'needs_review')),
    content_hash text,
    duplicate_of uuid REFERENCES documents(id) ON DELETE CASCADE,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX documents_ready_content_hash
    ON documents (content_hash) WHERE status = 'ready' AND duplicate_of IS NULL;

CREATE INDEX documents_status_created_at ON documents (status, created_at, id);

CREATE TABLE chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content text NOT NULL,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    token_count integer NOT NULL CHECK (token_count BETWEEN 1 AND 800),
    embedding vector(1536) NOT NULL,
    metadata jsonb NOT NULL,
    UNIQUE (document_id, chunk_index)
);

COMMIT;
