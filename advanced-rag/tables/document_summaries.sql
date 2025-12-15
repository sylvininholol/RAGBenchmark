CREATE TABLE IF NOT EXISTS document_summaries (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE UNIQUE,
    summary_text TEXT NOT NULL,
    embedding VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_summaries_embedding ON document_summaries USING hnsw (embedding vector_l2_ops);