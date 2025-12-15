CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    page_id INTEGER REFERENCES pages(id),
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536)
);