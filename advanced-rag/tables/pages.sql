CREATE TABLE pages (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'
);