CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scraping (
    id SERIAL PRIMARY KEY,
    titulo TEXT NOT NULL,
    texto TEXT NOT NULL,
    resumo TEXT NOT NULL,
    link TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perguntas (
    id SERIAL PRIMARY KEY,
    scraping_id INTEGER REFERENCES scraping(id),
    pergunta TEXT,
    embedding VECTOR(1536),
    tipo_avaliacao TEXT DEFAULT NULL,
    contexto_ids INTEGER[] DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS respostas (
    id SERIAL PRIMARY KEY,
    pergunta_id INTEGER REFERENCES perguntas(id),
    resposta TEXT,
    embedding VECTOR(1536)
);

-- Criar índices para embeddings
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'perguntas' AND indexname = 'perguntas_embedding_idx') THEN
        CREATE INDEX perguntas_embedding_idx ON perguntas USING hnsw (embedding vector_l2_ops);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'respostas' AND indexname = 'respostas_embedding_idx') THEN
        CREATE INDEX respostas_embedding_idx ON respostas USING hnsw (embedding vector_l2_ops);
    END IF;
END $$; 