-- PostgreSQL Schema for Court Case Q&A Backend
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS judgments (
    id SERIAL PRIMARY KEY,
    title TEXT,
    citation TEXT,
    court TEXT,
    decision_date DATE,
    full_text TEXT,
    source_file TEXT UNIQUE, -- Ensures duplicate check by filename works at database constraint level
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS legal_chunks (
    id SERIAL PRIMARY KEY,
    judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    section_role TEXT, -- head | facts | issues | reasoning | decision | other
    content TEXT,
    word_count INTEGER,
    citations_mentioned TEXT[],
    statutes_mentioned TEXT[],
    summary TEXT,
    embedding vector({{INSERT_DIM}}), -- Dynamically set on startup based on embedder dimension
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS legal_chunks_hnsw_idx ON legal_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS legal_chunks_tsv_idx ON legal_chunks USING gin (content_tsv);
