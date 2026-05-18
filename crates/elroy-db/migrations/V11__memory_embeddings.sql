CREATE TABLE IF NOT EXISTS memory_embeddings (
    file_path TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    created_at_unix INTEGER NOT NULL,
    updated_at_unix INTEGER NOT NULL
);
