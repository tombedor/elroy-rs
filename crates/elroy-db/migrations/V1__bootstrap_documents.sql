CREATE TABLE IF NOT EXISTS bootstrap_documents (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    stem TEXT NOT NULL,
    frontmatter_id INTEGER,
    agenda_date TEXT,
    is_completed INTEGER NOT NULL DEFAULT 0,
    status TEXT,
    body TEXT NOT NULL,
    updated_at_unix INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bootstrap_documents_kind
    ON bootstrap_documents(kind);

CREATE INDEX IF NOT EXISTS idx_bootstrap_documents_frontmatter_id
    ON bootstrap_documents(frontmatter_id);
