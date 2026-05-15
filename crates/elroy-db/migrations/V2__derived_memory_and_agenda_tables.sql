CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    bootstrap_document_id INTEGER NOT NULL UNIQUE,
    legacy_frontmatter_id INTEGER,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    updated_at_unix INTEGER NOT NULL,
    FOREIGN KEY (bootstrap_document_id) REFERENCES bootstrap_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_legacy_frontmatter_id
    ON memories(legacy_frontmatter_id);

CREATE TABLE IF NOT EXISTS agenda_items (
    id INTEGER PRIMARY KEY,
    bootstrap_document_id INTEGER NOT NULL UNIQUE,
    legacy_frontmatter_id INTEGER,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    agenda_date TEXT,
    is_completed INTEGER NOT NULL DEFAULT 0,
    status TEXT,
    trigger_datetime TEXT,
    trigger_context TEXT,
    closing_comment TEXT,
    body TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    updated_at_unix INTEGER NOT NULL,
    FOREIGN KEY (bootstrap_document_id) REFERENCES bootstrap_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agenda_items_legacy_frontmatter_id
    ON agenda_items(legacy_frontmatter_id);

CREATE INDEX IF NOT EXISTS idx_agenda_items_agenda_date
    ON agenda_items(agenda_date);
