CREATE TABLE agenda_items_new (
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
    is_active INTEGER DEFAULT 1,
    updated_at_unix INTEGER NOT NULL,
    checklist_total INTEGER NOT NULL DEFAULT 0,
    checklist_completed INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (bootstrap_document_id) REFERENCES bootstrap_documents(id) ON DELETE CASCADE
);

INSERT INTO agenda_items_new (
    id,
    bootstrap_document_id,
    legacy_frontmatter_id,
    name,
    file_path,
    agenda_date,
    is_completed,
    status,
    trigger_datetime,
    trigger_context,
    closing_comment,
    body,
    is_active,
    updated_at_unix,
    checklist_total,
    checklist_completed
)
SELECT
    id,
    bootstrap_document_id,
    legacy_frontmatter_id,
    name,
    file_path,
    agenda_date,
    is_completed,
    status,
    trigger_datetime,
    trigger_context,
    closing_comment,
    body,
    is_active,
    updated_at_unix,
    checklist_total,
    checklist_completed
FROM agenda_items;

DROP TABLE agenda_items;

ALTER TABLE agenda_items_new RENAME TO agenda_items;

CREATE INDEX IF NOT EXISTS idx_agenda_items_legacy_frontmatter_id
    ON agenda_items(legacy_frontmatter_id);

CREATE INDEX IF NOT EXISTS idx_agenda_items_agenda_date
    ON agenda_items(agenda_date);
