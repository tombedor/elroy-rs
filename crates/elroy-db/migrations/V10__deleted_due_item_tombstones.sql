CREATE TABLE IF NOT EXISTS deleted_due_item_tombstones (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    agenda_date TEXT,
    trigger_datetime TEXT,
    trigger_context TEXT,
    closing_comment TEXT,
    body TEXT NOT NULL,
    original_file_path TEXT NOT NULL,
    deleted_at_unix INTEGER NOT NULL
);
