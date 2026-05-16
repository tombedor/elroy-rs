CREATE TABLE IF NOT EXISTS memory_operation_tracker (
    user_token TEXT PRIMARY KEY,
    memories_since_consolidation INTEGER NOT NULL DEFAULT 0,
    messages_since_memory INTEGER NOT NULL DEFAULT 0,
    created_at_unix INTEGER NOT NULL,
    updated_at_unix INTEGER NOT NULL
);
