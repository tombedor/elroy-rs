CREATE TABLE IF NOT EXISTS context_message_sets (
    id INTEGER PRIMARY KEY,
    user_token TEXT NOT NULL UNIQUE,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at_unix INTEGER NOT NULL,
    updated_at_unix INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS context_messages (
    id INTEGER PRIMARY KEY,
    set_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    chat_model TEXT,
    created_at_unix INTEGER NOT NULL,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    FOREIGN KEY (set_id) REFERENCES context_message_sets(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_context_messages_set_position
    ON context_messages(set_id, position);

CREATE INDEX IF NOT EXISTS idx_context_message_sets_user_token
    ON context_message_sets(user_token);
