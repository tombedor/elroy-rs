CREATE TABLE IF NOT EXISTS user_preferences (
    user_token TEXT PRIMARY KEY,
    assistant_name TEXT,
    preferred_name TEXT,
    full_name TEXT,
    system_persona TEXT,
    created_at_unix INTEGER NOT NULL,
    updated_at_unix INTEGER NOT NULL
);
