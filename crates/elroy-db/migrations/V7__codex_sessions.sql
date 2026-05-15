CREATE TABLE IF NOT EXISTS codex_sessions (
    id INTEGER PRIMARY KEY,
    user_token TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    worktree_path TEXT,
    session_branch TEXT,
    target_branch TEXT,
    latest_prompt TEXT NOT NULL,
    latest_summary TEXT,
    latest_agent_message TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    command_count INTEGER NOT NULL DEFAULT 0,
    commands_json TEXT NOT NULL DEFAULT '[]',
    touched_paths_json TEXT NOT NULL DEFAULT '[]',
    dirty_paths_before_json TEXT NOT NULL DEFAULT '[]',
    dirty_paths_after_json TEXT NOT NULL DEFAULT '[]',
    session_file_path TEXT,
    created_at_unix INTEGER NOT NULL,
    updated_at_unix INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_codex_sessions_user_thread
    ON codex_sessions(user_token, thread_id);

CREATE INDEX IF NOT EXISTS idx_codex_sessions_user_updated_at
    ON codex_sessions(user_token, updated_at_unix DESC);
