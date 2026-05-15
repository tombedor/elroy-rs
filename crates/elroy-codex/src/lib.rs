use std::path::{Path, PathBuf};

use rusqlite::{Connection, params};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexCommandRecord {
    pub command: String,
    pub exit_code: Option<i64>,
    pub output_excerpt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexSessionRecord {
    pub id: i64,
    pub user_token: String,
    pub thread_id: String,
    pub repo_path: String,
    pub worktree_path: Option<String>,
    pub session_branch: Option<String>,
    pub target_branch: Option<String>,
    pub latest_prompt: String,
    pub latest_summary: Option<String>,
    pub latest_agent_message: Option<String>,
    pub status: String,
    pub command_count: i64,
    pub commands: Vec<CodexCommandRecord>,
    pub touched_paths: Vec<String>,
    pub dirty_paths_before: Vec<String>,
    pub dirty_paths_after: Vec<String>,
    pub session_file_path: Option<String>,
    pub created_at_unix: i64,
    pub updated_at_unix: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexSessionUpdate {
    pub repo_path: PathBuf,
    pub worktree_path: Option<PathBuf>,
    pub session_branch: Option<String>,
    pub target_branch: Option<String>,
    pub prompt: String,
    pub summary: String,
    pub agent_message: String,
    pub status: String,
    pub commands: Vec<CodexCommandRecord>,
    pub touched_paths: Vec<String>,
    pub dirty_paths_before: Vec<String>,
    pub dirty_paths_after: Vec<String>,
    pub session_file_path: Option<String>,
}

pub fn get_codex_session_by_thread_id(
    connection: &Connection,
    user_token: &str,
    thread_id: &str,
) -> rusqlite::Result<Option<CodexSessionRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            user_token,
            thread_id,
            repo_path,
            worktree_path,
            session_branch,
            target_branch,
            latest_prompt,
            latest_summary,
            latest_agent_message,
            status,
            command_count,
            commands_json,
            touched_paths_json,
            dirty_paths_before_json,
            dirty_paths_after_json,
            session_file_path,
            created_at_unix,
            updated_at_unix
        FROM codex_sessions
        WHERE user_token = ?1
          AND thread_id = ?2",
    )?;
    let result = statement.query_row(params![user_token, thread_id], map_codex_session_row);
    match result {
        Ok(record) => Ok(Some(record)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn list_recent_codex_sessions(
    connection: &Connection,
    user_token: &str,
    repo_path: Option<&Path>,
    limit: usize,
) -> rusqlite::Result<Vec<CodexSessionRecord>> {
    let mut statement = if repo_path.is_some() {
        connection.prepare(
            "SELECT
                id,
                user_token,
                thread_id,
                repo_path,
                worktree_path,
                session_branch,
                target_branch,
                latest_prompt,
                latest_summary,
                latest_agent_message,
                status,
                command_count,
                commands_json,
                touched_paths_json,
                dirty_paths_before_json,
                dirty_paths_after_json,
                session_file_path,
                created_at_unix,
                updated_at_unix
            FROM codex_sessions
            WHERE user_token = ?1
              AND repo_path = ?2
            ORDER BY updated_at_unix DESC
            LIMIT ?3",
        )?
    } else {
        connection.prepare(
            "SELECT
                id,
                user_token,
                thread_id,
                repo_path,
                worktree_path,
                session_branch,
                target_branch,
                latest_prompt,
                latest_summary,
                latest_agent_message,
                status,
                command_count,
                commands_json,
                touched_paths_json,
                dirty_paths_before_json,
                dirty_paths_after_json,
                session_file_path,
                created_at_unix,
                updated_at_unix
            FROM codex_sessions
            WHERE user_token = ?1
            ORDER BY updated_at_unix DESC
            LIMIT ?2",
        )?
    };

    let rows = match repo_path {
        Some(repo_path) => statement.query_map(
            params![user_token, repo_path.display().to_string(), limit as i64],
            map_codex_session_row,
        )?,
        None => statement.query_map(params![user_token, limit as i64], map_codex_session_row)?,
    };
    rows.collect()
}

pub fn upsert_codex_session(
    connection: &mut Connection,
    user_token: &str,
    thread_id: &str,
    update: &CodexSessionUpdate,
) -> rusqlite::Result<CodexSessionRecord> {
    let now = unix_timestamp_now();
    let existing = get_codex_session_by_thread_id(connection, user_token, thread_id)?;
    let created_at = existing
        .as_ref()
        .map(|record| record.created_at_unix)
        .unwrap_or(now);

    connection.execute(
        "INSERT INTO codex_sessions (
            user_token,
            thread_id,
            repo_path,
            worktree_path,
            session_branch,
            target_branch,
            latest_prompt,
            latest_summary,
            latest_agent_message,
            status,
            command_count,
            commands_json,
            touched_paths_json,
            dirty_paths_before_json,
            dirty_paths_after_json,
            session_file_path,
            created_at_unix,
            updated_at_unix
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
        ON CONFLICT(user_token, thread_id) DO UPDATE SET
            repo_path = excluded.repo_path,
            worktree_path = excluded.worktree_path,
            session_branch = excluded.session_branch,
            target_branch = excluded.target_branch,
            latest_prompt = excluded.latest_prompt,
            latest_summary = excluded.latest_summary,
            latest_agent_message = excluded.latest_agent_message,
            status = excluded.status,
            command_count = excluded.command_count,
            commands_json = excluded.commands_json,
            touched_paths_json = excluded.touched_paths_json,
            dirty_paths_before_json = excluded.dirty_paths_before_json,
            dirty_paths_after_json = excluded.dirty_paths_after_json,
            session_file_path = excluded.session_file_path,
            updated_at_unix = excluded.updated_at_unix",
        params![
            user_token,
            thread_id,
            update.repo_path.display().to_string(),
            update
                .worktree_path
                .as_ref()
                .map(|path| path.display().to_string()),
            update.session_branch,
            update.target_branch,
            update.prompt,
            update.summary,
            update.agent_message,
            update.status,
            update.commands.len() as i64,
            serde_json::to_string(&command_values(&update.commands)).map_err(json_to_sql_error)?,
            serde_json::to_string(&update.touched_paths).map_err(json_to_sql_error)?,
            serde_json::to_string(&update.dirty_paths_before).map_err(json_to_sql_error)?,
            serde_json::to_string(&update.dirty_paths_after).map_err(json_to_sql_error)?,
            update.session_file_path,
            created_at,
            now,
        ],
    )?;

    get_codex_session_by_thread_id(connection, user_token, thread_id)?
        .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)
}

fn map_codex_session_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CodexSessionRecord> {
    let commands_json: String = row.get(12)?;
    let touched_paths_json: String = row.get(13)?;
    let dirty_paths_before_json: String = row.get(14)?;
    let dirty_paths_after_json: String = row.get(15)?;

    Ok(CodexSessionRecord {
        id: row.get(0)?,
        user_token: row.get(1)?,
        thread_id: row.get(2)?,
        repo_path: row.get(3)?,
        worktree_path: row.get(4)?,
        session_branch: row.get(5)?,
        target_branch: row.get(6)?,
        latest_prompt: row.get(7)?,
        latest_summary: row.get(8)?,
        latest_agent_message: row.get(9)?,
        status: row.get(10)?,
        command_count: row.get(11)?,
        commands: parse_commands_json(&commands_json),
        touched_paths: parse_string_list_json(&touched_paths_json),
        dirty_paths_before: parse_string_list_json(&dirty_paths_before_json),
        dirty_paths_after: parse_string_list_json(&dirty_paths_after_json),
        session_file_path: row.get(16)?,
        created_at_unix: row.get(17)?,
        updated_at_unix: row.get(18)?,
    })
}

fn command_values(commands: &[CodexCommandRecord]) -> Vec<Value> {
    commands
        .iter()
        .map(|command| {
            serde_json::json!({
                "command": command.command,
                "exit_code": command.exit_code,
                "output_excerpt": command.output_excerpt,
            })
        })
        .collect()
}

fn parse_commands_json(raw: &str) -> Vec<CodexCommandRecord> {
    serde_json::from_str::<Vec<Value>>(raw)
        .unwrap_or_default()
        .into_iter()
        .map(|value| CodexCommandRecord {
            command: value
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            exit_code: value.get("exit_code").and_then(Value::as_i64),
            output_excerpt: value
                .get("output_excerpt")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        })
        .collect()
}

fn parse_string_list_json(raw: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}

fn json_to_sql_error(error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error))
}

fn unix_timestamp_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use rusqlite::Connection;

    use super::{
        CodexCommandRecord, CodexSessionUpdate, get_codex_session_by_thread_id,
        list_recent_codex_sessions, upsert_codex_session,
    };

    fn run_test_migrations(connection: &mut Connection) {
        connection
            .execute_batch(
                "CREATE TABLE codex_sessions (
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
                CREATE UNIQUE INDEX idx_codex_sessions_user_thread
                    ON codex_sessions(user_token, thread_id);
                CREATE INDEX idx_codex_sessions_user_updated_at
                    ON codex_sessions(user_token, updated_at_unix DESC);",
            )
            .expect("codex sessions schema should be created");
    }

    fn update(repo_name: &str, prompt: &str, status: &str) -> CodexSessionUpdate {
        CodexSessionUpdate {
            repo_path: PathBuf::from(format!("/tmp/{repo_name}")),
            worktree_path: Some(PathBuf::from(format!("/tmp/{repo_name}/.worktree"))),
            session_branch: Some(format!("session-{repo_name}")),
            target_branch: Some("agent".to_string()),
            prompt: prompt.to_string(),
            summary: format!("summary for {prompt}"),
            agent_message: format!("agent message for {prompt}"),
            status: status.to_string(),
            commands: vec![CodexCommandRecord {
                command: "/bin/zsh -lc pwd".to_string(),
                exit_code: Some(0),
                output_excerpt: "/tmp\n".to_string(),
            }],
            touched_paths: vec!["notes.txt".to_string()],
            dirty_paths_before: vec!["README.md".to_string()],
            dirty_paths_after: vec!["notes.txt".to_string()],
            session_file_path: Some(format!("/tmp/{repo_name}/thread.jsonl")),
        }
    }

    #[test]
    fn upsert_round_trips_and_updates_existing_session() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_test_migrations(&mut connection);

        let first = upsert_codex_session(
            &mut connection,
            "local-user",
            "thread-123",
            &update("sample", "initial prompt", "running"),
        )
        .expect("session should persist");
        let second = upsert_codex_session(
            &mut connection,
            "local-user",
            "thread-123",
            &update("sample", "follow up", "completed"),
        )
        .expect("session should update");

        assert_eq!(first.thread_id, "thread-123");
        assert_eq!(second.id, first.id);
        assert_eq!(second.latest_prompt, "follow up");
        assert_eq!(second.status, "completed");
        assert_eq!(second.commands.len(), 1);
        assert_eq!(second.touched_paths, vec!["notes.txt"]);
    }

    #[test]
    fn get_and_list_recent_sessions_scope_to_user_and_repo() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_test_migrations(&mut connection);

        upsert_codex_session(
            &mut connection,
            "local-user",
            "thread-123",
            &update("sample", "first prompt", "completed"),
        )
        .expect("first session should persist");
        upsert_codex_session(
            &mut connection,
            "local-user",
            "thread-456",
            &update("other", "second prompt", "running"),
        )
        .expect("second session should persist");
        upsert_codex_session(
            &mut connection,
            "other-user",
            "thread-789",
            &update("sample", "third prompt", "completed"),
        )
        .expect("other user session should persist");

        let exact = get_codex_session_by_thread_id(&connection, "local-user", "thread-123")
            .expect("session should query")
            .expect("session should exist");
        let listed = list_recent_codex_sessions(&connection, "local-user", None, 10)
            .expect("sessions should list");
        let filtered = list_recent_codex_sessions(
            &connection,
            "local-user",
            Some(Path::new("/tmp/sample")),
            10,
        )
        .expect("filtered sessions should list");

        assert_eq!(exact.thread_id, "thread-123");
        assert_eq!(listed.len(), 2);
        assert!(
            listed
                .iter()
                .all(|record| record.user_token == "local-user")
        );
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].thread_id, "thread-123");
    }
}
