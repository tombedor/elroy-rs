use std::path::{Path, PathBuf};
use std::sync::Arc;

use elroy_config::AppConfig;
use elroy_core::{clear_background_status, set_background_status};
use elroy_db::{LOCAL_USER_TOKEN, open_sqlite_connection, run_migrations};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec, argument_limit};
use serde_json::{Value, json};

use super::{
    CodexSessionResult, dispatch_codex_session_with_hook, dispatch_codex_session_with_bin,
    get_codex_session_by_thread_id, list_recent_codex_sessions, resume_codex_session_with_bin,
    resume_codex_session_with_hook,
};

pub fn codex_background_status_key(session_id: &str) -> String {
    format!("codex-session-{session_id}")
}

fn codex_background_status_message(session_id: &str) -> String {
    format!("codex session {session_id} running...")
}

fn codex_completion_followup_status_message(session_id: &str) -> String {
    format!("processing codex session {session_id} completion...")
}

fn codex_session_result_payload(result: CodexSessionResult) -> String {
    json!({
        "session_id": result.session_id,
        "repo_path": result.repo_path,
        "worktree_path": result.worktree_path,
        "session_branch": result.session_branch,
        "target_branch": result.target_branch,
        "status": result.status,
        "final_message": result.final_message,
        "summary": result.summary,
        "touched_paths": result.touched_paths,
        "dirty_paths_before": result.dirty_paths_before,
        "dirty_paths_after": result.dirty_paths_after,
        "commands": result.commands.into_iter().map(|command| {
            json!({
                "command": command.command,
                "exit_code": command.exit_code,
                "output_excerpt": command.output_excerpt,
            })
        }).collect::<Vec<_>>(),
        "session_file_path": result.session_file_path,
        "resume_command": result.resume_command,
        "running_in_background": result.running_in_background,
    })
    .to_string()
}

fn with_tool_connection(
    database_path: &Path,
    operation: impl FnOnce(&rusqlite::Connection) -> rusqlite::Result<ToolExecutionResult>,
) -> ToolExecutionResult {
    let mut connection = match open_sqlite_connection(database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    if let Err(error) = run_migrations(&mut connection) {
        return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
    }
    match operation(&connection) {
        Ok(result) => result,
        Err(error) => ToolExecutionResult::error(format!("database query failed: {error}")),
    }
}

pub fn codex_tools(
    config: AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook: Arc<dyn Fn(CodexSessionResult) + Send + Sync>,
) -> Vec<ExecutableTool> {
    let database_path = config.database_path.clone();
    let codex_bin_for_dispatch = codex_bin_override.clone();
    let codex_completion_hook_for_dispatch = codex_completion_hook.clone();
    let dispatch_codex_session = ExecutableTool::new(
        ToolSpec::new(
            "dispatch_codex_session",
            "Launch a background Codex session against a repository and persist its running state.",
            JsonSchema::object(
                [
                    ("prompt", json!({"type": "string"})),
                    ("repo_path", json!({"type": "string"})),
                    ("model", json!({"type": "string"})),
                ],
                ["prompt"],
            ),
        ),
        move |arguments| {
            let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "dispatch_codex_session requires a string prompt",
                );
            };
            let repo_path = arguments.get("repo_path").and_then(Value::as_str);
            let model = arguments.get("model").and_then(Value::as_str);
            let mut connection = match open_sqlite_connection(&database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            drop(connection);

            let completion_hook = {
                let upstream_hook = codex_completion_hook_for_dispatch.clone();
                Arc::new(move |result: CodexSessionResult| {
                    set_background_status(
                        codex_background_status_key(&result.session_id),
                        codex_completion_followup_status_message(&result.session_id),
                    );
                    upstream_hook(result.clone());
                    clear_background_status(&codex_background_status_key(&result.session_id));
                })
            };

            let result = if let Some(codex_bin) = codex_bin_for_dispatch.as_deref() {
                dispatch_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    prompt,
                    repo_path.map(Path::new),
                    model,
                    codex_bin,
                    Some(completion_hook),
                )
            } else {
                dispatch_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    prompt,
                    repo_path.map(Path::new),
                    model,
                    Some(completion_hook),
                )
            };
            match result {
                Ok(result) => {
                    set_background_status(
                        codex_background_status_key(&result.session_id),
                        codex_background_status_message(&result.session_id),
                    );
                    ToolExecutionResult::success(codex_session_result_payload(result))
                }
                Err(error) => ToolExecutionResult::error(error.to_string()),
            }
        },
    );

    let database_path = config.database_path.clone();
    let codex_bin_for_resume = codex_bin_override.clone();
    let codex_completion_hook_for_resume = codex_completion_hook.clone();
    let resume_codex_session = ExecutableTool::new(
        ToolSpec::new(
            "resume_codex_session",
            "Resume a previously recorded Codex session and persist its running state.",
            JsonSchema::object(
                [
                    ("session_id", json!({"type": "string"})),
                    ("prompt", json!({"type": "string"})),
                    ("model", json!({"type": "string"})),
                ],
                ["session_id", "prompt"],
            ),
        ),
        move |arguments| {
            let Some(session_id) = arguments.get("session_id").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "resume_codex_session requires a string session_id",
                );
            };
            let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
                return ToolExecutionResult::error("resume_codex_session requires a string prompt");
            };
            let model = arguments.get("model").and_then(Value::as_str);
            let mut connection = match open_sqlite_connection(&database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            drop(connection);

            let completion_hook = {
                let upstream_hook = codex_completion_hook_for_resume.clone();
                Arc::new(move |result: CodexSessionResult| {
                    set_background_status(
                        codex_background_status_key(&result.session_id),
                        codex_completion_followup_status_message(&result.session_id),
                    );
                    upstream_hook(result.clone());
                    clear_background_status(&codex_background_status_key(&result.session_id));
                })
            };

            let result = if let Some(codex_bin) = codex_bin_for_resume.as_deref() {
                resume_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    session_id,
                    prompt,
                    model,
                    codex_bin,
                    Some(completion_hook),
                )
            } else {
                resume_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    session_id,
                    prompt,
                    model,
                    Some(completion_hook),
                )
            };
            match result {
                Ok(result) => {
                    set_background_status(
                        codex_background_status_key(&result.session_id),
                        codex_background_status_message(&result.session_id),
                    );
                    ToolExecutionResult::success(codex_session_result_payload(result))
                }
                Err(error) => ToolExecutionResult::error(error.to_string()),
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_codex_sessions = ExecutableTool::new(
        ToolSpec::new(
            "list_codex_sessions",
            "List recently recorded Codex sessions for this local user.",
            JsonSchema::object(
                [
                    ("repo_path", json!({"type": "string"})),
                    ("limit", json!({"type": "integer"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 5);
            let repo_path = arguments.get("repo_path").and_then(Value::as_str);
            with_tool_connection(&database_path, |connection| {
                let sessions = list_recent_codex_sessions(
                    connection,
                    LOCAL_USER_TOKEN,
                    repo_path.map(Path::new),
                    limit,
                )?;
                let payload = sessions
                    .into_iter()
                    .map(|session| {
                        json!({
                            "session_id": session.thread_id,
                            "repo_path": session.repo_path,
                            "worktree_path": session.worktree_path,
                            "session_branch": session.session_branch,
                            "target_branch": session.target_branch,
                            "status": session.status,
                            "updated_at_unix": session.updated_at_unix,
                            "summary": session.latest_summary,
                            "final_message": session.latest_agent_message,
                            "touched_paths": session.touched_paths,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("codex session payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let show_codex_session = ExecutableTool::new(
        ToolSpec::new(
            "show_codex_session",
            "Show one recorded Codex session by exact session id.",
            JsonSchema::object([("session_id", json!({"type": "string"}))], ["session_id"]),
        ),
        move |arguments| {
            let Some(session_id) = arguments.get("session_id").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "show_codex_session requires a string session_id",
                );
            };
            with_tool_connection(&database_path, |connection| {
                let Some(session) =
                    get_codex_session_by_thread_id(connection, LOCAL_USER_TOKEN, session_id)?
                else {
                    return Ok(ToolExecutionResult::error(format!(
                        "codex session not found: {session_id}"
                    )));
                };
                Ok(ToolExecutionResult::success(
                    json!({
                        "session_id": session.thread_id,
                        "repo_path": session.repo_path,
                        "worktree_path": session.worktree_path,
                        "session_branch": session.session_branch,
                        "target_branch": session.target_branch,
                        "latest_prompt": session.latest_prompt,
                        "summary": session.latest_summary,
                        "final_message": session.latest_agent_message,
                        "status": session.status,
                        "command_count": session.command_count,
                        "commands": session.commands.into_iter().map(|command| {
                            json!({
                                "command": command.command,
                                "exit_code": command.exit_code,
                                "output_excerpt": command.output_excerpt,
                            })
                        }).collect::<Vec<_>>(),
                        "touched_paths": session.touched_paths,
                        "dirty_paths_before": session.dirty_paths_before,
                        "dirty_paths_after": session.dirty_paths_after,
                        "session_file_path": session.session_file_path,
                        "updated_at_unix": session.updated_at_unix,
                    })
                    .to_string(),
                ))
            })
        },
    );

    vec![
        dispatch_codex_session,
        resume_codex_session,
        list_codex_sessions,
        show_codex_session,
    ]
}
