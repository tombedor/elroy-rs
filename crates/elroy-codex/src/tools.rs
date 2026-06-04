use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use elroy_config::AppConfig;
use elroy_core::{clear_background_status, set_background_status};
use elroy_db::{LOCAL_USER_TOKEN, open_sqlite_connection, run_migrations};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};
use serde_json::{Value, json};

use super::{
    CodexSessionResult, dispatch_codex_session_with_bin, dispatch_codex_session_with_hook,
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

fn ensure_tool_database(database_path: &Path) -> Result<(), String> {
    let mut connection = open_sqlite_connection(database_path)
        .map_err(|error| format!("failed to open database: {error}"))?;
    run_migrations(&mut connection)
        .map_err(|error| format!("failed to run migrations: {error}"))?;
    Ok(())
}

fn codex_scope_repo_filter(
    scope: Option<&str>,
    repo_path: Option<&str>,
    home_dir: &Path,
) -> Result<Option<PathBuf>, String> {
    match scope {
        Some("contrib") => Ok(Some(home_dir.join("contrib"))),
        Some("elroy") => std::env::current_dir()
            .map(Some)
            .map_err(|error| format!("failed to resolve current repo: {error}")),
        Some(_) => Err("scope must be 'contrib', 'elroy', or omitted".to_string()),
        None => Ok(repo_path.map(PathBuf::from)),
    }
}

fn codex_list_limit(arguments: &Value, default_limit: usize) -> Result<usize, String> {
    let Some(value) = arguments.get("limit") else {
        return Ok(default_limit);
    };
    let Some(limit) = value.as_i64() else {
        return Err("limit must be an integer".to_string());
    };
    if limit < 1 {
        return Err("limit must be at least 1".to_string());
    }
    Ok((limit as usize).min(50))
}

fn build_inspection_prompt(prompt: &str, include_logs: bool, log_path: &Path) -> String {
    let log_section = if include_logs {
        let excerpt = tail_log_excerpt(log_path, 20_000);
        format!(
            "\n\nElroy log path: {}\nRecent log excerpt:\n{}",
            log_path.display(),
            excerpt
        )
    } else {
        String::new()
    };
    format!(
        "Inspect the running Elroy codebase and logs. Do not edit files. Report concrete findings, risks, and suggested next steps.\n\nInspection request:\n{prompt}{log_section}"
    )
}

fn build_contrib_edit_prompt(prompt: &str, contrib_repo: &Path) -> String {
    format!(
        "Edit only the Elroy contrib plugin directory for this task. Do not modify core Elroy source code. Contrib Python files may define @tool functions and/or ELROY_PLUGIN = ElroyPlugin(...) sidebar plugins. Keep changes small and directly related to the request.\n\nContrib directory: {}\n\nEdit request:\n{prompt}",
        contrib_repo.display()
    )
}

fn tail_log_excerpt(log_path: &Path, max_chars: usize) -> String {
    let Ok(contents) = fs::read_to_string(log_path) else {
        return String::new();
    };
    let char_count = contents.chars().count();
    contents
        .chars()
        .skip(char_count.saturating_sub(max_chars))
        .collect()
}

fn ensure_contrib_repo(home_dir: &Path) -> Result<PathBuf, String> {
    let target = home_dir.join("contrib");
    fs::create_dir_all(&target)
        .map_err(|error| format!("failed to create contrib repo directory: {error}"))?;

    write_default_contrib_files(&target)
        .map_err(|error| format!("failed to write default contrib files: {error}"))?;

    if !target.join(".git").exists() {
        run_git_for_contrib(None, ["init", target.to_string_lossy().as_ref()])?;
    }

    run_git_for_contrib(
        Some(&target),
        ["config", "user.email", "elroy-contrib@example.local"],
    )?;
    run_git_for_contrib(Some(&target), ["config", "user.name", "Elroy Contrib"])?;

    if !contrib_has_head(&target)? {
        write_default_contrib_files(&target)
            .map_err(|error| format!("failed to write default contrib files: {error}"))?;
        run_git_for_contrib(Some(&target), ["add", "."])?;
        run_git_for_contrib(
            Some(&target),
            ["commit", "-m", "Initial Elroy contrib repo"],
        )?;
    }

    Ok(target)
}

fn write_default_contrib_files(target: &Path) -> std::io::Result<()> {
    let gitignore = target.join(".gitignore");
    if !gitignore.exists() {
        fs::write(
            gitignore,
            "__pycache__/\n*.py[cod]\n*.egg-info/\n.eggs/\ndist/\nbuild/\n",
        )?;
    }

    let readme = target.join("README.md");
    if !readme.exists() {
        fs::write(
            readme,
            "# Elroy Contrib\n\nElroy loads this directory at startup. Add @tool functions in Python files to expose assistant tools, or define an ELROY_PLUGIN value with ElroyPlugin to add a sidebar plugin.\n",
        )?;
    }
    Ok(())
}

fn contrib_has_head(target: &Path) -> Result<bool, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(target)
        .output()
        .map_err(|error| format!("git rev-parse failed in {}: {error}", target.display()))?;
    Ok(output.status.success())
}

fn run_git_for_contrib<const N: usize>(cwd: Option<&Path>, args: [&str; N]) -> Result<(), String> {
    let mut command = Command::new("git");
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command
        .output()
        .map_err(|error| format!("git command failed to start: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(format!(
        "git command failed: {}",
        if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        }
    ))
}

fn wrap_codex_completion_hook(
    upstream_hook: Arc<dyn Fn(CodexSessionResult) + Send + Sync>,
) -> Arc<dyn Fn(CodexSessionResult) + Send + Sync> {
    Arc::new(move |result: CodexSessionResult| {
        set_background_status(
            codex_background_status_key(&result.session_id),
            codex_completion_followup_status_message(&result.session_id),
        );
        upstream_hook(result.clone());
        clear_background_status(&codex_background_status_key(&result.session_id));
    })
}

pub fn codex_tools(
    config: AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook: Arc<dyn Fn(CodexSessionResult) + Send + Sync>,
) -> Vec<ExecutableTool> {
    let database_path = config.database_path.clone();
    let home_dir = config.home_dir.clone();
    let log_path = config.home_dir.join("logs").join("elroy.log");
    let codex_bin_for_inspect = codex_bin_override.clone();
    let codex_completion_hook_for_inspect = codex_completion_hook.clone();
    let inspect_elroy_with_codex = ExecutableTool::new(
        ToolSpec::new(
            "inspect_elroy_with_codex",
            "Run a read-only background Codex inspection session against the running Elroy source tree.",
            JsonSchema::object(
                [
                    ("prompt", json!({"type": "string"})),
                    ("include_logs", json!({"type": "boolean"})),
                    ("model", json!({"type": "string"})),
                ],
                ["prompt"],
            ),
        ),
        move |arguments| {
            let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "inspect_elroy_with_codex requires a string prompt",
                );
            };
            let include_logs = arguments
                .get("include_logs")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let model = arguments.get("model").and_then(Value::as_str);
            if let Err(error) = ensure_tool_database(&database_path) {
                return ToolExecutionResult::error(error);
            }
            let prompt = build_inspection_prompt(prompt, include_logs, &log_path);
            let repo_path = match std::env::current_dir() {
                Ok(repo_path) => repo_path,
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to resolve current repo: {error}"
                    ));
                }
            };
            let completion_hook =
                wrap_codex_completion_hook(codex_completion_hook_for_inspect.clone());
            let result = if let Some(codex_bin) = codex_bin_for_inspect.as_deref() {
                dispatch_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    &prompt,
                    Some(&repo_path),
                    model,
                    codex_bin,
                    Some(completion_hook),
                )
            } else {
                dispatch_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    &prompt,
                    Some(&repo_path),
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
    let codex_bin_for_contrib = codex_bin_override.clone();
    let codex_completion_hook_for_contrib = codex_completion_hook.clone();
    let edit_contrib_with_codex = ExecutableTool::new(
        ToolSpec::new(
            "edit_contrib_with_codex",
            "Run a background Codex edit session for the Elroy-home contrib plugin repository.",
            JsonSchema::object(
                [
                    ("prompt", json!({"type": "string"})),
                    ("model", json!({"type": "string"})),
                ],
                ["prompt"],
            ),
        ),
        move |arguments| {
            let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "edit_contrib_with_codex requires a string prompt",
                );
            };
            let model = arguments.get("model").and_then(Value::as_str);
            if let Err(error) = ensure_tool_database(&database_path) {
                return ToolExecutionResult::error(error);
            }
            let contrib_repo = match ensure_contrib_repo(&home_dir) {
                Ok(contrib_repo) => contrib_repo,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let prompt = build_contrib_edit_prompt(prompt, &contrib_repo);
            let completion_hook =
                wrap_codex_completion_hook(codex_completion_hook_for_contrib.clone());
            let result = if let Some(codex_bin) = codex_bin_for_contrib.as_deref() {
                dispatch_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    &prompt,
                    Some(&contrib_repo),
                    model,
                    codex_bin,
                    Some(completion_hook),
                )
            } else {
                dispatch_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    &prompt,
                    Some(&contrib_repo),
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

            let completion_hook =
                wrap_codex_completion_hook(codex_completion_hook_for_dispatch.clone());

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

            let completion_hook =
                wrap_codex_completion_hook(codex_completion_hook_for_resume.clone());

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
    let home_dir = config.home_dir.clone();
    let list_codex_sessions = ExecutableTool::new(
        ToolSpec::new(
            "list_codex_sessions",
            "List recently recorded Codex sessions for this local user.",
            JsonSchema::object(
                [
                    ("scope", json!({"type": "string"})),
                    ("repo_path", json!({"type": "string"})),
                    ("limit", json!({"type": "integer"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let limit = match codex_list_limit(&arguments, 5) {
                Ok(limit) => limit,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let scope = arguments.get("scope").and_then(Value::as_str);
            let repo_path = arguments.get("repo_path").and_then(Value::as_str);
            let repo_filter = match codex_scope_repo_filter(scope, repo_path, &home_dir) {
                Ok(repo_filter) => repo_filter,
                Err(error) => return ToolExecutionResult::error(error),
            };
            with_tool_connection(&database_path, |connection| {
                let sessions = list_recent_codex_sessions(
                    connection,
                    LOCAL_USER_TOKEN,
                    repo_filter.as_deref(),
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
        inspect_elroy_with_codex,
        edit_contrib_with_codex,
        dispatch_codex_session,
        resume_codex_session,
        list_codex_sessions,
        show_codex_session,
    ]
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::{build_inspection_prompt, codex_tools};
    use crate::{
        CodexCommandRecord, CodexSessionRecord, CodexSessionResult, CodexSessionUpdate,
        get_codex_session_by_thread_id, upsert_codex_session,
    };
    use elroy_config::AppConfig;
    use elroy_db::{LOCAL_USER_TOKEN, open_sqlite_connection, run_migrations};
    use elroy_tools::ExecutableToolRegistry;

    #[test]
    fn codex_tools_can_list_and_show_sessions() {
        let unique = format!(
            "elroy-rs-codex-tools-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::create_dir_all(home.join("contrib")).expect("contrib dir should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        upsert_codex_session(
            &mut connection,
            LOCAL_USER_TOKEN,
            "thread-123",
            &CodexSessionUpdate {
                repo_path: PathBuf::from("/tmp/sample"),
                worktree_path: Some(PathBuf::from("/tmp/.elroy-codex-worktrees/sample")),
                session_branch: Some("elroy-codex-abcd1234".to_string()),
                target_branch: Some("agent".to_string()),
                prompt: "Fix the parser".to_string(),
                summary: "Codex updated the parser.".to_string(),
                agent_message: "Parser update complete.".to_string(),
                status: "completed".to_string(),
                commands: vec![CodexCommandRecord {
                    command: "/bin/zsh -lc cargo test".to_string(),
                    exit_code: Some(0),
                    output_excerpt: "ok\n".to_string(),
                }],
                touched_paths: vec!["src/parser.rs".to_string()],
                dirty_paths_before: vec!["README.md".to_string()],
                dirty_paths_after: vec!["src/parser.rs".to_string()],
                session_file_path: Some("/tmp/.codex/sessions/thread-123.jsonl".to_string()),
            },
        )
        .expect("codex session should persist");
        upsert_codex_session(
            &mut connection,
            LOCAL_USER_TOKEN,
            "thread-contrib",
            &CodexSessionUpdate {
                repo_path: home.join("contrib"),
                worktree_path: None,
                session_branch: None,
                target_branch: None,
                prompt: "Edit contrib plugin".to_string(),
                summary: "Codex edited contrib plugin.".to_string(),
                agent_message: "Contrib update complete.".to_string(),
                status: "completed".to_string(),
                commands: vec![],
                touched_paths: vec!["tools.py".to_string()],
                dirty_paths_before: vec![],
                dirty_paths_after: vec!["tools.py".to_string()],
                session_file_path: None,
            },
        )
        .expect("contrib codex session should persist");
        let current_repo = std::env::current_dir().expect("current dir should resolve");
        upsert_codex_session(
            &mut connection,
            LOCAL_USER_TOKEN,
            "thread-elroy",
            &CodexSessionUpdate {
                repo_path: current_repo,
                worktree_path: None,
                session_branch: None,
                target_branch: None,
                prompt: "Inspect running source".to_string(),
                summary: "Codex inspected running source.".to_string(),
                agent_message: "Inspection complete.".to_string(),
                status: "completed".to_string(),
                commands: vec![],
                touched_paths: vec![],
                dirty_paths_before: vec![],
                dirty_paths_after: vec![],
                session_file_path: None,
            },
        )
        .expect("elroy codex session should persist");

        let registry = ExecutableToolRegistry::new(codex_tools(
            config,
            None,
            Arc::new(|_: CodexSessionResult| {}),
        ));
        let listed = registry.invoke("list_codex_sessions", "{\"limit\":5}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("thread-123"));
        assert!(listed.content.contains("/tmp/sample"));

        let filtered = registry.invoke(
            "list_codex_sessions",
            "{\"repo_path\":\"/tmp/sample\",\"limit\":5}",
        );
        assert!(!filtered.is_error);
        assert!(filtered.content.contains("thread-123"));

        let contrib_filtered =
            registry.invoke("list_codex_sessions", "{\"scope\":\"contrib\",\"limit\":5}");
        assert!(!contrib_filtered.is_error);
        assert!(contrib_filtered.content.contains("thread-contrib"));
        assert!(!contrib_filtered.content.contains("thread-123"));

        let elroy_filtered =
            registry.invoke("list_codex_sessions", "{\"scope\":\"elroy\",\"limit\":5}");
        assert!(!elroy_filtered.is_error);
        assert!(elroy_filtered.content.contains("thread-elroy"));
        assert!(!elroy_filtered.content.contains("thread-contrib"));

        let invalid_scope =
            registry.invoke("list_codex_sessions", "{\"scope\":\"other\",\"limit\":5}");
        assert!(invalid_scope.is_error);
        assert_eq!(
            invalid_scope.content,
            "scope must be 'contrib', 'elroy', or omitted"
        );

        let invalid_limit =
            registry.invoke("list_codex_sessions", "{\"scope\":\"contrib\",\"limit\":0}");
        assert!(invalid_limit.is_error);
        assert_eq!(invalid_limit.content, "limit must be at least 1");

        let shown = registry.invoke("show_codex_session", "{\"session_id\":\"thread-123\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("Fix the parser"));
        assert!(shown.content.contains("cargo test"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn inspection_prompt_includes_optional_log_context() {
        let unique = format!(
            "elroy-rs-codex-inspection-prompt-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let log_path = home.join("logs").join("elroy.log");
        fs::create_dir_all(log_path.parent().expect("log path should have parent"))
            .expect("log dir should be created");
        fs::write(&log_path, "older\nrecent traceback\n").expect("log should be written");

        let with_logs = build_inspection_prompt("why did startup fail?", true, &log_path);
        let without_logs = build_inspection_prompt("why did startup fail?", false, &log_path);

        assert!(with_logs.contains("Do not edit files"));
        assert!(with_logs.contains("Inspection request:\nwhy did startup fail?"));
        assert!(with_logs.contains(&log_path.display().to_string()));
        assert!(with_logs.contains("recent traceback"));
        assert!(!without_logs.contains("Elroy log path:"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn named_contrib_codex_tool_bootstraps_repo_and_persists_prompt_contract() {
        let unique = format!(
            "elroy-rs-codex-contrib-tool-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let bin_dir = home.join("bin");
        let database_path = home.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should be created");
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = home.join("memories");
        config.agenda_dir = home.join("agenda");
        config.database_path = database_path.clone();

        let registry = ExecutableToolRegistry::new(codex_tools(
            config,
            Some(bin_dir.join("codex")),
            Arc::new(|_: CodexSessionResult| {}),
        ));

        let result = registry.invoke(
            "edit_contrib_with_codex",
            "{\"prompt\":\"add generated contrib helper\",\"model\":\"test-model\"}",
        );
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("\"status\":\"running\""));
        assert!(home.join("contrib").join(".git").exists());

        let completed = wait_for_status(&database_path, "thread-123", "completed");
        assert_eq!(
            PathBuf::from(&completed.repo_path)
                .canonicalize()
                .expect("completed repo path should canonicalize"),
            home.join("contrib")
                .canonicalize()
                .expect("contrib path should canonicalize")
        );
        assert!(
            completed
                .latest_prompt
                .contains("Edit only the Elroy contrib plugin directory")
        );
        assert!(
            completed
                .latest_prompt
                .contains("Edit request:\nadd generated contrib helper")
        );
        assert_eq!(
            completed.latest_agent_message.as_deref(),
            Some("updated notes")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    fn wait_for_status(
        database_path: &Path,
        session_id: &str,
        expected_status: &str,
    ) -> CodexSessionRecord {
        let started = Instant::now();
        loop {
            let connection =
                open_sqlite_connection(database_path).expect("database should open for polling");
            if let Some(record) =
                get_codex_session_by_thread_id(&connection, LOCAL_USER_TOKEN, session_id)
                    .expect("codex session should query")
                && record.status == expected_status
            {
                return record;
            }
            assert!(
                started.elapsed() <= Duration::from_secs(5),
                "timed out waiting for {session_id} to become {expected_status}"
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn write_fake_codex_script(path: &Path) {
        let script = r#"#!/bin/sh
mode="dispatch"
prompt=""
for arg in "$@"; do
  if [ "$arg" = "resume" ]; then
    mode="resume"
  fi
  prompt="$arg"
done

if [ "$mode" = "resume" ]; then
  printf "after resume\n" > notes.txt
  echo '{"type":"thread.started","thread_id":"thread-123"}'
  echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"resume complete"}}'
  exit 0
fi

printf "%s\n" "$prompt" > generated_contrib_note.txt
echo '{"type":"thread.started","thread_id":"thread-123"}'
echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"updated notes"}}'
"#;
        fs::write(path, script).expect("script should be written");
        let mut permissions = fs::metadata(path)
            .expect("script metadata should load")
            .permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            permissions.set_mode(0o755);
        }
        fs::set_permissions(path, permissions).expect("script should be executable");
    }
}
