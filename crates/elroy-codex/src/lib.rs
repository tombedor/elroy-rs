use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use rusqlite::{Connection, params};
use serde_json::Value;

const AGENT_BRANCH: &str = "agent";
const WORKTREE_ROOT_DIRNAME: &str = ".elroy-codex-worktrees";
const MAX_COMMAND_OUTPUT_CHARS: usize = 400;
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexSessionResult {
    pub session_id: String,
    pub repo_path: String,
    pub worktree_path: Option<String>,
    pub session_branch: Option<String>,
    pub target_branch: Option<String>,
    pub status: String,
    pub final_message: String,
    pub summary: String,
    pub touched_paths: Vec<String>,
    pub dirty_paths_before: Vec<String>,
    pub dirty_paths_after: Vec<String>,
    pub commands: Vec<CodexCommandRecord>,
    pub session_file_path: Option<String>,
    pub resume_command: String,
    pub running_in_background: bool,
}

#[derive(Debug)]
pub enum CodexWorkflowError {
    Io(std::io::Error),
    Sql(rusqlite::Error),
    Json(serde_json::Error),
    InvalidRepo(String),
    Git(String),
    Codex(String),
    UnknownSession(String),
    SessionStillRunning(String),
    SessionMissingWorkspace(String),
}

impl std::fmt::Display for CodexWorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Sql(error) => write!(f, "{error}"),
            Self::Json(error) => write!(f, "{error}"),
            Self::InvalidRepo(error) => write!(f, "{error}"),
            Self::Git(error) => write!(f, "{error}"),
            Self::Codex(error) => write!(f, "{error}"),
            Self::UnknownSession(error) => write!(f, "{error}"),
            Self::SessionStillRunning(error) => write!(f, "{error}"),
            Self::SessionMissingWorkspace(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CodexWorkflowError {}

impl From<std::io::Error> for CodexWorkflowError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<rusqlite::Error> for CodexWorkflowError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sql(value)
    }
}

impl From<serde_json::Error> for CodexWorkflowError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone)]
struct RepoSnapshot {
    dirty_paths: Vec<String>,
}

#[derive(Debug, Clone)]
struct SessionWorkspace {
    source_repo_root: PathBuf,
    worktree_root: PathBuf,
    session_branch: String,
    target_branch: String,
    agent_worktree: PathBuf,
}

struct BackgroundRunContext {
    database_path: PathBuf,
    user_token: String,
    prompt: String,
    workspace: SessionWorkspace,
    before: RepoSnapshot,
    child: Child,
    stdout: ChildOutputReader,
    stderr: Option<ChildStderr>,
    initial_stdout: String,
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

pub fn dispatch_codex_session(
    database_path: &Path,
    user_token: &str,
    prompt: &str,
    repo_path: Option<&Path>,
    model: Option<&str>,
) -> Result<CodexSessionResult, CodexWorkflowError> {
    dispatch_codex_session_with_bin(
        database_path,
        user_token,
        prompt,
        repo_path,
        model,
        &codex_bin(),
    )
}

pub fn dispatch_codex_session_with_bin(
    database_path: &Path,
    user_token: &str,
    prompt: &str,
    repo_path: Option<&Path>,
    model: Option<&str>,
    codex_bin: &Path,
) -> Result<CodexSessionResult, CodexWorkflowError> {
    let source_repo_root = resolve_repo_path(repo_path)?;
    let workspace = create_session_workspace(&source_repo_root)?;
    let before = snapshot_repo(&workspace.worktree_root)?;
    let args = build_codex_args(prompt, &workspace.worktree_root, None, model);
    let (child, mut stdout, stderr) =
        spawn_codex_process(codex_bin, &args, &workspace.worktree_root)?;
    let (resolved_session_id, initial_stdout) = read_until_thread_started(stdout.take_reader())?;

    let running_result = build_running_result(&resolved_session_id, &workspace, &before);
    persist_codex_result(database_path, user_token, prompt, &running_result)?;

    let context = BackgroundRunContext {
        database_path: database_path.to_path_buf(),
        user_token: user_token.to_string(),
        prompt: prompt.to_string(),
        workspace,
        before,
        child,
        stdout,
        stderr,
        initial_stdout,
    };
    thread::Builder::new()
        .name(format!("codex-session-{resolved_session_id}"))
        .spawn(move || complete_codex_session_in_background(context))?;

    Ok(running_result)
}

pub fn resume_codex_session(
    database_path: &Path,
    user_token: &str,
    session_id: &str,
    prompt: &str,
    model: Option<&str>,
) -> Result<CodexSessionResult, CodexWorkflowError> {
    resume_codex_session_with_bin(
        database_path,
        user_token,
        session_id,
        prompt,
        model,
        &codex_bin(),
    )
}

pub fn resume_codex_session_with_bin(
    database_path: &Path,
    user_token: &str,
    session_id: &str,
    prompt: &str,
    model: Option<&str>,
    codex_bin: &Path,
) -> Result<CodexSessionResult, CodexWorkflowError> {
    let connection = Connection::open(database_path)?;
    let Some(record) = get_codex_session_by_thread_id(&connection, user_token, session_id)? else {
        return Err(CodexWorkflowError::UnknownSession(format!(
            "Unknown Codex session '{session_id}'. Use list_codex_sessions to inspect available sessions."
        )));
    };
    if record.status == "running" {
        return Err(CodexWorkflowError::SessionStillRunning(format!(
            "Codex session '{session_id}' is still running."
        )));
    }
    let Some(session_branch) = record.session_branch.clone() else {
        return Err(CodexWorkflowError::SessionMissingWorkspace(format!(
            "Codex session '{session_id}' is missing branch/worktree metadata and cannot be resumed."
        )));
    };
    let Some(worktree_path) = record.worktree_path.clone() else {
        return Err(CodexWorkflowError::SessionMissingWorkspace(format!(
            "Codex session '{session_id}' is missing branch/worktree metadata and cannot be resumed."
        )));
    };
    let source_repo_root = PathBuf::from(&record.repo_path);
    let target_branch = ensure_existing_session_workspace(
        &source_repo_root,
        &session_branch,
        Path::new(&worktree_path),
    )?;
    let workspace = SessionWorkspace {
        source_repo_root,
        worktree_root: PathBuf::from(&worktree_path),
        session_branch,
        target_branch,
        agent_worktree: agent_worktree_path(Path::new(&record.repo_path))?,
    };
    drop(connection);

    let before = snapshot_repo(&workspace.worktree_root)?;
    let args = build_codex_args(prompt, &workspace.worktree_root, Some(session_id), model);
    let (child, mut stdout, stderr) =
        spawn_codex_process(codex_bin, &args, &workspace.worktree_root)?;
    let (resolved_session_id, initial_stdout) = read_until_thread_started(stdout.take_reader())?;

    let running_result = build_running_result(&resolved_session_id, &workspace, &before);
    persist_codex_result(database_path, user_token, prompt, &running_result)?;

    let context = BackgroundRunContext {
        database_path: database_path.to_path_buf(),
        user_token: user_token.to_string(),
        prompt: prompt.to_string(),
        workspace,
        before,
        child,
        stdout,
        stderr,
        initial_stdout,
    };
    thread::Builder::new()
        .name(format!("codex-session-{resolved_session_id}"))
        .spawn(move || complete_codex_session_in_background(context))?;

    Ok(running_result)
}

fn complete_codex_session_in_background(mut context: BackgroundRunContext) {
    let result = (|| -> Result<CodexSessionResult, CodexWorkflowError> {
        let mut stdout_tail = String::new();
        context.stdout.read_to_string(&mut stdout_tail)?;

        let mut stderr_text = String::new();
        if let Some(stderr) = context.stderr.as_mut() {
            stderr.read_to_string(&mut stderr_text)?;
        }
        let status = context.child.wait()?;
        let stdout = format!("{}{}", context.initial_stdout, stdout_tail);
        if !status.success() {
            let session_id =
                parse_thread_started(&stdout)?.unwrap_or_else(|| "unknown".to_string());
            return Ok(build_failed_result(
                &session_id,
                &context.workspace,
                &context.before,
                if stderr_text.trim().is_empty() {
                    stdout.trim().to_string()
                } else {
                    stderr_text.trim().to_string()
                },
            ));
        }

        let mut result = build_result_from_stdout(&stdout, &context.workspace, &context.before)?;
        let commit_note = commit_session_worktree_if_needed(
            &context.workspace.worktree_root,
            &context.workspace.session_branch,
        )?;
        let merge_note = merge_session_branch_into_agent(
            &context.workspace.source_repo_root,
            &context.workspace.agent_worktree,
            &context.workspace.session_branch,
            &context.workspace.target_branch,
        )?;
        if let Some(note) = commit_note.or(merge_note) {
            result.summary = format!("{}\n{}", result.summary, note);
        }
        Ok(result)
    })();

    let final_result = match result {
        Ok(result) => result,
        Err(error) => build_failed_result(
            "unknown",
            &context.workspace,
            &context.before,
            error.to_string(),
        ),
    };

    let _ = persist_codex_result(
        &context.database_path,
        &context.user_token,
        &context.prompt,
        &final_result,
    );
}

fn persist_codex_result(
    database_path: &Path,
    user_token: &str,
    prompt: &str,
    result: &CodexSessionResult,
) -> Result<(), CodexWorkflowError> {
    let mut connection = Connection::open(database_path)?;
    let update = CodexSessionUpdate {
        repo_path: PathBuf::from(&result.repo_path),
        worktree_path: result.worktree_path.as_ref().map(PathBuf::from),
        session_branch: result.session_branch.clone(),
        target_branch: result.target_branch.clone(),
        prompt: prompt.to_string(),
        summary: result.summary.clone(),
        agent_message: result.final_message.clone(),
        status: result.status.clone(),
        commands: result.commands.clone(),
        touched_paths: result.touched_paths.clone(),
        dirty_paths_before: result.dirty_paths_before.clone(),
        dirty_paths_after: result.dirty_paths_after.clone(),
        session_file_path: result.session_file_path.clone(),
    };
    upsert_codex_session(&mut connection, user_token, &result.session_id, &update)?;
    Ok(())
}

fn resolve_repo_path(repo_path: Option<&Path>) -> Result<PathBuf, CodexWorkflowError> {
    let candidate = match repo_path {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => development_root()?.join(path),
        None => std::env::current_dir()?,
    };
    let output = run_command([
        "git",
        "-C",
        &candidate.display().to_string(),
        "rev-parse",
        "--show-toplevel",
    ])?;
    Ok(PathBuf::from(output.trim()))
}

fn development_root() -> Result<PathBuf, CodexWorkflowError> {
    let cwd = std::env::current_dir()?;
    if let Some(root) = development_ancestor(&cwd) {
        return Ok(root);
    }
    Ok(cwd.parent().map(Path::to_path_buf).unwrap_or(cwd))
}

fn development_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    loop {
        if current
            .file_name()
            .is_some_and(|name| name == "development")
        {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

fn create_session_workspace(repo_root: &Path) -> Result<SessionWorkspace, CodexWorkflowError> {
    let target_branch = ensure_agent_branch_workspace(repo_root)?;
    let session_branch = unique_session_branch();
    let session_worktree = session_worktree_path(repo_root, &session_branch)?;
    ensure_parent_dir(&session_worktree)?;
    run_command([
        "git",
        "-C",
        &repo_root.display().to_string(),
        "worktree",
        "add",
        "-b",
        &session_branch,
        &session_worktree.display().to_string(),
        &target_branch,
    ])?;
    Ok(SessionWorkspace {
        source_repo_root: repo_root.to_path_buf(),
        worktree_root: session_worktree,
        session_branch,
        target_branch,
        agent_worktree: agent_worktree_path(repo_root)?,
    })
}

fn ensure_existing_session_workspace(
    repo_root: &Path,
    session_branch: &str,
    worktree_path: &Path,
) -> Result<String, CodexWorkflowError> {
    let target_branch = ensure_agent_branch_workspace(repo_root)?;
    if !worktree_path.exists() {
        ensure_parent_dir(worktree_path)?;
        run_command([
            "git",
            "-C",
            &repo_root.display().to_string(),
            "worktree",
            "add",
            &worktree_path.display().to_string(),
            session_branch,
        ])?;
    }
    Ok(target_branch)
}

fn ensure_agent_branch_workspace(repo_root: &Path) -> Result<String, CodexWorkflowError> {
    let agent_worktree = agent_worktree_path(repo_root)?;
    if !git_branch_exists(repo_root, AGENT_BRANCH)? {
        run_command([
            "git",
            "-C",
            &repo_root.display().to_string(),
            "branch",
            AGENT_BRANCH,
            "HEAD",
        ])?;
    }
    if !agent_worktree.exists() {
        ensure_parent_dir(&agent_worktree)?;
        run_command([
            "git",
            "-C",
            &repo_root.display().to_string(),
            "worktree",
            "add",
            &agent_worktree.display().to_string(),
            AGENT_BRANCH,
        ])?;
    }
    Ok(AGENT_BRANCH.to_string())
}

fn git_branch_exists(repo_root: &Path, branch_name: &str) -> Result<bool, CodexWorkflowError> {
    let output = Command::new("git")
        .args([
            "-C",
            &repo_root.display().to_string(),
            "show-ref",
            "--verify",
            &format!("refs/heads/{branch_name}"),
        ])
        .output()?;
    Ok(output.status.success())
}

fn worktree_root(repo_root: &Path) -> Result<PathBuf, CodexWorkflowError> {
    let development_root = development_root_for_repo(repo_root)?;
    Ok(development_root
        .join(WORKTREE_ROOT_DIRNAME)
        .join(repo_slug(repo_root)))
}

fn agent_worktree_path(repo_root: &Path) -> Result<PathBuf, CodexWorkflowError> {
    Ok(worktree_root(repo_root)?.join(AGENT_BRANCH))
}

fn session_worktree_path(
    repo_root: &Path,
    session_branch: &str,
) -> Result<PathBuf, CodexWorkflowError> {
    Ok(worktree_root(repo_root)?
        .join("sessions")
        .join(session_branch))
}

fn development_root_for_repo(repo_root: &Path) -> Result<PathBuf, CodexWorkflowError> {
    if let Some(root) = development_ancestor(repo_root) {
        Ok(root)
    } else {
        repo_root.parent().map(Path::to_path_buf).ok_or_else(|| {
            CodexWorkflowError::InvalidRepo("repository has no parent directory".to_string())
        })
    }
}

fn repo_slug(repo_root: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    repo_root.hash(&mut hasher);
    format!(
        "{}-{:x}",
        repo_root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("repo"),
        hasher.finish()
    )
}

fn unique_session_branch() -> String {
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let unix_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    format!("elroy-codex-{unix_nanos:x}{counter:x}")
}

fn ensure_parent_dir(path: &Path) -> Result<(), CodexWorkflowError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn snapshot_repo(repo_root: &Path) -> Result<RepoSnapshot, CodexWorkflowError> {
    let output = run_command([
        "git",
        "-C",
        &repo_root.display().to_string(),
        "status",
        "--porcelain=v1",
        "-uall",
    ])?;
    Ok(RepoSnapshot {
        dirty_paths: parse_status_paths(&output),
    })
}

fn parse_status_paths(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            if line.len() < 4 {
                return None;
            }
            let mut path_text = line[3..].to_string();
            if let Some((_, right)) = path_text.split_once(" -> ") {
                path_text = right.to_string();
            }
            Some(path_text)
        })
        .collect()
}

fn build_codex_args(
    prompt: &str,
    repo_root: &Path,
    session_id: Option<&str>,
    model: Option<&str>,
) -> Vec<OsString> {
    let mut args = Vec::new();
    args.push("exec".into());
    if let Some(model) = model {
        args.push("-m".into());
        args.push(model.into());
    }
    if let Some(session_id) = session_id {
        args.push("resume".into());
        args.push("--json".into());
        args.push("--dangerously-bypass-approvals-and-sandbox".into());
        args.push(session_id.into());
        args.push(prompt.into());
    } else {
        args.push("--json".into());
        args.push("--dangerously-bypass-approvals-and-sandbox".into());
        args.push("-C".into());
        args.push(repo_root.as_os_str().to_os_string());
        args.push(prompt.into());
    }
    args
}

fn spawn_codex_process(
    codex_bin: &Path,
    args: &[OsString],
    cwd: &Path,
) -> Result<(Child, ChildOutputReader, Option<ChildStderr>), CodexWorkflowError> {
    let mut command = Command::new(codex_bin);
    command
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CodexWorkflowError::Codex("Codex stdout was not captured.".to_string()))?;
    let stderr = child.stderr.take();
    Ok((child, ChildOutputReader::new(stdout), stderr))
}

fn codex_bin() -> PathBuf {
    std::env::var_os("ELROY_CODEX_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("codex"))
}

fn read_until_thread_started(
    stdout: &mut BufReader<ChildStdout>,
) -> Result<(String, String), CodexWorkflowError> {
    let mut accumulated = String::new();
    loop {
        let mut line = String::new();
        let bytes = stdout.read_line(&mut line)?;
        if bytes == 0 {
            return Err(CodexWorkflowError::Codex(
                "Codex exited before emitting a thread id.".to_string(),
            ));
        }
        accumulated.push_str(&line);
        if let Some(session_id) = parse_thread_started(&accumulated)? {
            return Ok((session_id, accumulated));
        }
    }
}

fn parse_thread_started(stdout: &str) -> Result<Option<String>, CodexWorkflowError> {
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let payload: Value = serde_json::from_str(line)?;
        if payload.get("type").and_then(Value::as_str) == Some("thread.started") {
            let thread_id = payload
                .get("thread_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !thread_id.is_empty() {
                return Ok(Some(thread_id));
            }
        }
    }
    Ok(None)
}

fn parse_codex_exec_output(
    stdout: &str,
) -> Result<(String, String, Vec<CodexCommandRecord>), CodexWorkflowError> {
    let mut thread_id = String::new();
    let mut final_message = String::new();
    let mut commands = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let payload: Value = serde_json::from_str(line)?;
        let event_type = payload.get("type").and_then(Value::as_str);
        if event_type == Some("thread.started") {
            thread_id = payload
                .get("thread_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            continue;
        }
        if event_type != Some("item.completed") {
            continue;
        }
        let item = payload.get("item").cloned().unwrap_or(Value::Null);
        let item_type = item.get("type").and_then(Value::as_str);
        if item_type == Some("agent_message") {
            final_message = item
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        } else if item_type == Some("command_execution") {
            commands.push(CodexCommandRecord {
                command: item
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                exit_code: item.get("exit_code").and_then(Value::as_i64),
                output_excerpt: item
                    .get("aggregated_output")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .chars()
                    .take(MAX_COMMAND_OUTPUT_CHARS)
                    .collect(),
            });
        }
    }

    if thread_id.is_empty() {
        return Err(CodexWorkflowError::Codex(
            "Codex did not emit a thread id.".to_string(),
        ));
    }
    if final_message.is_empty() {
        return Err(CodexWorkflowError::Codex(
            "Codex did not emit a final assistant message.".to_string(),
        ));
    }
    Ok((thread_id, final_message, commands))
}

fn build_running_result(
    session_id: &str,
    workspace: &SessionWorkspace,
    before: &RepoSnapshot,
) -> CodexSessionResult {
    CodexSessionResult {
        session_id: session_id.to_string(),
        repo_path: workspace.source_repo_root.display().to_string(),
        worktree_path: Some(workspace.worktree_root.display().to_string()),
        session_branch: Some(workspace.session_branch.clone()),
        target_branch: Some(workspace.target_branch.clone()),
        status: "running".to_string(),
        final_message: String::new(),
        summary: format!(
            "Codex session {session_id} is running asynchronously in isolated worktree {} on branch {}, targeting {} for source repo {}.",
            workspace.worktree_root.display(),
            workspace.session_branch,
            workspace.target_branch,
            workspace.source_repo_root.display(),
        ),
        touched_paths: Vec::new(),
        dirty_paths_before: before.dirty_paths.clone(),
        dirty_paths_after: before.dirty_paths.clone(),
        commands: Vec::new(),
        session_file_path: None,
        resume_command: format!(
            "codex resume {session_id} -C {}",
            workspace.worktree_root.display()
        ),
        running_in_background: true,
    }
}

fn build_failed_result(
    session_id: &str,
    workspace: &SessionWorkspace,
    before: &RepoSnapshot,
    message: String,
) -> CodexSessionResult {
    CodexSessionResult {
        session_id: session_id.to_string(),
        repo_path: workspace.source_repo_root.display().to_string(),
        worktree_path: Some(workspace.worktree_root.display().to_string()),
        session_branch: Some(workspace.session_branch.clone()),
        target_branch: Some(workspace.target_branch.clone()),
        status: "failed".to_string(),
        final_message: message.clone(),
        summary: format!(
            "Codex session {session_id} failed in isolated worktree {}: {}",
            workspace.worktree_root.display(),
            message
        ),
        touched_paths: Vec::new(),
        dirty_paths_before: before.dirty_paths.clone(),
        dirty_paths_after: before.dirty_paths.clone(),
        commands: Vec::new(),
        session_file_path: None,
        resume_command: format!(
            "codex resume {session_id} -C {}",
            workspace.worktree_root.display()
        ),
        running_in_background: false,
    }
}

fn build_result_from_stdout(
    stdout: &str,
    workspace: &SessionWorkspace,
    before: &RepoSnapshot,
) -> Result<CodexSessionResult, CodexWorkflowError> {
    let (session_id, final_message, commands) = parse_codex_exec_output(stdout)?;
    let after = snapshot_repo(&workspace.worktree_root)?;
    let touched_paths = touched_paths(before, &after);
    let summary = build_summary(
        &session_id,
        workspace,
        &touched_paths,
        &before.dirty_paths,
        &after.dirty_paths,
        &final_message,
    );
    Ok(CodexSessionResult {
        session_id: session_id.clone(),
        repo_path: workspace.source_repo_root.display().to_string(),
        worktree_path: Some(workspace.worktree_root.display().to_string()),
        session_branch: Some(workspace.session_branch.clone()),
        target_branch: Some(workspace.target_branch.clone()),
        status: "completed".to_string(),
        final_message,
        summary,
        touched_paths,
        dirty_paths_before: before.dirty_paths.clone(),
        dirty_paths_after: after.dirty_paths.clone(),
        commands,
        session_file_path: None,
        resume_command: format!(
            "codex resume {session_id} -C {}",
            workspace.worktree_root.display()
        ),
        running_in_background: false,
    })
}

fn touched_paths(before: &RepoSnapshot, after: &RepoSnapshot) -> Vec<String> {
    let mut paths = before.dirty_paths.clone();
    for path in &after.dirty_paths {
        if !paths.iter().any(|existing| existing == path) {
            paths.push(path.clone());
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn build_summary(
    session_id: &str,
    workspace: &SessionWorkspace,
    touched_paths: &[String],
    dirty_before: &[String],
    dirty_after: &[String],
    final_message: &str,
) -> String {
    let mut lines = vec![format!(
        "Codex session {session_id} ran in isolated worktree {} for source repo {}.",
        workspace.worktree_root.display(),
        workspace.source_repo_root.display()
    )];
    lines.push(format!("Session branch: {}", workspace.session_branch));
    lines.push(format!("Target agent branch: {}", workspace.target_branch));
    if touched_paths.is_empty() {
        lines.push("No repository changes were detected during this run.".to_string());
    } else {
        lines.push(format!(
            "Paths changed during this run: {}",
            touched_paths.join(", ")
        ));
    }
    if !dirty_before.is_empty() {
        lines.push(format!(
            "Repo was already dirty before run: {}",
            dirty_before.join(", ")
        ));
    }
    if !dirty_after.is_empty() {
        lines.push(format!("Dirty paths after run: {}", dirty_after.join(", ")));
    }
    lines.push(format!("Final message: {}", final_message.trim()));
    lines.join("\n")
}

fn commit_session_worktree_if_needed(
    worktree_root: &Path,
    session_branch: &str,
) -> Result<Option<String>, CodexWorkflowError> {
    let snapshot = snapshot_repo(worktree_root)?;
    if snapshot.dirty_paths.is_empty() {
        return Ok(None);
    }
    run_command([
        "git",
        "-C",
        &worktree_root.display().to_string(),
        "add",
        "-A",
    ])?;
    run_command([
        "git",
        "-C",
        &worktree_root.display().to_string(),
        "commit",
        "-m",
        &format!("Elroy Codex session updates ({session_branch})"),
    ])?;
    Ok(Some(format!(
        "Committed unmerged worktree changes on {session_branch} before integration."
    )))
}

fn merge_session_branch_into_agent(
    source_repo_root: &Path,
    agent_worktree: &Path,
    session_branch: &str,
    target_branch: &str,
) -> Result<Option<String>, CodexWorkflowError> {
    let merge = Command::new("git")
        .args([
            "-C",
            &agent_worktree.display().to_string(),
            "merge",
            "--no-ff",
            "--no-edit",
            session_branch,
        ])
        .output()?;
    if !merge.status.success() {
        let _ = Command::new("git")
            .args([
                "-C",
                &agent_worktree.display().to_string(),
                "merge",
                "--abort",
            ])
            .status();
        let error = String::from_utf8_lossy(&merge.stderr).trim().to_string();
        return Err(CodexWorkflowError::Git(format!(
            "Merge into {target_branch} failed for repo {}: {}",
            source_repo_root.display(),
            if error.is_empty() {
                "unknown git merge error"
            } else {
                &error
            }
        )));
    }
    Ok(Some(format!(
        "Merged session branch {session_branch} into {target_branch}."
    )))
}

fn run_command<const N: usize>(args: [&str; N]) -> Result<String, CodexWorkflowError> {
    let output = Command::new(args[0]).args(&args[1..]).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return Err(CodexWorkflowError::Git(if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!("command failed: {}", args.join(" "))
        }));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

struct ChildOutputReader {
    reader: BufReader<ChildStdout>,
}

impl ChildOutputReader {
    fn new(stdout: ChildStdout) -> Self {
        Self {
            reader: BufReader::new(stdout),
        }
    }

    fn take_reader(&mut self) -> &mut BufReader<ChildStdout> {
        &mut self.reader
    }
}

impl Read for ChildOutputReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
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
    use std::{
        fs,
        path::{Path, PathBuf},
        thread,
        time::{Duration, Instant},
    };

    use rusqlite::Connection;

    use super::{
        CodexCommandRecord, CodexSessionUpdate, dispatch_codex_session_with_bin,
        get_codex_session_by_thread_id, list_recent_codex_sessions, resume_codex_session_with_bin,
        upsert_codex_session,
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

    #[test]
    fn dispatch_codex_session_runs_in_background_and_persists_completion() {
        let unique = unique_name("elroy-rs-codex-dispatch");
        let root = std::env::temp_dir().join(unique);
        let development_root = root.join("development");
        let repo_root = development_root.join("sample");
        let bin_dir = root.join("bin");
        let database_path = root.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should exist");
        init_repo(&repo_root);
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut connection = Connection::open(&database_path).expect("sqlite should open");
        run_test_migrations(&mut connection);
        drop(connection);

        let result = dispatch_codex_session_with_bin(
            &database_path,
            "local-user",
            "update notes",
            Some(&repo_root),
            None,
            &bin_dir.join("codex"),
        )
        .expect("dispatch should succeed");

        assert_eq!(result.session_id, "thread-123");
        assert_eq!(result.status, "running");
        assert!(result.running_in_background);
        assert!(result.worktree_path.is_some());
        assert!(result.session_branch.is_some());
        assert_eq!(result.target_branch.as_deref(), Some("agent"));

        let completed = wait_for_status(&database_path, "local-user", "thread-123", "completed");
        assert_eq!(
            completed.latest_agent_message.as_deref(),
            Some("updated notes")
        );
        assert_eq!(completed.status, "completed");

        let agent_head = std::process::Command::new("git")
            .args([
                "-C",
                &repo_root.display().to_string(),
                "show",
                "agent:notes.txt",
            ])
            .output()
            .expect("git show should run");
        assert!(agent_head.status.success());
        assert_eq!(String::from_utf8_lossy(&agent_head.stdout), "after\n");

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn resume_and_list_codex_sessions_runs_async() {
        let unique = unique_name("elroy-rs-codex-resume");
        let root = std::env::temp_dir().join(unique);
        let development_root = root.join("development");
        let repo_root = development_root.join("sample");
        let bin_dir = root.join("bin");
        let database_path = root.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should exist");
        init_repo(&repo_root);
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut connection = Connection::open(&database_path).expect("sqlite should open");
        run_test_migrations(&mut connection);
        drop(connection);

        let initial = dispatch_codex_session_with_bin(
            &database_path,
            "local-user",
            "update notes",
            Some(&repo_root),
            None,
            &bin_dir.join("codex"),
        )
        .expect("initial dispatch should succeed");
        let _ = wait_for_status(&database_path, "local-user", "thread-123", "completed");

        let resumed = resume_codex_session_with_bin(
            &database_path,
            "local-user",
            "thread-123",
            "follow up",
            None,
            &bin_dir.join("codex"),
        )
        .expect("resume should succeed");
        let completed = wait_for_status(&database_path, "local-user", "thread-123", "completed");
        let listed = list_recent_codex_sessions(
            &Connection::open(&database_path).expect("db should open"),
            "local-user",
            None,
            5,
        )
        .expect("sessions should list");

        assert_eq!(initial.status, "running");
        assert_eq!(resumed.status, "running");
        assert_eq!(
            completed.latest_agent_message.as_deref(),
            Some("resume complete")
        );
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].thread_id, "thread-123");
        assert_eq!(listed[0].status, "completed");

        let agent_head = std::process::Command::new("git")
            .args([
                "-C",
                &repo_root.display().to_string(),
                "show",
                "agent:notes.txt",
            ])
            .output()
            .expect("git show should run");
        assert!(agent_head.status.success());
        assert_eq!(
            String::from_utf8_lossy(&agent_head.stdout),
            "after resume\n"
        );

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn resume_running_session_is_rejected() {
        let unique = unique_name("elroy-rs-codex-running");
        let root = std::env::temp_dir().join(unique);
        let development_root = root.join("development");
        let repo_root = development_root.join("sample");
        let bin_dir = root.join("bin");
        let database_path = root.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should exist");
        init_repo(&repo_root);
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut connection = Connection::open(&database_path).expect("sqlite should open");
        run_test_migrations(&mut connection);
        drop(connection);

        let dispatched = dispatch_codex_session_with_bin(
            &database_path,
            "local-user",
            "hold open",
            Some(&repo_root),
            None,
            &bin_dir.join("codex"),
        )
        .expect("dispatch should succeed");
        assert_eq!(dispatched.status, "running");

        let error = resume_codex_session_with_bin(
            &database_path,
            "local-user",
            "thread-123",
            "follow up",
            None,
            &bin_dir.join("codex"),
        )
        .expect_err("resume should reject running session");
        assert!(error.to_string().contains("still running"));

        thread::sleep(Duration::from_millis(2300));
        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn dispatch_allows_multiple_isolated_sessions_for_same_repo() {
        let unique = unique_name("elroy-rs-codex-multi");
        let root = std::env::temp_dir().join(unique);
        let development_root = root.join("development");
        let repo_root = development_root.join("sample");
        let bin_dir = root.join("bin");
        let database_path = root.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should exist");
        init_repo(&repo_root);
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut connection = Connection::open(&database_path).expect("sqlite should open");
        run_test_migrations(&mut connection);
        drop(connection);

        let first = dispatch_codex_session_with_bin(
            &database_path,
            "local-user",
            "first prompt",
            Some(&repo_root),
            None,
            &bin_dir.join("codex"),
        )
        .expect("first dispatch should succeed");
        let second = dispatch_codex_session_with_bin(
            &database_path,
            "local-user",
            "second prompt",
            Some(&repo_root),
            None,
            &bin_dir.join("codex"),
        )
        .expect("second dispatch should succeed");
        let listed = list_recent_codex_sessions(
            &Connection::open(&database_path).expect("db should open"),
            "local-user",
            None,
            10,
        )
        .expect("sessions should list");

        assert_eq!(first.status, "running");
        assert_eq!(second.status, "running");
        assert_ne!(first.session_id, second.session_id);
        assert_ne!(first.worktree_path, second.worktree_path);
        assert_ne!(first.session_branch, second.session_branch);
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().all(|record| record.status == "running"));

        thread::sleep(Duration::from_millis(2300));
        fs::remove_dir_all(root).expect("root should be removed");
    }

    fn init_repo(repo_root: &Path) {
        fs::create_dir_all(repo_root).expect("repo root should exist");
        git(repo_root, ["init"]);
        git(repo_root, ["config", "user.email", "test@example.com"]);
        git(repo_root, ["config", "user.name", "Test User"]);
        fs::write(repo_root.join("notes.txt"), "before\n").expect("notes should be written");
        git(repo_root, ["add", "notes.txt"]);
        git(repo_root, ["commit", "-m", "init"]);
    }

    fn git<const N: usize>(repo_root: &Path, args: [&str; N]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(repo_root)
            .output()
            .expect("git command should run");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
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

if [ "$prompt" = "hold open" ]; then
  echo '{"type":"thread.started","thread_id":"thread-123"}'
  sleep 2
  echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"slow complete"}}'
  exit 0
fi

if [ "$prompt" = "first prompt" ]; then
  echo '{"type":"thread.started","thread_id":"thread-1"}'
  sleep 2
  echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"first complete"}}'
  exit 0
fi

if [ "$prompt" = "second prompt" ]; then
  echo '{"type":"thread.started","thread_id":"thread-2"}'
  sleep 2
  echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"second complete"}}'
  exit 0
fi

printf "after\n" > notes.txt
pwd_out="$(pwd)"
echo '{"type":"thread.started","thread_id":"thread-123"}'
printf '{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"/bin/zsh -lc pwd","aggregated_output":"%s\\n","exit_code":0,"status":"completed"}}\n' "$pwd_out"
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

    fn unique_name(prefix: &str) -> String {
        format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        )
    }

    fn wait_for_status(
        database_path: &Path,
        user_token: &str,
        session_id: &str,
        expected_status: &str,
    ) -> super::CodexSessionRecord {
        let started = Instant::now();
        loop {
            let connection = Connection::open(database_path).expect("sqlite should open");
            let record = get_codex_session_by_thread_id(&connection, user_token, session_id)
                .expect("session should query")
                .expect("session should exist");
            if record.status == expected_status {
                return record;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "timed out waiting for status {expected_status}, last status {}",
                record.status
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
}
