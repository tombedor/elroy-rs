use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Local;
use serde_json::{Value, json};

use crate::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};

const DEFAULT_MAX_LIST_ENTRIES: usize = 50;
const DEFAULT_MAX_LIST_DEPTH: usize = 2;
const DEFAULT_READ_LINE_LIMIT: usize = 200;
const DEFAULT_RESTART_RESUME_PROMPT: &str =
    "Elroy just restarted. Send a brief message that you are back and ready to continue.";

type RestartRequester = dyn Fn(&str) -> Result<(), String> + Send + Sync;
type CommandLister = dyn Fn() -> Vec<(String, String)> + Send + Sync;
type ConfigReportRenderer = dyn Fn() -> String + Send + Sync;

#[derive(Clone)]
pub struct BaseToolCallbacks {
    pub request_restart: Arc<RestartRequester>,
    pub list_commands: Arc<CommandLister>,
    pub render_config_report: Arc<ConfigReportRenderer>,
    pub log_path: PathBuf,
}

pub fn base_tools(callbacks: BaseToolCallbacks) -> Vec<ExecutableTool> {
    let get_current_date = ExecutableTool::new(
        ToolSpec::new(
            "get_current_date",
            "Return the current local date and time.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            ToolExecutionResult::success(
                Local::now().format("%A, %B %d, %Y %I:%M %p %Z").to_string(),
            )
        },
    );

    let pwd = ExecutableTool::new(
        ToolSpec::new(
            "pwd",
            "Return the current working directory for filesystem tool calls.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| match std::env::current_dir() {
            Ok(path) => ToolExecutionResult::success(path.display().to_string()),
            Err(error) => ToolExecutionResult::error(format!(
                "Unable to resolve current working directory: {error}"
            )),
        },
    );

    let ls = ExecutableTool::new(
        ToolSpec::new(
            "ls",
            "List a file or directory, with bounded recursive expansion for directories.",
            JsonSchema::object(
                [
                    ("path", json!({"type": "string"})),
                    ("recursive", json!({"type": "boolean"})),
                    ("max_entries", json!({"type": "integer"})),
                    ("max_depth", json!({"type": "integer"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            let recursive = arguments
                .get("recursive")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let max_entries = arguments
                .get("max_entries")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_MAX_LIST_ENTRIES);
            let max_depth = arguments
                .get("max_depth")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(DEFAULT_MAX_LIST_DEPTH);
            if max_entries < 1 {
                return ToolExecutionResult::error("max_entries must be at least 1");
            }
            let target = match resolve_filesystem_tool_path(path) {
                Ok(path) => path,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let target_type = filesystem_entry_type(&target);
            if target_type != "dir" {
                let entry = match build_filesystem_entry(&target) {
                    Ok(entry) => entry,
                    Err(error) => return ToolExecutionResult::error(error),
                };
                return ToolExecutionResult::success(
                    json!({
                        "path": filesystem_display_path(&target),
                        "type": target_type,
                        "recursive": false,
                        "max_entries": max_entries,
                        "max_depth": max_depth,
                        "truncated": false,
                        "entries": [entry],
                    })
                    .to_string(),
                );
            }

            let mut entries = Vec::new();
            let mut truncated = false;
            if let Err(error) = walk_directory(
                &target,
                0,
                recursive,
                max_depth,
                max_entries,
                &mut entries,
                &mut truncated,
            ) {
                return ToolExecutionResult::error(error);
            }

            ToolExecutionResult::success(
                json!({
                    "path": filesystem_display_path(&target),
                    "type": target_type,
                    "recursive": recursive,
                    "max_entries": max_entries,
                    "max_depth": max_depth,
                    "truncated": truncated,
                    "entries": entries,
                })
                .to_string(),
            )
        },
    );

    let read_file = ExecutableTool::new(
        ToolSpec::new(
            "read_file",
            "Read a text file, optionally constrained to a line range.",
            JsonSchema::object(
                [
                    ("path", json!({"type": "string"})),
                    ("start_line", json!({"type": "integer"})),
                    ("end_line", json!({"type": "integer"})),
                ],
                ["path"],
            ),
        ),
        move |arguments| {
            let Some(path) = arguments.get("path").and_then(Value::as_str) else {
                return ToolExecutionResult::error("read_file requires a string path");
            };
            let start_line = match parse_optional_line_number_argument(&arguments, "start_line") {
                Ok(Some(value)) => value,
                Ok(None) => 1,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let end_line = match parse_optional_line_number_argument(&arguments, "end_line") {
                Ok(value) => value,
                Err(error) => return ToolExecutionResult::error(error),
            };
            if start_line < 1 {
                return ToolExecutionResult::error("start_line must be at least 1");
            }
            if end_line.is_some_and(|value| value < start_line) {
                return ToolExecutionResult::error(
                    "end_line must be greater than or equal to start_line",
                );
            }
            let target = match resolve_filesystem_tool_path(path) {
                Ok(path) => path,
                Err(error) => return ToolExecutionResult::error(error),
            };
            if target.is_dir() {
                return ToolExecutionResult::error(format!(
                    "Path is a directory, not a file: {path}"
                ));
            }
            let content = match std::fs::read_to_string(&target) {
                Ok(content) => content,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                    return ToolExecutionResult::error(format!(
                        "Unable to decode file as text: {path}"
                    ));
                }
                Err(_) => {
                    return ToolExecutionResult::error(format!("Unable to read file: {path}"));
                }
            };
            let lines = content.lines().collect::<Vec<_>>();
            let total_lines = lines.len();
            let end_line = end_line.unwrap_or(start_line + DEFAULT_READ_LINE_LIMIT as i64 - 1);
            let start_index = (start_line as usize).saturating_sub(1);
            let end_index = usize::min(end_line as usize, total_lines);
            let numbered_lines = if start_index >= total_lines {
                String::new()
            } else {
                lines[start_index..end_index]
                    .iter()
                    .enumerate()
                    .map(|(offset, line)| format!("{}: {}", start_line + offset as i64, line))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            ToolExecutionResult::success(
                json!({
                    "path": filesystem_display_path(&target),
                    "start_line": start_line,
                    "end_line": end_index,
                    "total_lines": total_lines,
                    "truncated": end_index < total_lines,
                    "content": numbered_lines,
                })
                .to_string(),
            )
        },
    );

    let request_restart = callbacks.request_restart.clone();
    let restart_session = ExecutableTool::new(
        ToolSpec::new(
            "restart_session",
            "Restart the active Elroy session after the current response completes.",
            JsonSchema::object(
                [("resume_message", json!({"type": "string"}))],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let resume_message = arguments
                .get("resume_message")
                .and_then(Value::as_str)
                .unwrap_or(DEFAULT_RESTART_RESUME_PROMPT);
            match request_restart(resume_message) {
                Ok(()) => ToolExecutionResult::success(
                    "Restart scheduled. Elroy will restart after this response completes."
                        .to_string(),
                ),
                Err(error) => ToolExecutionResult::error(error),
            }
        },
    );

    let render_config_report = callbacks.render_config_report.clone();
    let print_config = ExecutableTool::new(
        ToolSpec::new(
            "print_config",
            "Print the current Elroy configuration in a formatted report.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| ToolExecutionResult::success(render_config_report()),
    );

    let log_path = callbacks.log_path.clone();
    let tail_elroy_logs = ExecutableTool::new(
        ToolSpec::new(
            "tail_elroy_logs",
            "Return the last lines of the Elroy log file.",
            JsonSchema::object([("lines", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let lines = arguments.get("lines").and_then(Value::as_i64).unwrap_or(20) as usize;
            match std::fs::read_to_string(&log_path) {
                Ok(content) => {
                    let tail = content
                        .lines()
                        .rev()
                        .take(lines)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("\n");
                    ToolExecutionResult::success(if tail.is_empty() {
                        String::new()
                    } else {
                        format!("{tail}\n")
                    })
                }
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to read log file {}: {}",
                    log_path.display(),
                    error
                )),
            }
        },
    );

    let list_commands = callbacks.list_commands.clone();
    let get_help = ExecutableTool::new(
        ToolSpec::new(
            "get_help",
            "Print the available system commands.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut commands = list_commands();
            commands.sort_by(|left, right| left.0.cmp(&right.0));

            let rows = commands
                .into_iter()
                .map(|(name, description)| vec![name, description])
                .collect::<Vec<_>>();
            ToolExecutionResult::success(render_plain_text_table(
                "Available Slash Commands",
                &["Command", "Description"],
                &rows,
            ))
        },
    );

    vec![
        get_current_date,
        pwd,
        ls,
        read_file,
        restart_session,
        tail_elroy_logs,
        get_help,
        print_config,
    ]
}

fn parse_optional_line_number_argument(
    arguments: &Value,
    key: &str,
) -> Result<Option<i64>, String> {
    let Some(value) = arguments.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => number
            .as_i64()
            .ok_or_else(|| format!("{key} must be an integer"))
            .map(Some),
        Value::String(raw) => raw
            .parse::<i64>()
            .map(Some)
            .map_err(|_| format!("{key} must be an integer")),
        _ => Err(format!("{key} must be an integer")),
    }
}

pub fn render_plain_text_table(title: &str, headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths = headers
        .iter()
        .map(|header| header.len())
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if let Some(width) = widths.get_mut(index) {
                *width = (*width).max(cell.len());
            }
        }
    }

    let render_row = |cells: &[String]| {
        cells
            .iter()
            .enumerate()
            .map(|(index, cell)| format!("{cell:<width$}", width = widths[index]))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let header_cells = headers
        .iter()
        .map(|header| (*header).to_string())
        .collect::<Vec<_>>();
    let separator = widths
        .iter()
        .map(|width| "-".repeat(*width))
        .collect::<Vec<_>>()
        .join("-+-");

    let mut lines = vec![
        title.to_string(),
        String::new(),
        render_row(&header_cells),
        separator,
    ];
    for row in rows {
        lines.push(render_row(row));
    }
    lines.join("\n")
}

fn filesystem_display_path(target: &Path) -> String {
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    target
        .strip_prefix(&current_dir)
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|_| target.display().to_string())
}

fn filesystem_entry_type(target: &Path) -> &'static str {
    if target.is_symlink() {
        "symlink"
    } else if target.is_dir() {
        "dir"
    } else {
        "file"
    }
}

fn resolve_filesystem_tool_path(path: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(path);
    let target = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("Unable to resolve current working directory: {error}"))?
            .join(candidate)
    };
    let resolved = target
        .canonicalize()
        .map_err(|_| format!("Path does not exist: {path}"))?;
    Ok(resolved)
}

fn build_filesystem_entry(target: &Path) -> Result<Value, String> {
    let size_bytes = if target.is_dir() {
        None
    } else {
        Some(
            target
                .metadata()
                .map_err(|error| format!("Unable to inspect path: {error}"))?
                .len(),
        )
    };
    Ok(json!({
        "path": filesystem_display_path(target),
        "type": filesystem_entry_type(target),
        "size_bytes": size_bytes,
    }))
}

fn walk_directory(
    directory: &Path,
    depth: usize,
    recursive: bool,
    max_depth: usize,
    max_entries: usize,
    entries: &mut Vec<Value>,
    truncated: &mut bool,
) -> Result<bool, String> {
    let mut children = directory
        .read_dir()
        .map_err(|error| format!("Unable to inspect path: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Unable to inspect path: {error}"))?;
    children.sort_by(|left, right| {
        let left_path = left.path();
        let right_path = right.path();
        (
            filesystem_entry_type(&left_path) != "dir",
            left.file_name().to_string_lossy().to_lowercase(),
            left.file_name().to_string_lossy().to_string(),
        )
            .cmp(&(
                filesystem_entry_type(&right_path) != "dir",
                right.file_name().to_string_lossy().to_lowercase(),
                right.file_name().to_string_lossy().to_string(),
            ))
    });
    for child in children {
        let child_path = child.path();
        entries.push(build_filesystem_entry(&child_path)?);
        if entries.len() >= max_entries {
            *truncated = true;
            return Ok(false);
        }
        if recursive
            && depth < max_depth
            && child_path.is_dir()
            && !child_path.is_symlink()
            && !walk_directory(
                &child_path,
                depth + 1,
                recursive,
                max_depth,
                max_entries,
                entries,
                truncated,
            )?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use super::{BaseToolCallbacks, DEFAULT_RESTART_RESUME_PROMPT, base_tools};
    use crate::ExecutableToolRegistry;

    fn test_callbacks(
        restart_requests: Arc<Mutex<Vec<String>>>,
        log_path: PathBuf,
    ) -> BaseToolCallbacks {
        BaseToolCallbacks {
            request_restart: Arc::new(move |resume_message| {
                restart_requests
                    .lock()
                    .expect("restart capture should lock")
                    .push(resume_message.to_string());
                Ok(())
            }),
            list_commands: Arc::new(|| {
                vec![
                    (
                        "tail_elroy_logs".to_string(),
                        "Return the last lines of the Elroy log file.".to_string(),
                    ),
                    (
                        "get_help".to_string(),
                        "Print the available system commands.".to_string(),
                    ),
                ]
            }),
            render_config_report: Arc::new(|| "config report".to_string()),
            log_path,
        }
    }

    #[test]
    fn base_tools_cover_filesystem_and_time_behaviors() {
        let unique = format!(
            "elroy-rs-tools-base-filesystem-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let notes_dir = home.join("notes");
        fs::create_dir_all(&notes_dir).expect("notes dir should be created");
        fs::write(
            notes_dir.join("todo.txt"),
            "alpha\nbeta\ngamma\ndelta\nepsilon\n",
        )
        .expect("fixture file should be written");

        let restart_requests = Arc::new(Mutex::new(Vec::new()));
        let registry = ExecutableToolRegistry::new(base_tools(test_callbacks(
            restart_requests,
            home.join("elroy.log"),
        )));

        let previous_dir = std::env::current_dir().expect("cwd should resolve");
        std::env::set_current_dir(&home).expect("cwd should switch");

        let current_date = registry.invoke("get_current_date", "{}");
        let pwd = registry.invoke("pwd", "{}");
        let listing = registry.invoke(
            "ls",
            "{\"path\":\"notes\",\"recursive\":true,\"max_entries\":10,\"max_depth\":2}",
        );
        let file = registry.invoke(
            "read_file",
            "{\"path\":\"notes/todo.txt\",\"start_line\":2,\"end_line\":3}",
        );
        let file_with_string_lines = registry.invoke(
            "read_file",
            "{\"path\":\"notes/todo.txt\",\"start_line\":\"2\",\"end_line\":\"3\"}",
        );
        let bad_range = registry.invoke(
            "read_file",
            "{\"path\":\"notes/todo.txt\",\"start_line\":3,\"end_line\":2}",
        );

        std::env::set_current_dir(previous_dir).expect("cwd should restore");

        assert!(!current_date.is_error);
        assert!(current_date.content.contains(","));
        assert!(!pwd.is_error);
        assert_eq!(
            PathBuf::from(&pwd.content)
                .canonicalize()
                .expect("pwd result should canonicalize"),
            home.canonicalize().expect("home should canonicalize")
        );
        assert!(!listing.is_error);
        assert!(listing.content.contains("\"path\":\"notes\""));
        assert!(listing.content.contains("\"path\":\"notes/todo.txt\""));
        assert!(!file.is_error);
        assert!(file.content.contains("\"start_line\":2"));
        assert!(file.content.contains("2: beta\\n3: gamma"));
        assert!(!file_with_string_lines.is_error);
        assert!(file_with_string_lines.content.contains("\"start_line\":2"));
        assert!(
            file_with_string_lines
                .content
                .contains("2: beta\\n3: gamma")
        );
        assert!(bad_range.is_error);
        assert!(
            bad_range
                .content
                .contains("end_line must be greater than or equal to start_line")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn restart_session_uses_callback_and_default_prompt() {
        let restart_requests = Arc::new(Mutex::new(Vec::new()));
        let registry = ExecutableToolRegistry::new(base_tools(test_callbacks(
            restart_requests.clone(),
            std::env::temp_dir().join("unused-elroy.log"),
        )));

        let explicit = registry.invoke(
            "restart_session",
            "{\"resume_message\":\"Restarted successfully. Ready to continue.\"}",
        );
        let defaulted = registry.invoke("restart_session", "{}");

        assert!(!explicit.is_error);
        assert_eq!(
            explicit.content,
            "Restart scheduled. Elroy will restart after this response completes."
        );
        assert!(!defaulted.is_error);
        assert_eq!(
            restart_requests
                .lock()
                .expect("restart capture should lock")
                .as_slice(),
            &[
                "Restarted successfully. Ready to continue.".to_string(),
                DEFAULT_RESTART_RESUME_PROMPT.to_string(),
            ]
        );
    }

    #[test]
    fn help_tail_and_print_config_use_base_callbacks() {
        let unique = format!(
            "elroy-rs-tools-base-help-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(&home).expect("home should be created");
        let log_path = home.join("elroy.log");
        fs::write(&log_path, "line one\nline two\nline three\n")
            .expect("log file should be written");

        let restart_requests = Arc::new(Mutex::new(Vec::new()));
        let registry =
            ExecutableToolRegistry::new(base_tools(test_callbacks(restart_requests, log_path)));

        let help = registry.invoke("get_help", "{}");
        let printed = registry.invoke("print_config", "{}");
        let tailed = registry.invoke("tail_elroy_logs", "{\"lines\":2}");

        assert!(!help.is_error);
        assert!(help.content.contains("Available Slash Commands"));
        assert!(help.content.contains("Command"));
        assert!(help.content.contains("Description"));
        assert!(help.content.contains("get_help"));
        assert!(help.content.contains("tail_elroy_logs"));

        assert!(!printed.is_error);
        assert_eq!(printed.content, "config report");

        assert!(!tailed.is_error);
        assert_eq!(tailed.content, "line two\nline three\n");

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
