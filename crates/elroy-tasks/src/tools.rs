use std::path::Path;

use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, TimeZone, Utc};
use elroy_config::{AppConfig, LlmProvider};
use elroy_context::load_validated_runtime_transcript;
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, LOCAL_USER_TOKEN, find_active_agenda_item_by_name,
    open_sqlite_connection, replace_context_messages, run_migrations,
};
use elroy_memory::sanitize_filename;
use elroy_recall::{
    context_task_tool_call_id, context_task_tool_messages, message_matches_tool_call_id,
    transcript_contains_context_task,
};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec, argument_limit};
use serde_json::{Value, json};

use super::{
    complete_task_file, create_task_file_with_schedule, delete_task_file, find_task_by_name,
    list_active_tasks, list_due_tasks, list_today_tasks, list_triggered_tasks, rename_task_file,
    update_task_text_file,
};

pub fn task_tools(config: AppConfig) -> Vec<ExecutableTool> {
    let config_for_task_write = config.clone();
    let create_task = ExecutableTool::new(
        ToolSpec::new(
            "create_task",
            "Create a new agenda-backed task and rebuild derived state.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("item_date", json!({"type": "string"})),
                    ("date", json!({"type": "string"})),
                    ("trigger_datetime", json!({"type": "string"})),
                    ("trigger_context", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_task requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_task requires string text");
            };
            if name.trim().is_empty() {
                return ToolExecutionResult::error("Task name cannot be empty");
            }
            let date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("date").and_then(Value::as_str));
            let date = match parse_optional_agenda_item_date(date) {
                Ok(date) => date,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let trigger_datetime = arguments.get("trigger_datetime").and_then(Value::as_str);
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);
            if let Some(trigger_datetime) = trigger_datetime {
                match parse_trigger_datetime_for_validation(trigger_datetime) {
                    Ok(parsed) if parsed < Utc::now() => {
                        return ToolExecutionResult::error(format!(
                            "Attempted to create a due item for {}, which is in the past. The current time is {}",
                            parsed,
                            Utc::now()
                        ));
                    }
                    Ok(_) => {}
                    Err(error) => return ToolExecutionResult::error(error),
                }
            }
            match (|| -> Result<String, std::io::Error> {
                let mut connection = open_sqlite_connection(&config_for_task_write.database_path)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                run_migrations(&mut connection)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if find_active_agenda_item_by_name(&connection, name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                    .is_some()
                {
                    return Err(std::io::Error::other(format!(
                        "Task '{name}' already exists"
                    )));
                }
                let path = create_task_file_with_schedule(
                    &config_for_task_write.agenda_dir,
                    name,
                    text,
                    date.as_deref(),
                    trigger_datetime,
                    trigger_context,
                )?;
                let logical_task_name = sanitize_filename(name).replace('_', " ");
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_task_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                connection
                    .execute(
                        "UPDATE agenda_items
                         SET name = ?1
                         WHERE file_path = ?2
                           AND is_active = 1",
                        rusqlite::params![logical_task_name, path.to_string_lossy().as_ref()],
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let task = find_active_agenda_item_by_name(&connection, &logical_task_name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                    .filter(|item| {
                        item.trigger_datetime.is_none() && item.trigger_context.is_none()
                    });
                if let Some(task) = task {
                    let mut transcript = load_validated_runtime_transcript(
                        &mut connection,
                        &config_for_task_write.assistant_name,
                        config_for_task_write.llm_provider() == LlmProvider::Anthropic,
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                    if !transcript_contains_context_task(&transcript, &task.name) {
                        transcript.extend(context_task_tool_messages(&task));
                        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
                            .map_err(|error| std::io::Error::other(error.to_string()))?;
                    }
                }
                Ok(logical_task_name)
            })() {
                Ok(logical_task_name) => ToolExecutionResult::success(format!(
                    "Task '{logical_task_name}' has been created."
                )),
                Err(error) if error.to_string().starts_with("Task '") => {
                    ToolExecutionResult::error(error.to_string())
                }
                Err(error) => ToolExecutionResult::error(format!("failed to create task: {error}")),
            }
        },
    );

    let config_for_task_text = config.clone();
    let update_task_text = ExecutableTool::new(
        ToolSpec::new(
            "update_task_text",
            "Replace the body text of one active task.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_task_text requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_task_text requires string text");
            };
            let result = mutate_task_file_from_config_with_result(
                &config_for_task_text,
                name,
                || format!("Active task '{name}' not found."),
                |path, _| {
                    update_task_text_file(path, text)?;
                    Ok(format!("Task '{name}' text has been updated."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_task_text, name, Some(name)) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_task_rename = config.clone();
    let rename_task = ExecutableTool::new(
        ToolSpec::new(
            "rename_task",
            "Rename one active task.",
            JsonSchema::object(
                [
                    ("old_name", json!({"type": "string"})),
                    ("new_name", json!({"type": "string"})),
                ],
                ["new_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("old_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("rename_task requires a string name");
            };
            let Some(new_name) = arguments.get("new_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("rename_task requires string new_name");
            };
            let result = mutate_task_file_from_config_with_result(
                &config_for_task_rename,
                name,
                || format!("Active task '{name}' not found."),
                |path, task_names| {
                    if task_names.iter().any(|existing| existing == new_name) {
                        return Err(std::io::Error::other(format!(
                            "Active task '{new_name}' already exists."
                        )));
                    }
                    let _renamed = rename_task_file(path, new_name)?;
                    Ok(format!("Task '{name}' has been renamed to '{new_name}'."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_task_rename, name, Some(new_name)) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_task_complete = config.clone();
    let complete_task = ExecutableTool::new(
        ToolSpec::new(
            "complete_task",
            "Mark one active task completed.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("closing_comment", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("complete_task requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            let result = mutate_task_file_from_config_with_result(
                &config_for_task_complete,
                name,
                || format!("Active task '{name}' not found."),
                |path, _| {
                    complete_task_file(path, closing_comment)?;
                    Ok(match closing_comment {
                        Some(closing_comment) => format!(
                            "Task '{name}' has been marked as completed. Comment: {closing_comment}"
                        ),
                        None => format!("Task '{name}' has been marked as completed."),
                    })
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_task_complete, name, None) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_task_delete = config.clone();
    let delete_task = ExecutableTool::new(
        ToolSpec::new(
            "delete_task",
            "Mark one active task deleted.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("closing_comment", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("delete_task requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            let result = mutate_task_file_from_config_with_result(
                &config_for_task_delete,
                name,
                || format!("Active task '{name}' not found."),
                |path, _| {
                    delete_task_file(path, closing_comment)?;
                    Ok(match closing_comment {
                        Some(closing_comment) => {
                            format!("Task '{name}' has been deleted. Comment: {closing_comment}")
                        }
                        None => format!("Task '{name}' has been deleted."),
                    })
                },
            );
            if result.is_error {
                return result;
            }
            let tool_call_id = context_task_tool_call_id(name);
            match remove_context_tool_messages_by_id(&config_for_task_delete, &tool_call_id) {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to remove task from context: {error}"
                )),
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_tasks = ExecutableTool::new(
        ToolSpec::new(
            "list_tasks",
            "List active agenda-backed tasks.",
            JsonSchema::object(std::iter::empty::<(&str, Value)>(), [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_active_tasks(connection, limit)?;
                let payload = items.into_iter().map(task_payload).collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload).expect("task payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_triggered_tasks_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_triggered_tasks",
            "List active tasks that have trigger metadata.",
            JsonSchema::object(std::iter::empty::<(&str, Value)>(), [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_triggered_tasks(connection, limit)?;
                let payload = items.into_iter().map(task_payload).collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("triggered task payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_due_tasks_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_due_tasks",
            "List active tasks whose trigger time is due.",
            JsonSchema::object(std::iter::empty::<(&str, Value)>(), [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            let now = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
            with_tool_connection(&database_path, |connection| {
                let items = list_due_tasks(connection, limit, &now)?;
                let payload = items.into_iter().map(task_payload).collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("due task payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_today_tasks_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_today_tasks",
            "List active tasks scheduled for today.",
            JsonSchema::object(std::iter::empty::<(&str, Value)>(), [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
            with_tool_connection(&database_path, |connection| {
                let items = list_today_tasks(connection, limit, &today)?;
                let payload = items.into_iter().map(task_payload).collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("today task payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let show_task = ExecutableTool::new(
        ToolSpec::new(
            "show_task",
            "Show one active task by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("show_task requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(item) = find_task_by_name(connection, name)? else {
                    return Ok(ToolExecutionResult::error(format!(
                        "Active task '{name}' not found."
                    )));
                };
                Ok(ToolExecutionResult::success(task_payload(item).to_string()))
            })
        },
    );

    vec![
        create_task,
        update_task_text,
        rename_task,
        complete_task,
        delete_task,
        list_tasks,
        list_triggered_tasks_tool,
        list_due_tasks_tool,
        list_today_tasks_tool,
        show_task,
    ]
}

fn task_payload(item: AgendaItemRecord) -> Value {
    json!({
        "name": item.name,
        "file_path": item.file_path,
        "agenda_date": item.agenda_date,
        "trigger_datetime": item.trigger_datetime,
        "trigger_context": item.trigger_context,
        "status": item.status,
        "checklist_total": item.checklist_total,
        "checklist_completed": item.checklist_completed,
        "body": item.body,
    })
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

fn parse_trigger_datetime_for_validation(raw: &str) -> Result<DateTime<Utc>, String> {
    let raw = raw.trim();
    if let Ok(parsed) = DateTime::parse_from_rfc3339(raw) {
        return Ok(parsed.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S") {
        let local = Local
            .from_local_datetime(&parsed)
            .single()
            .ok_or_else(|| format!("Invalid datetime format: '{raw}'"))?;
        return Ok(local.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        let local = Local
            .from_local_datetime(&parsed)
            .single()
            .ok_or_else(|| format!("Invalid datetime format: '{raw}'"))?;
        return Ok(local.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M") {
        let local = Local
            .from_local_datetime(&parsed)
            .single()
            .ok_or_else(|| format!("Invalid datetime format: '{raw}'"))?;
        return Ok(local.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        let naive = parsed
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| format!("Invalid datetime format: '{raw}'"))?;
        let local = Local
            .from_local_datetime(&naive)
            .single()
            .ok_or_else(|| format!("Invalid datetime format: '{raw}'"))?;
        return Ok(local.with_timezone(&Utc));
    }
    Err(format!(
        "Invalid datetime format: '{}'. Expected formats: 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD', or ISO 8601 format",
        raw
    ))
}

fn parse_agenda_item_date(raw: Option<&str>) -> Result<String, String> {
    match raw {
        Some(raw) => NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
            .map(|date| date.format("%Y-%m-%d").to_string())
            .map_err(|_| format!("Invalid date format '{raw}'. Use YYYY-MM-DD.")),
        None => Ok(Local::now().date_naive().format("%Y-%m-%d").to_string()),
    }
}

fn parse_optional_agenda_item_date(raw: Option<&str>) -> Result<Option<String>, String> {
    match raw {
        Some(raw) => parse_agenda_item_date(Some(raw)).map(Some),
        None => Ok(None),
    }
}

fn mutate_task_file_from_config_with_result(
    config: &AppConfig,
    name: &str,
    missing_message: impl FnOnce() -> String,
    operation: impl FnOnce(&Path, &[String]) -> std::io::Result<String>,
) -> ToolExecutionResult {
    let mut connection = match open_sqlite_connection(&config.database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    if let Err(error) = run_migrations(&mut connection) {
        return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
    }
    let tasks = match list_active_tasks(&connection, 1_000) {
        Ok(tasks) => tasks,
        Err(error) => {
            return ToolExecutionResult::error(format!("database query failed: {error}"));
        }
    };
    let task_names = tasks
        .iter()
        .map(|task| task.name.clone())
        .collect::<Vec<_>>();
    let task = match tasks.iter().find(|task| task.name == name) {
        Some(task) => task,
        None => return ToolExecutionResult::error(missing_message()),
    };
    match operation(Path::new(&task.file_path), &task_names).and_then(|payload| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(payload)
    }) {
        Ok(payload) => ToolExecutionResult::success(payload),
        Err(error) if error.to_string().starts_with("Active task '") => {
            ToolExecutionResult::error(error.to_string())
        }
        Err(error) => ToolExecutionResult::error(format!("task mutation failed: {error}")),
    }
}

fn sync_task_context_after_mutation(
    config: &AppConfig,
    old_name: &str,
    current_name: Option<&str>,
) -> anyhow::Result<()> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let transcript = load_validated_runtime_transcript(
        &mut connection,
        &config.assistant_name,
        config.llm_provider() == LlmProvider::Anthropic,
    )?;
    let old_tool_call_id = context_task_tool_call_id(old_name);
    let mut updated_transcript = transcript
        .into_iter()
        .filter(|message| !message_matches_tool_call_id(message, &old_tool_call_id))
        .collect::<Vec<_>>();

    if let Some(current_name) = current_name
        && let Some(task) = find_active_agenda_item_by_name(&connection, current_name)?
            .filter(|item| item.trigger_datetime.is_none() && item.trigger_context.is_none())
        && !transcript_contains_context_task(&updated_transcript, &task.name)
    {
        updated_transcript.extend(context_task_tool_messages(&task));
    }

    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

fn remove_context_tool_messages_by_id(
    config: &AppConfig,
    tool_call_id: &str,
) -> anyhow::Result<()> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let transcript = load_validated_runtime_transcript(
        &mut connection,
        &config.assistant_name,
        config.llm_provider() == LlmProvider::Anthropic,
    )?;
    let updated_transcript = transcript
        .into_iter()
        .filter(|message| !message_matches_tool_call_id(message, tool_call_id))
        .collect::<Vec<_>>();
    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::{context_task_tool_call_id, task_tools};
    use elroy_config::AppConfig;
    use elroy_db::{
        BootstrapPlan, LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection,
        replace_context_messages,
    };
    use elroy_llm::{ConversationMessage, MessageRole};
    use elroy_tools::ExecutableToolRegistry;

    fn context_dump(database_path: &Path) -> String {
        let mut connection = open_sqlite_connection(database_path).expect("database should open");
        let messages = load_context_messages(&mut connection, LOCAL_USER_TOKEN)
            .expect("context messages should load");
        serde_json::to_string(&messages).expect("context messages should serialize")
    }

    #[test]
    fn task_tools_can_manage_tasks_and_refresh_context() {
        let unique = format!(
            "elroy-rs-task-tools-{}",
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

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path.clone();

        let registry = ExecutableToolRegistry::new(task_tools(config.clone()));
        let created = registry.invoke(
            "create_task",
            "{\"name\":\"Job Search\",\"text\":\"Reach out to three contacts\"}",
        );
        assert!(!created.is_error);
        assert_eq!(created.content, "Task 'job search' has been created.");

        let duplicate = registry.invoke(
            "create_task",
            "{\"name\":\"job search\",\"text\":\"Reach out to one contact\"}",
        );
        assert!(duplicate.is_error);
        assert_eq!(duplicate.content, "Task 'job search' already exists");

        let updated = registry.invoke(
            "update_task_text",
            "{\"name\":\"job search\",\"text\":\"Reach out to four contacts\"}",
        );
        assert!(!updated.is_error);
        assert_eq!(updated.content, "Task 'job search' text has been updated.");
        let missing_updated = registry.invoke(
            "update_task_text",
            "{\"name\":\"missing\",\"text\":\"No-op\"}",
        );
        assert!(missing_updated.is_error);
        assert_eq!(missing_updated.content, "Active task 'missing' not found.");

        let renamed = registry.invoke(
            "rename_task",
            "{\"old_name\":\"job search\",\"new_name\":\"Career Search\"}",
        );
        assert!(!renamed.is_error);
        assert_eq!(
            renamed.content,
            "Task 'job search' has been renamed to 'Career Search'."
        );
        assert!(agenda_dir.join("career_search.md").exists());
        let missing_renamed = registry.invoke(
            "rename_task",
            "{\"old_name\":\"missing\",\"new_name\":\"Backup Search\"}",
        );
        assert!(missing_renamed.is_error);
        assert_eq!(missing_renamed.content, "Active task 'missing' not found.");
        let duplicate_renamed = registry.invoke(
            "rename_task",
            "{\"old_name\":\"career search\",\"new_name\":\"career search\"}",
        );
        assert!(duplicate_renamed.is_error);
        assert_eq!(
            duplicate_renamed.content,
            "Active task 'career search' already exists."
        );

        let listed = registry.invoke("list_tasks", "{\"limit\":10}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("career search"));

        let shown = registry.invoke("show_task", "{\"name\":\"career search\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("Reach out to four contacts"));
        let missing_shown = registry.invoke("show_task", "{\"name\":\"missing\"}");
        assert!(missing_shown.is_error);
        assert_eq!(missing_shown.content, "Active task 'missing' not found.");

        let completed = registry.invoke(
            "complete_task",
            "{\"name\":\"career search\",\"closing_comment\":\"done\"}",
        );
        assert!(!completed.is_error);
        assert_eq!(
            completed.content,
            "Task 'career search' has been marked as completed. Comment: done"
        );
        let missing_completed = registry.invoke("complete_task", "{\"name\":\"missing\"}");
        assert!(missing_completed.is_error);
        assert_eq!(
            missing_completed.content,
            "Active task 'missing' not found."
        );
        let completed_recreated = registry.invoke(
            "create_task",
            "{\"name\":\"Career Search\",\"text\":\"Follow up with recruiters\"}",
        );
        assert!(!completed_recreated.is_error);
        assert_eq!(
            completed_recreated.content,
            "Task 'career search' has been created."
        );
        let completed_recreated_shown =
            registry.invoke("show_task", "{\"name\":\"career search\"}");
        assert!(!completed_recreated_shown.is_error);
        assert!(
            completed_recreated_shown
                .content
                .contains("Follow up with recruiters")
        );

        let plain_created = registry.invoke(
            "create_task",
            "{\"name\":\"Inbox Zero\",\"text\":\"Clear email backlog\"}",
        );
        assert!(!plain_created.is_error);
        let task_context = context_dump(&database_path);
        assert!(task_context.contains(&context_task_tool_call_id("inbox zero")));
        let plain_updated = registry.invoke(
            "update_task_text",
            "{\"name\":\"inbox zero\",\"text\":\"Clear email backlog tonight\"}",
        );
        assert!(!plain_updated.is_error);
        let updated_task_context = context_dump(&database_path);
        assert!(updated_task_context.contains(&context_task_tool_call_id("inbox zero")));
        assert!(updated_task_context.contains("Clear email backlog tonight"));
        let plain_renamed = registry.invoke(
            "rename_task",
            "{\"old_name\":\"inbox zero\",\"new_name\":\"Inbox Clean\"}",
        );
        assert!(!plain_renamed.is_error);
        let renamed_task_context = context_dump(&database_path);
        assert!(!renamed_task_context.contains(&context_task_tool_call_id("inbox zero")));
        assert!(renamed_task_context.contains(&context_task_tool_call_id("inbox clean")));
        let plain_completed = registry.invoke(
            "complete_task",
            "{\"name\":\"inbox clean\",\"closing_comment\":\"done\"}",
        );
        assert!(!plain_completed.is_error);
        let completed_task_context = context_dump(&database_path);
        assert!(!completed_task_context.contains(&context_task_tool_call_id("inbox clean")));

        let deleted_created = registry.invoke(
            "create_task",
            "{\"name\":\"Desk Reset\",\"text\":\"Tidy the desk\"}",
        );
        assert!(!deleted_created.is_error);
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let mut transcript =
            load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("context should load");
        transcript.insert(
            0,
            ConversationMessage::new(MessageRole::User, "keep context"),
        );
        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
            .expect("task context should persist");

        let deleted = registry.invoke(
            "delete_task",
            "{\"name\":\"desk reset\",\"closing_comment\":\"superseded\"}",
        );
        assert!(!deleted.is_error);
        assert_eq!(
            deleted.content,
            "Task 'desk reset' has been deleted. Comment: superseded"
        );

        let deleted_text =
            fs::read_to_string(agenda_dir.join("desk_reset.md")).expect("task file should read");
        assert!(deleted_text.contains("status: deleted"));
        assert!(deleted_text.contains("closing_comment: superseded"));
        let stripped_context = context_dump(&database_path);
        assert!(!stripped_context.contains(&context_task_tool_call_id("desk reset")));
        assert!(stripped_context.contains("keep context"));
        let recreated = registry.invoke(
            "create_task",
            "{\"name\":\"Desk Reset\",\"text\":\"Tidy the desk again\"}",
        );
        assert!(!recreated.is_error);
        assert_eq!(recreated.content, "Task 'desk reset' has been created.");
        let recreated_shown = registry.invoke("show_task", "{\"name\":\"desk reset\"}");
        assert!(!recreated_shown.is_error);
        assert!(recreated_shown.content.contains("Tidy the desk again"));
        let recreated_context = context_dump(&database_path);
        assert!(recreated_context.contains(&context_task_tool_call_id("desk reset")));
        let recreated_connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let deleted_rows: i64 = recreated_connection
            .query_row(
                "SELECT COUNT(*) FROM agenda_items WHERE name = ?1 AND status = 'deleted' AND is_active IS NULL",
                rusqlite::params!["desk reset"],
                |row| row.get(0),
            )
            .expect("deleted task rows should query");
        let active_rows: i64 = recreated_connection
            .query_row(
                "SELECT COUNT(*) FROM agenda_items WHERE name = ?1 AND status = 'created' AND is_active = 1",
                rusqlite::params!["desk reset"],
                |row| row.get(0),
            )
            .expect("active task rows should query");
        assert_eq!(deleted_rows, 1);
        assert_eq!(active_rows, 1);
        let missing_deleted = registry.invoke("delete_task", "{\"name\":\"missing\"}");
        assert!(missing_deleted.is_error);
        assert_eq!(missing_deleted.content, "Active task 'missing' not found.");

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn list_due_tasks_excludes_future_and_context_only_tasks() {
        let unique = format!(
            "elroy-rs-task-due-filtering-{}",
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

        fs::write(
            agenda_dir.join("past_due.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\n---\n\nPast due task\n",
        )
        .expect("past due task should be written");
        fs::write(
            agenda_dir.join("future_due.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2099-01-01T09:00:00\n---\n\nFuture due task\n",
        )
        .expect("future due task should be written");
        fs::write(
            agenda_dir.join("context_only.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after breakfast\n---\n\nContext-only task\n",
        )
        .expect("context-only task should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(task_tools(config));
        let due = registry.invoke("list_due_tasks", "{}");
        assert!(!due.is_error);
        assert!(due.content.contains("past due"));
        assert!(!due.content.contains("future due"));
        assert!(!due.content.contains("context-only"));

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
