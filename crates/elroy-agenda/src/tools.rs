use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use elroy_config::{AppConfig, embedding_provider_config_from_app_config};
use elroy_context::load_validated_runtime_transcript;
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, LOCAL_USER_TOKEN, find_active_agenda_item_by_name,
    list_active_agenda_items, list_active_due_items, list_active_plain_agenda_items,
    list_inactive_due_items, open_sqlite_connection, record_deleted_due_item_tombstone,
    replace_context_messages, run_migrations, upsert_memory_embedding,
};
use elroy_recall::{
    best_effort_embedding_client, context_task_tool_messages, format_due_item_detail,
    parse_sidebar_trigger_datetime, sync_due_item_context_after_mutation,
    sync_task_context_after_mutation, transcript_contains_context_task,
};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec, argument_limit};

use super::{
    add_checklist_item, append_agenda_update, create_agenda_file, mark_agenda_item_completed,
    rename_agenda_file, update_agenda_body, update_checklist_item,
};

pub fn agenda_tools(config: &AppConfig) -> Vec<ExecutableTool> {
    let config_for_agenda_write = config.clone();
    let add_agenda_item = ExecutableTool::new(
        ToolSpec::new(
            "add_agenda_item",
            "Create a new file-backed agenda item and rebuild derived state.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("date", json!({"type": "string"})),
                    ("trigger_datetime", json!({"type": "string"})),
                    ("trigger_context", json!({"type": "string"})),
                ],
                ["text"],
            ),
        ),
        move |arguments| {
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item requires string text");
            };
            let derived_name = derive_agenda_item_name(text);
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
                .unwrap_or(&derived_name);
            let date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("date").and_then(Value::as_str));
            let effective_date = match parse_agenda_item_date(date) {
                Ok(date) => date,
                Err(error) => return ToolExecutionResult::error(error),
            };
            let trigger_datetime = arguments
                .get("trigger_time")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("trigger_datetime").and_then(Value::as_str));
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);

            match (|| -> Result<String, std::io::Error> {
                let mut connection = open_sqlite_connection(&config_for_agenda_write.database_path)
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
                let path = create_agenda_file(
                    &config_for_agenda_write.agenda_dir,
                    name,
                    text,
                    Some(&effective_date),
                    trigger_datetime,
                    trigger_context,
                )?;
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_agenda_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let logical_name = path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_string();
                connection
                    .execute(
                        "UPDATE agenda_items
                         SET name = ?1
                         WHERE file_path = ?2
                           AND is_active = 1",
                        rusqlite::params![
                            logical_name.replace('_', " "),
                            path.to_string_lossy().as_ref()
                        ],
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if let Some(item) =
                    find_active_agenda_item_by_name(&connection, &logical_name.replace('_', " "))
                        .map_err(|error| std::io::Error::other(error.to_string()))?
                {
                    persist_agenda_item_embedding_if_possible(
                        &connection,
                        &config_for_agenda_write,
                        &item,
                    );
                }
                if let Some(task) =
                    find_active_agenda_item_by_name(&connection, &logical_name.replace('_', " "))
                        .map_err(|error| std::io::Error::other(error.to_string()))?
                        .filter(|item| {
                            item.trigger_datetime.is_none() && item.trigger_context.is_none()
                        })
                {
                    let mut transcript = load_validated_runtime_transcript(
                        &mut connection,
                        &config_for_agenda_write.assistant_name,
                        config_for_agenda_write.llm_provider()
                            == elroy_config::LlmProvider::Anthropic,
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                    if !transcript_contains_context_task(&transcript, &task.name) {
                        transcript.extend(context_task_tool_messages(&task));
                        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
                            .map_err(|error| std::io::Error::other(error.to_string()))?;
                    }
                }
                Ok(logical_name)
            })() {
                Ok(logical_name) => ToolExecutionResult::success(format!(
                    "Agenda item added for {effective_date}: {logical_name}"
                )),
                Err(error) if error.to_string().starts_with("Task '") => {
                    ToolExecutionResult::error(error.to_string())
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create agenda item: {error}"))
                }
            }
        },
    );

    let config_for_due_item_write = config.clone();
    let create_due_item = ExecutableTool::new(
        ToolSpec::new(
            "create_due_item",
            "Create a new file-backed due item and rebuild derived state.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("trigger_datetime", json!({"type": "string"})),
                    ("trigger_context", json!({"type": "string"})),
                ],
                ["text"],
            ),
        ),
        move |arguments| {
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_due_item requires string text");
            };
            let trigger_datetime = arguments
                .get("trigger_time")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("trigger_datetime").and_then(Value::as_str));
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);

            if trigger_datetime.is_none() && trigger_context.is_none() {
                return ToolExecutionResult::error(
                    "Either trigger_time or trigger_context must be provided for due items",
                );
            }

            if let Some(trigger_time) = trigger_datetime {
                match parse_trigger_datetime_for_validation(trigger_time) {
                    Ok(parsed) if parsed < Utc::now() => {
                        return ToolExecutionResult::error(format!(
                            "Attempted to create a due item for {trigger_time}, which is in the past"
                        ));
                    }
                    Err(error) => return ToolExecutionResult::error(error),
                    _ => {}
                }
            }

            let derived_name = derive_agenda_item_name(text);
            let name = match arguments.get("name").and_then(Value::as_str) {
                Some(name) if name.trim().is_empty() => {
                    return ToolExecutionResult::error("Due item name cannot be empty");
                }
                Some(name) => name,
                None => &derived_name,
            };

            let created = (|| -> Result<PathBuf, std::io::Error> {
                let mut connection =
                    open_sqlite_connection(&config_for_due_item_write.database_path)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                run_migrations(&mut connection)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if find_active_agenda_item_by_name(&connection, name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                    .is_some()
                {
                    return Err(std::io::Error::other(format!(
                        "{} due item '{name}' already exists",
                        due_item_kind_label(trigger_datetime, trigger_context)
                    )));
                }
                let canonical_path = config_for_due_item_write
                    .agenda_dir
                    .join(format!("{}.md", super::sanitize_filename(name)));
                if canonical_path.exists() {
                    std::fs::remove_file(&canonical_path)?;
                }
                let path = create_agenda_file(
                    &config_for_due_item_write.agenda_dir,
                    name,
                    text,
                    None,
                    trigger_datetime,
                    trigger_context,
                )?;
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(
                    &config_for_due_item_write,
                ))
                .map_err(|error| std::io::Error::other(error.to_string()))?;
                if let Some(item) = find_active_agenda_item_by_name(&connection, name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                {
                    persist_agenda_item_embedding_if_possible(
                        &connection,
                        &config_for_due_item_write,
                        &item,
                    );
                }
                Ok(path)
            })();
            match created {
                Ok(_path) => {
                    if let Err(error) = sync_due_item_context_after_mutation(
                        &config_for_due_item_write,
                        name,
                        Some(name),
                    ) {
                        return ToolExecutionResult::error(format!(
                            "failed to refresh due item context: {error}"
                        ));
                    }
                    let message = match (trigger_datetime.is_some(), trigger_context.is_some()) {
                        (true, true) => format!(
                            "Timed/context due item '{name}' has been created for {}.",
                            trigger_datetime.unwrap_or_default()
                        ),
                        (true, false) => format!(
                            "Timed due item '{name}' has been created for {}.",
                            trigger_datetime.unwrap_or_default()
                        ),
                        (false, true) => {
                            format!("Contextual due item '{name}' has been created.")
                        }
                        (false, false) => format!("New due item created: {name}"),
                    };
                    ToolExecutionResult::success(message)
                }
                Err(error)
                    if error.to_string().starts_with("Due item '")
                        || error.to_string().contains(" due item '") =>
                {
                    ToolExecutionResult::error(error.to_string())
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create due item: {error}"))
                }
            }
        },
    );

    let config_for_due_item_text = config.clone();
    let update_due_item_text = ExecutableTool::new(
        ToolSpec::new(
            "update_due_item_text",
            "Replace the body text of one active due item.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("new_text", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_due_item_text requires a string name");
            };
            let Some(text) = arguments
                .get("new_text")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("text").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("update_due_item_text requires string text");
            };
            let result = mutate_due_item_file_from_config_with_result(
                &config_for_due_item_text,
                name,
                |due_item_names| {
                    let mut sorted_names = due_item_names.to_vec();
                    sorted_names.sort();
                    format!(
                        "Due item '{name}' not found. Valid items: {}",
                        sorted_names.join(", ")
                    )
                },
                |path, _| {
                    update_agenda_body(path, text)?;
                    Ok(format!("Due item '{name}' text has been updated."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_due_item_context_after_mutation(&config_for_due_item_text, name, Some(name))
            {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh due item context: {error}"
                )),
            }
        },
    );

    let config_for_due_item_rename = config.clone();
    let rename_due_item = ExecutableTool::new(
        ToolSpec::new(
            "rename_due_item",
            "Rename one active due item.",
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
                return ToolExecutionResult::error("rename_due_item requires a string name");
            };
            let Some(new_name) = arguments.get("new_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("rename_due_item requires string new_name");
            };
            let result = mutate_due_item_file_from_config_with_result(
                &config_for_due_item_rename,
                name,
                |due_item_names| {
                    format!(
                        "Active due item '{name}' not found. Active items: {}",
                        due_item_names.join(", ")
                    )
                },
                |path, due_item_names| {
                    if due_item_names.iter().any(|existing| existing == new_name) {
                        return Err(std::io::Error::other(format!(
                            "Active due item '{new_name}' already exists."
                        )));
                    }
                    let _renamed = rename_agenda_file(path, new_name)?;
                    Ok(format!(
                        "Due item '{name}' has been renamed to '{new_name}'."
                    ))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_due_item_context_after_mutation(
                &config_for_due_item_rename,
                name,
                Some(new_name),
            ) {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh due item context: {error}"
                )),
            }
        },
    );

    let config_for_due_item_complete = config.clone();
    let complete_due_item = ExecutableTool::new(
        ToolSpec::new(
            "complete_due_item",
            "Mark one active due item as completed.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("comment", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("complete_due_item requires a string name");
            };
            let result = mutate_due_item_file_from_config_with_result(
                &config_for_due_item_complete,
                name,
                |due_item_names| {
                    format!(
                        "Active due item '{name}' not found. Active due items: {}",
                        due_item_names.join(", ")
                    )
                },
                |path, _| {
                    let comment = arguments
                        .get("closing_comment")
                        .and_then(Value::as_str)
                        .or_else(|| arguments.get("comment").and_then(Value::as_str));
                    mark_agenda_item_completed(path, comment)?;
                    Ok(match comment {
                        Some(comment) => {
                            format!(
                                "Due item '{name}' has been marked as completed. Comment: {comment}"
                            )
                        }
                        None => format!("Due item '{name}' has been marked as completed."),
                    })
                },
            );
            if result.is_error {
                return result;
            }
            match sync_due_item_context_after_mutation(&config_for_due_item_complete, name, None) {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh due item context: {error}"
                )),
            }
        },
    );

    let config_for_due_item_delete = config.clone();
    let delete_due_item = ExecutableTool::new(
        ToolSpec::new(
            "delete_due_item",
            "Permanently delete one active due item.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("delete_due_item requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            let due_item_before_delete = (|| -> Result<Option<AgendaItemRecord>, String> {
                let mut connection =
                    open_sqlite_connection(&config_for_due_item_delete.database_path)
                        .map_err(|error| error.to_string())?;
                run_migrations(&mut connection).map_err(|error| error.to_string())?;
                let item = find_active_agenda_item_by_name(&connection, name)
                    .map_err(|error| error.to_string())?
                    .filter(|item| {
                        item.trigger_datetime.is_some() || item.trigger_context.is_some()
                    });
                Ok(item)
            })();
            let result = mutate_due_item_file_from_config_with_result(
                &config_for_due_item_delete,
                name,
                |due_item_names| {
                    format!(
                        "Active due item '{name}' not found. Active due items: {}",
                        due_item_names.join(", ")
                    )
                },
                |path, _| {
                    std::fs::remove_file(path)?;
                    Ok(match closing_comment {
                        Some(comment) => {
                            format!("Due item '{name}' has been deleted. Comment: {comment}")
                        }
                        None => format!("Due item '{name}' has been deleted."),
                    })
                },
            );
            if result.is_error {
                return result;
            }
            if let Ok(Some(item)) = due_item_before_delete {
                let mut connection =
                    match open_sqlite_connection(&config_for_due_item_delete.database_path) {
                        Ok(connection) => connection,
                        Err(error) => {
                            return ToolExecutionResult::error(format!(
                                "failed to open database: {error}"
                            ));
                        }
                    };
                if let Err(error) = run_migrations(&mut connection) {
                    return ToolExecutionResult::error(format!(
                        "failed to run migrations: {error}"
                    ));
                }
                if let Err(error) =
                    record_deleted_due_item_tombstone(&connection, &item, closing_comment)
                {
                    return ToolExecutionResult::error(format!(
                        "failed to record deleted due item tombstone: {error}"
                    ));
                }
            }
            match sync_due_item_context_after_mutation(&config_for_due_item_delete, name, None) {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh due item context: {error}"
                )),
            }
        },
    );

    let config_for_agenda_update = config.clone();
    let add_agenda_item_update = ExecutableTool::new(
        ToolSpec::new(
            "add_agenda_item_update",
            "Append a timestamped update note to one active agenda item.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("note", json!({"type": "string"})),
                ],
                ["name", "note"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("add_agenda_item_update requires a string name");
            };
            let Some(note) = arguments.get("note").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item_update requires a string note");
            };
            let result = mutate_agenda_file_from_config_with_result(
                &config_for_agenda_update,
                name,
                |path| {
                    let timestamp = append_agenda_update(path, note)?;
                    Ok(format!("Update added to '{name}' at {timestamp}."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_agenda_update, name, Some(name)) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_agenda_complete = config.clone();
    let complete_agenda_item = ExecutableTool::new(
        ToolSpec::new(
            "complete_agenda_item",
            "Mark one active agenda item as completed.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("closing_comment", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("complete_agenda_item requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            let result = mutate_agenda_file_from_config_with_result(
                &config_for_agenda_complete,
                name,
                |path| {
                    mark_agenda_item_completed(path, closing_comment)?;
                    Ok(format!("Agenda item '{name}' marked as completed."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_agenda_complete, name, None) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_agenda_delete = config.clone();
    let delete_agenda_item = ExecutableTool::new(
        ToolSpec::new(
            "delete_agenda_item",
            "Mark one active agenda item as deleted.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("closing_comment", json!({"type": "string"})),
                ],
                ["name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("delete_agenda_item requires a string name");
            };
            let result = mutate_agenda_file_from_config_with_result(
                &config_for_agenda_delete,
                name,
                |path| {
                    std::fs::remove_file(path)?;
                    Ok(format!("Agenda item '{name}' deleted."))
                },
            );
            if result.is_error {
                return result;
            }
            match sync_task_context_after_mutation(&config_for_agenda_delete, name, None) {
                Ok(()) => result,
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to refresh task context: {error}"))
                }
            }
        },
    );

    let config_for_checklist_add = config.clone();
    let add_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "add_agenda_checklist_item",
            "Add a new item to an agenda item's checklist.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("due_date", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error(
                    "add_agenda_checklist_item requires a string name",
                );
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "add_agenda_checklist_item requires string text",
                );
            };
            let due_date =
                match parse_agenda_due_date(arguments.get("due_date").and_then(Value::as_str)) {
                    Ok(date) => date,
                    Err(error) => return ToolExecutionResult::error(error),
                };
            mutate_agenda_file_from_config_with_result(&config_for_checklist_add, name, |path| {
                let item_id = add_checklist_item(path, text, due_date.as_deref())?;
                Ok(format!("Checklist item {item_id} added to '{name}'."))
            })
        },
    );

    let config_for_checklist_edit = config.clone();
    let edit_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "edit_agenda_checklist_item",
            "Edit the text of an existing agenda checklist item.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("item_id", json!({"type": "integer"})),
                    ("text", json!({"type": "string"})),
                ],
                ["name", "item_id", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires a string name",
                );
            };
            let item_id = match parse_optional_line_number_argument(&arguments, "checklist_item_id")
            {
                Ok(Some(id)) => id,
                Ok(None) => match parse_optional_line_number_argument(&arguments, "item_id") {
                    Ok(Some(id)) => id,
                    Ok(None) => {
                        return ToolExecutionResult::error(
                            "edit_agenda_checklist_item requires item_id",
                        );
                    }
                    Err(error) => return ToolExecutionResult::error(error),
                },
                Err(error) => return ToolExecutionResult::error(error),
            };
            let Some(text) = arguments
                .get("new_text")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("text").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires string text",
                );
            };
            mutate_agenda_file_from_config_with_result(&config_for_checklist_edit, name, |path| {
                match update_checklist_item(path, item_id, Some(text), None) {
                    Ok(_) => Ok(format!("Checklist item {item_id} on '{name}' updated.")),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        Err(std::io::Error::other(format!(
                            "Agenda item '{name}' has no checklist item {item_id}."
                        )))
                    }
                    Err(error) => Err(error),
                }
            })
        },
    );

    let config_for_checklist_complete = config.clone();
    let complete_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "complete_agenda_checklist_item",
            "Mark an agenda checklist item as completed.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("item_id", json!({"type": "integer"})),
                ],
                ["name", "item_id"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("item_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error(
                    "complete_agenda_checklist_item requires a string name",
                );
            };
            let item_id = match parse_optional_line_number_argument(&arguments, "checklist_item_id")
            {
                Ok(Some(id)) => id,
                Ok(None) => match parse_optional_line_number_argument(&arguments, "item_id") {
                    Ok(Some(id)) => id,
                    Ok(None) => {
                        return ToolExecutionResult::error(
                            "complete_agenda_checklist_item requires item_id",
                        );
                    }
                    Err(error) => return ToolExecutionResult::error(error),
                },
                Err(error) => return ToolExecutionResult::error(error),
            };
            mutate_agenda_file_from_config_with_result(
                &config_for_checklist_complete,
                name,
                |path| match update_checklist_item(path, item_id, None, Some(true)) {
                    Ok(_) => Ok(format!(
                        "Checklist item {item_id} on '{name}' marked as completed."
                    )),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        Err(std::io::Error::other(format!(
                            "Agenda item '{name}' has no checklist item {item_id}."
                        )))
                    }
                    Err(error) => Err(error),
                },
            )
        },
    );

    let database_path = config.database_path.clone();
    let list_agenda = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda",
            "List active agenda items and reminders.",
            JsonSchema::object([("date", json!({"type": "string"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("date").and_then(Value::as_str));
            let target_date = match parse_agenda_item_date(date) {
                Ok(date) => date,
                Err(error) => return ToolExecutionResult::error(error),
            };

            match open_sqlite_connection(&database_path) {
                Ok(connection) => {
                    let items = match list_active_agenda_items(&connection, 1_000) {
                        Ok(items) => items
                            .into_iter()
                            .filter(|item| item.agenda_date.as_deref() == Some(&target_date))
                            .collect::<Vec<_>>(),
                        Err(error) => {
                            return ToolExecutionResult::error(format!(
                                "database query failed: {error}"
                            ));
                        }
                    };

                    if items.is_empty() {
                        return ToolExecutionResult::success(format!(
                            "No agenda items found for {target_date}."
                        ));
                    }

                    let mut lines = vec![format!("Agenda for {target_date}:")];
                    for item in items {
                        let checklist_info = if item.checklist_total > 0 {
                            format!(
                                " [{}/{} checklist items done]",
                                item.checklist_completed, item.checklist_total
                            )
                        } else {
                            String::new()
                        };
                        lines.push(format!("- {}{}", item.name, checklist_info));
                    }
                    ToolExecutionResult::success(lines.join("\n"))
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_agenda_items = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda_items",
            "List active agenda items for a given date.",
            JsonSchema::object([("date", json!({"type": "string"}))], ["date"]),
        ),
        move |arguments| {
            let date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("date").and_then(Value::as_str));
            let target_date = match parse_agenda_item_date(date) {
                Ok(date) => date,
                Err(error) => return ToolExecutionResult::error(error),
            };

            match open_sqlite_connection(&database_path) {
                Ok(connection) => {
                    let items = match list_active_agenda_items(&connection, 1_000) {
                        Ok(items) => items
                            .into_iter()
                            .filter(|item| item.agenda_date.as_deref() == Some(&target_date))
                            .collect::<Vec<_>>(),
                        Err(error) => {
                            return ToolExecutionResult::error(format!(
                                "database query failed: {error}"
                            ));
                        }
                    };

                    ToolExecutionResult::success(
                        json!({
                            "item_date": target_date,
                            "items": items.into_iter().map(|item| {
                                json!({
                                    "name": item.name,
                                    "text": item.body,
                                })
                            }).collect::<Vec<_>>(),
                        })
                        .to_string(),
                    )
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_agenda_items_cmd = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda_items_cmd",
            "List active agenda items in detail format.",
            JsonSchema::object([("date", json!({"type": "string"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("date").and_then(Value::as_str));
            let target_date = match parse_agenda_item_date(date) {
                Ok(date) => date,
                Err(error) => return ToolExecutionResult::error(error),
            };
            match open_sqlite_connection(&database_path) {
                Ok(connection) => match list_active_plain_agenda_items(&connection, 1_000) {
                    Ok(items) => {
                        let items = items
                            .into_iter()
                            .filter(|item| item.agenda_date.as_deref() == Some(&target_date))
                            .collect::<Vec<_>>();
                        if items.is_empty() {
                            return ToolExecutionResult::success(format!(
                                "No agenda items for {target_date}."
                            ));
                        }
                        let mut lines = vec![format!("Agenda for {target_date}:")];
                        for item in items {
                            let checklist_info = if item.checklist_total > 0 {
                                format!(
                                    " [{}/{} checklist items done]",
                                    item.checklist_completed, item.checklist_total
                                )
                            } else {
                                String::new()
                            };
                            lines.push(format!(
                                "- {}: {}{}",
                                item.name,
                                elroy_core::excerpt(&item.body, 180),
                                checklist_info
                            ));
                        }
                        ToolExecutionResult::success(lines.join("\n"))
                    }
                    Err(error) => {
                        ToolExecutionResult::error(format!("database query failed: {error}"))
                    }
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_due_items = ExecutableTool::new(
        ToolSpec::new(
            "list_due_items",
            "List active due items.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            match open_sqlite_connection(&database_path) {
                Ok(connection) => match list_active_due_items(&connection, limit) {
                    Ok(items) => {
                        let mut lines = vec!["Due Items".to_string()];
                        for item in items {
                            lines.push(format!(
                                "- DueItem | {} | {}",
                                item.name,
                                elroy_core::excerpt(&item.body, 180)
                            ));
                        }
                        ToolExecutionResult::success(lines.join("\n"))
                    }
                    Err(error) => {
                        ToolExecutionResult::error(format!("database query failed: {error}"))
                    }
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let print_active_due_items = ExecutableTool::new(
        ToolSpec::new(
            "print_active_due_items",
            "Print active due items in a summary report.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            match open_sqlite_connection(&database_path) {
                Ok(connection) => match list_active_due_items(&connection, limit) {
                    Ok(items) => {
                        ToolExecutionResult::success(format_due_item_listing(&items, true))
                    }
                    Err(error) => {
                        ToolExecutionResult::error(format!("database query failed: {error}"))
                    }
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_inactive_due_items_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_inactive_due_items",
            "List inactive (completed or deleted) due items.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            match open_sqlite_connection(&database_path) {
                Ok(connection) => match list_inactive_due_items(&connection, limit) {
                    Ok(items) => {
                        let mut lines = vec!["Inactive Due Items".to_string()];
                        for item in items {
                            lines.push(format!(
                                "- DueItem | {} | {}{}",
                                item.name,
                                elroy_core::excerpt(&item.body, 180),
                                item.closing_comment
                                    .as_deref()
                                    .map(|comment| format!(" | {comment}"))
                                    .unwrap_or_default()
                            ));
                        }
                        ToolExecutionResult::success(lines.join("\n"))
                    }
                    Err(error) => {
                        ToolExecutionResult::error(format!("database query failed: {error}"))
                    }
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let print_inactive_due_items = ExecutableTool::new(
        ToolSpec::new(
            "print_inactive_due_items",
            "Print inactive due items in a summary report.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            match open_sqlite_connection(&database_path) {
                Ok(connection) => match list_inactive_due_items(&connection, limit) {
                    Ok(items) => {
                        ToolExecutionResult::success(format_due_item_listing(&items, false))
                    }
                    Err(error) => {
                        ToolExecutionResult::error(format!("database query failed: {error}"))
                    }
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let show_due_item = ExecutableTool::new(
        ToolSpec::new(
            "show_due_item",
            "Show detailed content for one active due item.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("show_due_item requires a string name");
            };
            match open_sqlite_connection(&database_path) {
                Ok(connection) => {
                    let item = find_active_agenda_item_by_name(&connection, name);
                    match item {
                        Ok(Some(item))
                            if item.trigger_datetime.is_some()
                                || item.trigger_context.is_some() =>
                        {
                            ToolExecutionResult::success(
                                json!({
                                    "name": item.name,
                                    "file_path": item.file_path,
                                    "trigger_datetime": item.trigger_datetime,
                                    "trigger_context": item.trigger_context,
                                    "status": item.status,
                                    "body": item.body,
                                })
                                .to_string(),
                            )
                        }
                        Ok(_) => ToolExecutionResult::error(due_item_not_found_message(
                            &connection,
                            name,
                        )),
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let print_due_item = ExecutableTool::new(
        ToolSpec::new(
            "print_due_item",
            "Print detailed content for one active due item.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("print_due_item requires a string name");
            };
            match open_sqlite_connection(&database_path) {
                Ok(connection) => {
                    let item = find_active_agenda_item_by_name(&connection, name);
                    match item {
                        Ok(Some(item))
                            if item.trigger_datetime.is_some()
                                || item.trigger_context.is_some() =>
                        {
                            ToolExecutionResult::success(format_due_item_detail(&item))
                        }
                        Ok(_) => ToolExecutionResult::error(due_item_not_found_message(
                            &connection,
                            name,
                        )),
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let show_agenda_item = ExecutableTool::new(
        ToolSpec::new(
            "show_agenda_item",
            "Show one active agenda item by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("show_agenda_item requires a string name");
            };
            match open_sqlite_connection(&database_path) {
                Ok(connection) => {
                    let item = match find_matching_active_agenda_item(&connection, name) {
                        Ok(item) => item,
                        Err(error) => return ToolExecutionResult::error(error),
                    };
                    ToolExecutionResult::success(
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
                        .to_string(),
                    )
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    vec![
        add_agenda_item,
        create_due_item,
        update_due_item_text,
        rename_due_item,
        complete_due_item,
        delete_due_item,
        add_agenda_item_update,
        complete_agenda_item,
        delete_agenda_item,
        add_agenda_checklist_item,
        edit_agenda_checklist_item,
        complete_agenda_checklist_item,
        list_agenda,
        list_agenda_items,
        list_agenda_items_cmd,
        list_due_items,
        print_active_due_items,
        list_inactive_due_items_tool,
        print_inactive_due_items,
        show_due_item,
        print_due_item,
        show_agenda_item,
    ]
}

fn agenda_item_embedding_text(item: &AgendaItemRecord) -> String {
    let mut text = format!("# {}\n{}", item.name, item.body.trim());
    if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
        text.push_str(&format!("\ntrigger_datetime: {trigger_datetime}"));
    }
    if let Some(trigger_context) = item.trigger_context.as_deref() {
        text.push_str(&format!("\ntrigger_context: {trigger_context}"));
    }
    text
}

fn persist_agenda_item_embedding_if_possible(
    connection: &rusqlite::Connection,
    config: &AppConfig,
    item: &AgendaItemRecord,
) {
    let provider_config = embedding_provider_config_from_app_config(config).ok();
    let Some(embedding_client) = best_effort_embedding_client(provider_config.as_ref()) else {
        return;
    };
    let embedding_text = agenda_item_embedding_text(item);
    let Ok(embedding) = embedding_client.embed(&embedding_text) else {
        return;
    };
    let _ = upsert_memory_embedding(connection, &item.file_path, &embedding, &embedding_text);
}

// ── helper functions (moved from elroy-app) ───────────────────────────────────

fn due_item_kind_label(
    trigger_datetime: Option<&str>,
    trigger_context: Option<&str>,
) -> &'static str {
    match (trigger_datetime.is_some(), trigger_context.is_some()) {
        (true, true) => "Timed/context",
        (true, false) => "Timed",
        (false, true) => "Contextual",
        (false, false) => "Generic",
    }
}

fn due_item_not_found_message(connection: &rusqlite::Connection, name: &str) -> String {
    let mut names = list_active_due_items(connection, 1_000)
        .unwrap_or_default()
        .into_iter()
        .map(|item| item.name)
        .collect::<Vec<_>>();
    names.sort();
    format!(
        "Due item '{name}' not found. Valid items: {}",
        names.join(", ")
    )
}

pub fn mutate_due_item_file_from_config_with_result(
    config: &AppConfig,
    name: &str,
    missing_message: impl FnOnce(&[String]) -> String,
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
    let due_items = match list_active_due_items(&connection, 1_000) {
        Ok(items) => items,
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    let due_item_names = due_items
        .iter()
        .map(|item| item.name.clone())
        .collect::<Vec<_>>();
    let Some(item) = due_items.iter().find(|item| item.name == name) else {
        return ToolExecutionResult::error(missing_message(&due_item_names));
    };
    match operation(Path::new(&item.file_path), &due_item_names).and_then(|payload| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(payload)
    }) {
        Ok(payload) => ToolExecutionResult::success(payload),
        Err(error) if error.to_string().starts_with("Active due item '") => {
            ToolExecutionResult::error(error.to_string())
        }
        Err(error) => ToolExecutionResult::error(format!("due item mutation failed: {error}")),
    }
}

pub fn mutate_agenda_file_from_config_with_result(
    config: &AppConfig,
    name: &str,
    operation: impl FnOnce(&Path) -> std::io::Result<String>,
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
    let item = match find_matching_active_agenda_item(&connection, name) {
        Ok(item) => item,
        Err(error) => return ToolExecutionResult::error(error),
    };
    match operation(Path::new(&item.file_path)).and_then(|payload| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(payload)
    }) {
        Ok(payload) => ToolExecutionResult::success(payload),
        Err(error) if error.to_string().starts_with("Agenda item '") => {
            ToolExecutionResult::error(error.to_string())
        }
        Err(error) => ToolExecutionResult::error(format!("agenda mutation failed: {error}")),
    }
}

pub fn find_matching_active_agenda_item(
    connection: &rusqlite::Connection,
    item_name: &str,
) -> Result<elroy_db::AgendaItemRecord, String> {
    let items = list_active_agenda_items(connection, 1_000)
        .map_err(|error| format!("database query failed: {error}"))?;
    let query = item_name.to_ascii_lowercase();
    let matches = items
        .into_iter()
        .filter(|item| {
            let item_name = item.name.to_ascii_lowercase();
            let stem = Path::new(&item.file_path)
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            item_name.contains(&query) || stem.contains(&query)
        })
        .collect::<Vec<_>>();
    let mut matches = matches;
    matches.sort_by(|left, right| {
        Path::new(&left.file_path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .cmp(
                Path::new(&right.file_path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default(),
            )
    });
    match matches.len() {
        0 => Err(format!("No agenda item found matching '{item_name}'.")),
        1 => Ok(matches.into_iter().next().expect("checked len")),
        _ => Err(format!(
            "Multiple agenda items match '{}': {}. Be more specific.",
            item_name,
            matches
                .iter()
                .map(|item| {
                    Path::new(&item.file_path)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or_default()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

pub fn parse_trigger_datetime_for_validation(raw: &str) -> Result<DateTime<Utc>, String> {
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

pub fn parse_agenda_item_date(raw: Option<&str>) -> Result<String, String> {
    match raw {
        Some(raw) => NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
            .map(|date| date.format("%Y-%m-%d").to_string())
            .map_err(|_| format!("Invalid date format '{raw}'. Use YYYY-MM-DD.")),
        None => Ok(Local::now().date_naive().format("%Y-%m-%d").to_string()),
    }
}

pub fn parse_agenda_due_date(raw: Option<&str>) -> Result<Option<String>, String> {
    match raw {
        Some(raw) => NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
            .map(|date| Some(date.format("%Y-%m-%d").to_string()))
            .map_err(|_| format!("Invalid due_date format '{raw}'. Use YYYY-MM-DD.")),
        None => Ok(None),
    }
}

pub fn parse_optional_line_number_argument(
    arguments: &Value,
    key: &str,
) -> Result<Option<i64>, String> {
    match arguments.get(key) {
        None => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be an integer")),
        Some(Value::String(raw)) => raw
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| format!("{key} must be an integer")),
        Some(_) => Err(format!("{key} must be an integer")),
    }
}

pub fn format_due_item_listing(items: &[AgendaItemRecord], active: bool) -> String {
    if items.is_empty() {
        let status = if active { "active" } else { "inactive" };
        return format!("No {status} due items found.");
    }

    let title = if active {
        "Active Due Items"
    } else {
        "Inactive Due Items"
    };
    let mut lines = vec![title.to_string()];
    for item in items {
        let item_type = if item.trigger_datetime.is_some() {
            "Timed"
        } else {
            "Contextual"
        };
        let trigger_time = item
            .trigger_datetime
            .as_deref()
            .map(|value| {
                parse_sidebar_trigger_datetime(value)
                    .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_else(|| value.to_string())
            })
            .unwrap_or_else(|| "N/A".to_string());
        let context = item.trigger_context.as_deref().unwrap_or("N/A").to_string();
        let mut line = format!(
            "- {} | Type: {} | Trigger Time: {} | Context: {} | Text: {}",
            item.name, item_type, trigger_time, context, item.body
        );
        if !active && let Some(closing_comment) = item.closing_comment.as_deref() {
            line.push_str(&format!(" | Comment: {closing_comment}"));
        }
        lines.push(line);
    }
    lines.join("\n")
}

pub fn derive_agenda_item_name(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or(text)
        .trim()
        .chars()
        .take(40)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::agenda_tools;
    use elroy_config::AppConfig;
    use elroy_db::{
        BootstrapPlan, LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection,
        run_migrations,
    };
    use elroy_llm::ConversationMessage;
    use elroy_recall::{context_due_item_tool_call_id, message_matches_tool_call_id};
    use elroy_tools::{ExecutableToolRegistry, JsonSchema};

    fn persisted_context_messages(config: &AppConfig) -> Vec<ConversationMessage> {
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("context should load")
    }

    #[test]
    fn agenda_tools_can_manage_checklists() {
        let unique = format!(
            "elroy-rs-agenda-checklists-{}",
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
            agenda_dir.join("trip.md"),
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\n---\n\nPack bags\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let added = registry.invoke(
            "add_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"text\":\"passport\",\"due_date\":\"2026-05-14\"}",
        );
        assert!(!added.is_error);
        assert_eq!(added.content, "Checklist item 1 added to 'trip'.");
        let invalid_due_date = registry.invoke(
            "add_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"text\":\"backup passport\",\"due_date\":\"2026/05/14\"}",
        );
        assert!(invalid_due_date.is_error);
        assert_eq!(
            invalid_due_date.content,
            "Invalid due_date format '2026/05/14'. Use YYYY-MM-DD."
        );

        let edited = registry.invoke(
            "edit_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"checklist_item_id\":1,\"new_text\":\"passport + visa\"}",
        );
        assert!(!edited.is_error);
        assert_eq!(edited.content, "Checklist item 1 on 'trip' updated.");
        let missing_edited = registry.invoke(
            "edit_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"checklist_item_id\":99,\"new_text\":\"backup\"}",
        );
        assert!(missing_edited.is_error);
        assert_eq!(
            missing_edited.content,
            "Agenda item 'trip' has no checklist item 99."
        );

        let completed = registry.invoke(
            "complete_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"checklist_item_id\":1}",
        );
        assert!(!completed.is_error);
        assert_eq!(
            completed.content,
            "Checklist item 1 on 'trip' marked as completed."
        );
        let missing_completed = registry.invoke(
            "complete_agenda_checklist_item",
            "{\"item_name\":\"trip\",\"checklist_item_id\":99}",
        );
        assert!(missing_completed.is_error);
        assert_eq!(
            missing_completed.content,
            "Agenda item 'trip' has no checklist item 99."
        );

        let file_text = fs::read_to_string(agenda_dir.join("trip.md")).expect("agenda should read");
        assert!(file_text.contains("passport + visa"));
        assert!(file_text.contains("completed: true"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_tools_can_show_and_list_inactive_due_items() {
        let unique = format!(
            "elroy-rs-agenda-inactive-due-items-{}",
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
            agenda_dir.join("call_mom.md"),
            "---\ndate: unscheduled\ncompleted: true\nstatus: completed\ntrigger_context: after dinner\nclosing_comment: done\n---\n\nCall mom tonight\n",
        )
        .expect("inactive due item should be written");
        fs::write(
            agenda_dir.join("pay_bill.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2026-05-15T09:00:00\n---\n\nPay bill\n",
        )
        .expect("active due item should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let shown = registry.invoke("show_due_item", "{\"name\":\"pay bill\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("Pay bill"));
        assert!(shown.content.contains("2026-05-15T09:00:00"));

        let printed = registry.invoke("print_due_item", "{\"name\":\"pay bill\"}");
        assert!(!printed.is_error);
        assert!(printed.content.contains("Due item 'pay bill':"));
        assert!(
            printed
                .content
                .contains("Trigger Time: 2026-05-15 09:00:00")
        );
        assert!(printed.content.contains("Text: Pay bill"));

        let missing_printed = registry.invoke("print_due_item", "{\"name\":\"missing\"}");
        assert!(missing_printed.is_error);
        assert_eq!(
            missing_printed.content,
            "Due item 'missing' not found. Valid items: pay bill"
        );

        let inactive = registry.invoke("list_inactive_due_items", "{\"limit\":10}");
        assert!(!inactive.is_error);
        assert!(inactive.content.contains("call mom"));
        assert!(inactive.content.contains("done"));

        let printed_active = registry.invoke("print_active_due_items", "{\"n\":10}");
        assert!(!printed_active.is_error);
        assert!(printed_active.content.contains("Active Due Items"));
        assert!(printed_active.content.contains("pay bill"));
        assert!(printed_active.content.contains("Type: Timed"));
        assert!(
            printed_active
                .content
                .contains("Trigger Time: 2026-05-15 09:00:00")
        );

        let printed_inactive = registry.invoke("print_inactive_due_items", "{\"n\":10}");
        assert!(!printed_inactive.is_error);
        assert!(printed_inactive.content.contains("Inactive Due Items"));
        assert!(printed_inactive.content.contains("call mom"));
        assert!(printed_inactive.content.contains("Type: Contextual"));

        let deleted = registry.invoke(
            "delete_due_item",
            "{\"name\":\"pay bill\",\"closing_comment\":\"paid online\"}",
        );
        assert!(!deleted.is_error);
        let inactive_after_delete = registry.invoke("list_inactive_due_items", "{\"limit\":10}");
        assert!(!inactive_after_delete.is_error);
        assert!(inactive_after_delete.content.contains("pay bill"));
        assert!(inactive_after_delete.content.contains("paid online"));

        let printed_inactive_after_delete =
            registry.invoke("print_inactive_due_items", "{\"n\":10}");
        assert!(!printed_inactive_after_delete.is_error);
        assert!(printed_inactive_after_delete.content.contains("pay bill"));
        assert!(
            printed_inactive_after_delete
                .content
                .contains("Comment: paid online")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_tools_can_show_exact_items_and_list_one_date() {
        let unique = format!(
            "elroy-rs-agenda-show-and-list-{}",
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
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\n---\n\nbring forms\n",
        )
        .expect("agenda file should be written");
        fs::write(
            agenda_dir.join("trip.md"),
            "---\ndate: 2026-05-16\ncompleted: false\nstatus: created\n---\n\npack snacks\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let shown = registry.invoke("show_agenda_item", "{\"name\":\"doctor visit\"}");
        let substring_shown = registry.invoke("show_agenda_item", "{\"name\":\"visit\"}");
        let missing_shown = registry.invoke("show_agenda_item", "{\"name\":\"dentist\"}");
        let listed = registry.invoke("list_agenda_items", "{\"item_date\":\"2026-05-15\"}");

        assert!(!shown.is_error);
        assert!(shown.content.contains("bring forms"));
        assert!(!substring_shown.is_error);
        assert!(substring_shown.content.contains("bring forms"));
        assert!(missing_shown.is_error);
        assert_eq!(
            missing_shown.content,
            "No agenda item found matching 'dentist'."
        );

        assert!(!listed.is_error);
        assert!(listed.content.contains("\"item_date\":\"2026-05-15\""));
        assert!(listed.content.contains("\"name\":\"doctor visit\""));
        assert!(listed.content.contains("\"text\":\"bring forms\""));
        assert!(!listed.content.contains("\"name\":\"trip\""));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_tools_can_format_agenda_items_cmd() {
        let unique = format!(
            "elroy-rs-agenda-list-cmd-{}",
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
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\nchecklist:\n  - id: 1\n    text: Bring insurance card\n    completed: true\n  - id: 2\n    text: Bring forms\n    completed: false\n---\n\nbring forms\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let listed = registry.invoke("list_agenda_items_cmd", "{\"item_date\":\"2026-05-15\"}");
        let empty = registry.invoke("list_agenda_items_cmd", "{\"item_date\":\"2026-05-16\"}");

        assert!(!listed.is_error);
        assert!(listed.content.contains("Agenda for 2026-05-15:"));
        assert!(
            listed
                .content
                .contains("- doctor visit: bring forms [1/2 checklist items done]")
        );
        assert!(!empty.is_error);
        assert_eq!(empty.content, "No agenda items for 2026-05-16.");

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_tools_can_update_complete_and_delete_agenda_items() {
        let unique = format!(
            "elroy-rs-agenda-mutations-{}",
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
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\n---\n\nbring forms\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let added = registry.invoke(
            "add_agenda_item",
            "{\"name\":\"Project Kickoff\",\"text\":\"Prepare slides\",\"date\":\"2026-05-18\"}",
        );
        assert!(!added.is_error);
        assert_eq!(
            added.content,
            "Agenda item added for 2026-05-18: project_kickoff"
        );
        assert!(agenda_dir.join("project_kickoff.md").exists());

        let duplicate_added = registry.invoke(
            "add_agenda_item",
            "{\"name\":\"Project Kickoff\",\"text\":\"Prepare other slides\",\"date\":\"2026-05-18\"}",
        );
        assert!(duplicate_added.is_error);
        assert_eq!(
            duplicate_added.content,
            "Task 'Project Kickoff' already exists"
        );

        let added_context = persisted_context_messages(&config);
        assert!(added_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-task:project kickoff")
        }));
        assert!(added_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-task:project kickoff")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("Prepare slides"))
        }));

        let update = registry.invoke(
            "add_agenda_item_update",
            "{\"item_name\":\"project kickoff\",\"note\":\"called ahead\"}",
        );
        assert!(!update.is_error);
        assert!(
            update
                .content
                .starts_with("Update added to 'project kickoff' at unix-")
        );
        let updated_text =
            fs::read_to_string(agenda_dir.join("project_kickoff.md")).expect("agenda should read");
        assert!(updated_text.contains("## Updates"));
        let update_timestamp = update
            .content
            .trim_start_matches("Update added to 'project kickoff' at ")
            .trim_end_matches('.');
        assert!(updated_text.contains(&format!("**{update_timestamp}**")));
        assert!(updated_text.contains("called ahead"));

        let updated_context = persisted_context_messages(&config);
        assert!(updated_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-task:project kickoff")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("called ahead"))
        }));

        let complete = registry.invoke(
            "complete_agenda_item",
            "{\"item_name\":\"project kickoff\",\"closing_comment\":\"done\"}",
        );
        assert!(!complete.is_error);
        assert_eq!(
            complete.content,
            "Agenda item 'project kickoff' marked as completed."
        );
        let completed_text =
            fs::read_to_string(agenda_dir.join("project_kickoff.md")).expect("agenda should read");
        assert!(completed_text.contains("completed: true"));
        assert!(completed_text.contains("status: completed"));
        let completed_context = persisted_context_messages(&config);
        assert!(!completed_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-task:project kickoff")
        }));

        fs::write(
            agenda_dir.join("call_mom.md"),
            "---\ndate: 2026-05-16\ncompleted: false\nstatus: created\n---\n\ncall mom\n",
        )
        .expect("second agenda file should be written");
        fs::write(
            agenda_dir.join("call_dad.md"),
            "---\ndate: 2026-05-16\ncompleted: false\nstatus: created\n---\n\ncall dad\n",
        )
        .expect("third agenda file should be written");
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let delete_created = registry.invoke(
            "add_agenda_item",
            "{\"name\":\"Desk Notes\",\"text\":\"Tidy desk\",\"date\":\"2026-05-16\"}",
        );
        assert!(!delete_created.is_error);
        let ambiguous = registry.invoke("delete_agenda_item", "{\"item_name\":\"call\"}");
        assert!(ambiguous.is_error);
        assert_eq!(
            ambiguous.content,
            "Multiple agenda items match 'call': call_dad, call_mom. Be more specific."
        );

        let delete = registry.invoke("delete_agenda_item", "{\"item_name\":\"desk notes\"}");
        assert!(!delete.is_error);
        assert_eq!(delete.content, "Agenda item 'desk notes' deleted.");
        assert!(!agenda_dir.join("desk_notes.md").exists());
        let deleted_context = persisted_context_messages(&config);
        assert!(
            !deleted_context.iter().any(|message| {
                message_matches_tool_call_id(message, "context-task:desk notes")
            })
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_tools_can_create_and_mutate_due_items() {
        let unique = format!(
            "elroy-rs-agenda-due-item-mutations-{}",
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
        config.database_path = database_path;

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let created = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom tonight\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!created.is_error);
        assert_eq!(
            created.content,
            "Contextual due item 'call mom' has been created."
        );

        let duplicate_contextual = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom tomorrow\",\"trigger_context\":\"tomorrow morning\"}",
        );
        assert!(duplicate_contextual.is_error);
        assert_eq!(
            duplicate_contextual.content,
            "Contextual due item 'call mom' already exists"
        );

        let missing_trigger = registry.invoke(
            "create_due_item",
            "{\"name\":\"call dad\",\"text\":\"This should fail\"}",
        );
        assert!(missing_trigger.is_error);
        assert_eq!(
            missing_trigger.content,
            "Either trigger_time or trigger_context must be provided for due items"
        );

        let blank_name = registry.invoke(
            "create_due_item",
            "{\"name\":\"   \",\"text\":\"This should fail\",\"trigger_context\":\"later\"}",
        );
        assert!(blank_name.is_error);
        assert_eq!(blank_name.content, "Due item name cannot be empty");
        assert!(agenda_dir.join("call_mom.md").exists());

        let context = persisted_context_messages(&config);
        assert!(context.iter().any(|message| {
            message.tool_calls.as_ref().is_some_and(|calls| {
                calls.iter().any(|call| {
                    call.id == context_due_item_tool_call_id("call mom")
                        && call.name == "get_fast_recall"
                })
            })
        }));
        assert!(context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call mom"))
        }));

        let timed = registry.invoke(
            "create_due_item",
            "{\"name\":\"pay rent\",\"text\":\"Pay rent\",\"trigger_time\":\"2099-05-16 09:00\"}",
        );
        assert!(!timed.is_error);
        assert_eq!(
            timed.content,
            "Timed due item 'pay rent' has been created for 2099-05-16 09:00."
        );

        let duplicate_timed = registry.invoke(
            "create_due_item",
            "{\"name\":\"pay rent\",\"text\":\"Pay rent later\",\"trigger_time\":\"2099-05-17 09:00\"}",
        );
        assert!(duplicate_timed.is_error);
        assert_eq!(
            duplicate_timed.content,
            "Timed due item 'pay rent' already exists"
        );

        let past_timed = registry.invoke(
            "create_due_item",
            "{\"name\":\"old reminder\",\"text\":\"This should fail\",\"trigger_time\":\"2000-01-01 09:00\"}",
        );
        assert!(past_timed.is_error);
        assert!(
            past_timed
                .content
                .contains("Attempted to create a due item for")
        );
        assert!(past_timed.content.contains("which is in the past"));

        let timed_context = persisted_context_messages(&config);
        assert!(timed_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("pay rent"))
        }));

        let updated = registry.invoke(
            "update_due_item_text",
            "{\"name\":\"call mom\",\"new_text\":\"Call mom after dinner\"}",
        );
        assert!(!updated.is_error);
        assert_eq!(
            updated.content,
            "Due item 'call mom' text has been updated."
        );
        let updated_context = persisted_context_messages(&config);
        assert!(updated_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call mom"))
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("Call mom after dinner"))
        }));

        let missing_updated = registry.invoke(
            "update_due_item_text",
            "{\"name\":\"missing\",\"new_text\":\"No-op\"}",
        );
        assert!(missing_updated.is_error);
        assert_eq!(
            missing_updated.content,
            "Due item 'missing' not found. Valid items: call mom, pay rent"
        );

        let renamed = registry.invoke(
            "rename_due_item",
            "{\"old_name\":\"call mom\",\"new_name\":\"Call Parents\"}",
        );
        assert!(!renamed.is_error);
        assert_eq!(
            renamed.content,
            "Due item 'call mom' has been renamed to 'Call Parents'."
        );
        assert!(agenda_dir.join("call_parents.md").exists());
        let renamed_context = persisted_context_messages(&config);
        assert!(!renamed_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call mom"))
        }));
        assert!(renamed_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call parents"))
        }));

        let missing_renamed = registry.invoke(
            "rename_due_item",
            "{\"old_name\":\"missing\",\"new_name\":\"Call Family\"}",
        );
        assert!(missing_renamed.is_error);
        assert!(
            missing_renamed
                .content
                .starts_with("Active due item 'missing' not found. Active items: ")
        );
        assert!(missing_renamed.content.contains("call parents"));
        assert!(missing_renamed.content.contains("pay rent"));

        let duplicate_renamed = registry.invoke(
            "rename_due_item",
            "{\"old_name\":\"call parents\",\"new_name\":\"call parents\"}",
        );
        assert!(duplicate_renamed.is_error);
        assert_eq!(
            duplicate_renamed.content,
            "Active due item 'call parents' already exists."
        );

        let completed = registry.invoke(
            "complete_due_item",
            "{\"name\":\"call parents\",\"closing_comment\":\"done\"}",
        );
        assert!(!completed.is_error);
        assert_eq!(
            completed.content,
            "Due item 'call parents' has been marked as completed. Comment: done"
        );
        let completed_text =
            fs::read_to_string(agenda_dir.join("call_parents.md")).expect("due item should read");
        assert!(completed_text.contains("completed: true"));
        let completed_context = persisted_context_messages(&config);
        assert!(!completed_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call parents"))
        }));

        let missing_completed = registry.invoke("complete_due_item", "{\"name\":\"missing\"}");
        assert!(missing_completed.is_error);
        assert_eq!(
            missing_completed.content,
            "Active due item 'missing' not found. Active due items: pay rent"
        );

        let deleted = registry.invoke(
            "delete_due_item",
            "{\"name\":\"pay rent\",\"closing_comment\":\"paid online\"}",
        );
        assert!(!deleted.is_error);
        assert_eq!(
            deleted.content,
            "Due item 'pay rent' has been deleted. Comment: paid online"
        );
        assert!(!agenda_dir.join("pay_bill.md").exists());

        let missing_deleted = registry.invoke("delete_due_item", "{\"name\":\"missing\"}");
        assert!(missing_deleted.is_error);
        assert_eq!(
            missing_deleted.content,
            "Active due item 'missing' not found. Active due items: "
        );

        let completed_recreated = registry.invoke(
            "create_due_item",
            "{\"name\":\"Call Parents\",\"text\":\"Call parents tomorrow\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!completed_recreated.is_error);
        assert_eq!(
            completed_recreated.content,
            "Contextual due item 'Call Parents' has been created."
        );
        let completed_recreated_shown =
            registry.invoke("show_due_item", "{\"name\":\"call parents\"}");
        assert!(!completed_recreated_shown.is_error);
        assert!(
            completed_recreated_shown
                .content
                .contains("Call parents tomorrow")
        );
        let completed_recreated_context = persisted_context_messages(&config);
        assert!(completed_recreated_context.iter().any(|message| {
            message_matches_tool_call_id(message, &context_due_item_tool_call_id("call parents"))
        }));

        let contextual_deleted = registry.invoke(
            "delete_due_item",
            "{\"name\":\"call parents\",\"closing_comment\":\"done\"}",
        );
        assert!(!contextual_deleted.is_error);
        let recreated = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom again\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!recreated.is_error);
        assert_eq!(
            recreated.content,
            "Contextual due item 'call mom' has been created."
        );

        let listed = registry.invoke("list_due_items", "{\"limit\":10}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("call mom"));
        assert!(!listed.content.contains("call parents"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_due_item_persists_embedding_when_embedding_config_is_available() {
        let unique = format!(
            "elroy-rs-agenda-due-item-embedding-{}",
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

        let mut server = mockito::Server::new();
        let embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex("Call mom tonight".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "data": [{"embedding": [1.0, 0.0, 0.5]}]
                })
                .to_string(),
            )
            .create();

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path.clone();
        config.embedding_model_api_key = Some("embedding-test-key".to_string());
        config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

        let registry = ExecutableToolRegistry::new(agenda_tools(&config));
        let created = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom tonight\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!created.is_error);

        let connection =
            open_sqlite_connection(&database_path).expect("sqlite connection should open");
        let due_item = elroy_db::find_active_agenda_item_by_name(&connection, "call mom")
            .expect("lookup should work")
            .expect("due item should exist");
        let embeddings = elroy_db::load_memory_embeddings_for_paths(
            &connection,
            std::slice::from_ref(&due_item.file_path),
        )
        .expect("embeddings should load");
        let embedding = embeddings
            .get(&due_item.file_path)
            .expect("due item embedding should exist");
        assert!(embedding.embedding_text.contains("Call mom tonight"));
        assert!(
            embedding
                .embedding_text
                .contains("trigger_context: after dinner")
        );

        embedding_mock.assert();
        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn due_item_tool_schemas_match_python_surface() {
        let config = AppConfig::defaults();
        let registry = ExecutableToolRegistry::new(agenda_tools(&config));

        let create_spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "create_due_item")
            .expect("create_due_item tool should exist");
        let create_properties = match &create_spec.parameters {
            JsonSchema::Object { properties, .. } => properties,
        };
        assert_eq!(create_properties.len(), 4);
        assert!(create_properties.contains_key("name"));
        assert!(create_properties.contains_key("text"));
        assert!(create_properties.contains_key("trigger_datetime"));
        assert!(create_properties.contains_key("trigger_context"));
        assert!(!create_properties.contains_key("date"));

        let update_spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "update_due_item_text")
            .expect("update_due_item_text tool should exist");
        let update_properties = match &update_spec.parameters {
            JsonSchema::Object { properties, .. } => properties,
        };
        assert_eq!(update_properties.len(), 2);
        assert!(update_properties.contains_key("name"));
        assert!(update_properties.contains_key("new_text"));
        assert!(!update_properties.contains_key("text"));

        let rename_spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "rename_due_item")
            .expect("rename_due_item tool should exist");
        let rename_properties = match &rename_spec.parameters {
            JsonSchema::Object { properties, .. } => properties,
        };
        assert_eq!(rename_properties.len(), 2);
        assert!(rename_properties.contains_key("old_name"));
        assert!(rename_properties.contains_key("new_name"));
        assert!(!rename_properties.contains_key("name"));

        for tool_name in ["print_active_due_items", "print_inactive_due_items"] {
            let print_spec = registry
                .specs()
                .into_iter()
                .find(|spec| spec.name == tool_name)
                .unwrap_or_else(|| panic!("{tool_name} tool should exist"));
            let print_properties = match &print_spec.parameters {
                JsonSchema::Object { properties, .. } => properties,
            };
            assert_eq!(
                print_properties.len(),
                1,
                "{tool_name} should expose only one field"
            );
            assert!(
                print_properties.contains_key("n"),
                "{tool_name} should expose n"
            );
            assert!(
                !print_properties.contains_key("limit"),
                "{tool_name} should not expose limit"
            );
        }
    }
}
