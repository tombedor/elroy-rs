use std::path::{Path, PathBuf};

use elroy_agenda::{
    add_checklist_item, append_agenda_update, create_agenda_file, get_checklist,
    mark_agenda_item_completed, mark_agenda_item_deleted, rename_agenda_file, update_agenda_body,
    update_checklist_item,
};
use elroy_config::{AppConfig, LlmProvider};
use elroy_core::{ConversationOrchestrator, LiveProviderModel, LocalToolExecutor};
use elroy_db::{
    BootstrapPlan, find_active_agenda_item_by_name, find_active_memory_by_name,
    list_active_due_items, list_active_memories, list_active_plain_agenda_items,
    load_context_messages, open_sqlite_connection, replace_context_messages, run_migrations,
    search_active_memories,
};
use elroy_llm::{LiveModelClient, Provider, ProviderConfig, StreamEvent};
use elroy_memory::{archive_memory_file, create_memory_file, update_memory_body};
use elroy_tools::{
    ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry, ToolSpec,
};
use elroy_tui::{SidebarAction, SidebarSection, TuiSnapshot};
use serde_json::{Value, json};

const LOCAL_USER_TOKEN: &str = "local-user";

#[derive(Debug)]
pub enum AppError {
    Io(std::io::Error),
    Sql(rusqlite::Error),
    Migration(refinery::Error),
    ProviderConfig(String),
    Model(elroy_core::ModelClientError),
    Runtime(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Sql(error) => write!(f, "{error}"),
            Self::Migration(error) => write!(f, "{error}"),
            Self::ProviderConfig(error) => write!(f, "{error}"),
            Self::Model(error) => write!(f, "{error}"),
            Self::Runtime(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sql(value)
    }
}

impl From<refinery::Error> for AppError {
    fn from(value: refinery::Error) -> Self {
        Self::Migration(value)
    }
}

impl From<elroy_core::ModelClientError> for AppError {
    fn from(value: elroy_core::ModelClientError) -> Self {
        Self::Model(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRunResult {
    pub events: Vec<StreamEvent>,
    pub snapshot: TuiSnapshot,
}

#[derive(Debug, Clone)]
pub struct AppRuntime {
    config: AppConfig,
}

impl AppRuntime {
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn load_snapshot(&self) -> Result<TuiSnapshot, AppError> {
        let mut connection = self.open_connection()?;
        let conversation_lines = load_context_messages(&mut connection, LOCAL_USER_TOKEN)?
            .into_iter()
            .map(|message| {
                let role = match message.role {
                    elroy_llm::MessageRole::System => "system",
                    elroy_llm::MessageRole::User => "user",
                    elroy_llm::MessageRole::Assistant => "assistant",
                    elroy_llm::MessageRole::Tool => "tool",
                };
                let content = message.content.unwrap_or_default();
                format!("{role}: {content}")
            })
            .collect::<Vec<_>>();
        let memory_titles = list_active_memories(&connection, 15)?
            .into_iter()
            .map(|memory| memory.name)
            .collect::<Vec<_>>();
        let agenda_titles = list_active_plain_agenda_items(&connection, 15)?
            .into_iter()
            .map(|item| item.name)
            .collect::<Vec<_>>();

        Ok(TuiSnapshot {
            conversation_lines,
            memory_titles,
            agenda_titles,
            status: Some("loaded persisted transcript and sidebar data".to_string()),
        })
    }

    pub fn submit_prompt(&self, prompt: &str) -> Result<PromptRunResult, AppError> {
        let mut connection = self.open_connection()?;
        let provider_config =
            provider_config_from_app_config(&self.config).map_err(AppError::ProviderConfig)?;
        let client = LiveModelClient::new(provider_config)
            .map_err(|error| AppError::Runtime(error.to_string()))?;
        let executable_tools = build_live_tool_registry(&self.config);
        let tools = ToolRegistry::new(executable_tools.specs());
        let model =
            LiveProviderModel::new(client, format!("You are {}.", self.config.assistant_name));
        let orchestrator = ConversationOrchestrator::new(2);
        let tool_executor = LocalToolExecutor::new(executable_tools);
        let existing_transcript = load_context_messages(&mut connection, LOCAL_USER_TOKEN)?;

        let turn_run = orchestrator.run_turn_with_transcript(
            &model,
            tools.specs(),
            &tool_executor,
            &existing_transcript,
            prompt,
        )?;
        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &turn_run.transcript)?;

        Ok(PromptRunResult {
            events: turn_run.events,
            snapshot: self.load_snapshot()?,
        })
    }

    pub fn open_sidebar_item(
        &self,
        section: SidebarSection,
        title: &str,
    ) -> Result<String, AppError> {
        let connection = self.open_read_connection()?;
        match section {
            SidebarSection::Memories => {
                let Some(memory) = find_active_memory_by_name(&connection, title)? else {
                    return Err(AppError::Runtime(format!("memory not found: {title}")));
                };
                Ok(format!(
                    "memory: {}\npath: {}\n\n{}",
                    memory.name, memory.file_path, memory.body
                ))
            }
            SidebarSection::Agenda => {
                let Some(item) = find_active_agenda_item_by_name(&connection, title)? else {
                    return Err(AppError::Runtime(format!("agenda item not found: {title}")));
                };
                let mut lines = vec![
                    format!("agenda: {}", item.name),
                    format!("path: {}", item.file_path),
                ];
                if let Some(date) = item.agenda_date {
                    lines.push(format!("date: {date}"));
                }
                if let Some(trigger_datetime) = item.trigger_datetime {
                    lines.push(format!("trigger_datetime: {trigger_datetime}"));
                }
                if let Some(trigger_context) = item.trigger_context {
                    lines.push(format!("trigger_context: {trigger_context}"));
                }
                if item.checklist_total > 0 {
                    lines.push(format!(
                        "checklist: {}/{} completed",
                        item.checklist_completed, item.checklist_total
                    ));
                }
                lines.push(String::new());
                lines.push(item.body);
                Ok(lines.join("\n"))
            }
        }
    }

    pub fn mutate_sidebar_item(
        &self,
        section: SidebarSection,
        title: &str,
        action: SidebarAction,
    ) -> Result<TuiSnapshot, AppError> {
        let registry = build_live_tool_registry(&self.config);
        let result = match (section, action) {
            (SidebarSection::Memories, SidebarAction::Archive) => {
                registry.invoke("archive_memory", &json!({ "name": title }).to_string())
            }
            (SidebarSection::Agenda, SidebarAction::Complete) => registry.invoke(
                "complete_agenda_item",
                &json!({ "name": title }).to_string(),
            ),
            (SidebarSection::Agenda, SidebarAction::Delete) => {
                registry.invoke("delete_agenda_item", &json!({ "name": title }).to_string())
            }
            (SidebarSection::Memories, SidebarAction::Complete | SidebarAction::Delete) => {
                return Err(AppError::Runtime(
                    "unsupported memory sidebar action".to_string(),
                ));
            }
            (SidebarSection::Agenda, SidebarAction::Archive) => {
                return Err(AppError::Runtime(
                    "unsupported agenda sidebar action".to_string(),
                ));
            }
        };
        if result.is_error {
            return Err(AppError::Runtime(result.content));
        }
        self.load_snapshot()
    }

    fn open_connection(&self) -> Result<rusqlite::Connection, AppError> {
        let mut connection = open_sqlite_connection(&self.config.database_path)?;
        run_migrations(&mut connection)?;
        Ok(connection)
    }

    fn open_read_connection(&self) -> Result<rusqlite::Connection, AppError> {
        let connection = open_sqlite_connection(&self.config.database_path)?;
        Ok(connection)
    }
}

pub fn build_live_tool_registry(config: &AppConfig) -> ExecutableToolRegistry {
    let config_for_memory_write = config.clone();
    let create_memory = ExecutableTool::new(
        ToolSpec::new(
            "create_memory",
            "Create a new file-backed memory and rebuild derived state.",
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
                return ToolExecutionResult::error("create_memory requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_memory requires string text");
            };
            match create_memory_file(&config_for_memory_write.memory_dir, name, text).and_then(
                |path| {
                    elroy_db::bootstrap_database(&BootstrapPlan::from_config(
                        &config_for_memory_write,
                    ))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                    Ok(path)
                },
            ) {
                Ok(path) => ToolExecutionResult::success(
                    json!({
                        "created": true,
                        "file_path": path.display().to_string(),
                    })
                    .to_string(),
                ),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create memory: {error}"))
                }
            }
        },
    );

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
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item requires string text");
            };
            let date = arguments.get("date").and_then(Value::as_str);
            let trigger_datetime = arguments.get("trigger_datetime").and_then(Value::as_str);
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);

            match create_agenda_file(
                &config_for_agenda_write.agenda_dir,
                name,
                text,
                date,
                trigger_datetime,
                trigger_context,
            )
            .and_then(|path| {
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_agenda_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            }) {
                Ok(path) => ToolExecutionResult::success(
                    json!({
                        "created": true,
                        "file_path": path.display().to_string(),
                    })
                    .to_string(),
                ),
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
                    ("date", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_due_item requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_due_item requires string text");
            };
            let trigger_datetime = arguments.get("trigger_datetime").and_then(Value::as_str);
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);
            if trigger_datetime.is_none() && trigger_context.is_none() {
                return ToolExecutionResult::error(
                    "create_due_item requires trigger_datetime or trigger_context",
                );
            }
            let date = arguments.get("date").and_then(Value::as_str);
            match create_agenda_file(
                &config_for_due_item_write.agenda_dir,
                name,
                text,
                date,
                trigger_datetime,
                trigger_context,
            )
            .and_then(|path| {
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(
                    &config_for_due_item_write,
                ))
                .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            }) {
                Ok(path) => ToolExecutionResult::success(
                    json!({
                        "created": true,
                        "file_path": path.display().to_string(),
                    })
                    .to_string(),
                ),
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
                    ("text", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_due_item_text requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_due_item_text requires string text");
            };
            mutate_agenda_file_from_config(&config_for_due_item_text, name, |path| {
                update_agenda_body(path, text)
            })
        },
    );

    let config_for_due_item_rename = config.clone();
    let rename_due_item = ExecutableTool::new(
        ToolSpec::new(
            "rename_due_item",
            "Rename one active due item.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("new_name", json!({"type": "string"})),
                ],
                ["name", "new_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("rename_due_item requires a string name");
            };
            let Some(new_name) = arguments.get("new_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("rename_due_item requires string new_name");
            };
            mutate_agenda_file_from_config_with_result(&config_for_due_item_rename, name, |path| {
                let renamed = rename_agenda_file(path, new_name)?;
                Ok(json!({
                    "updated": true,
                    "new_file_path": renamed.display().to_string(),
                })
                .to_string())
            })
        },
    );

    let config_for_due_item_complete = config.clone();
    let complete_due_item = ExecutableTool::new(
        ToolSpec::new(
            "complete_due_item",
            "Mark one active due item completed.",
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
                return ToolExecutionResult::error("complete_due_item requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            mutate_agenda_file_from_config(&config_for_due_item_complete, name, |path| {
                mark_agenda_item_completed(path, closing_comment)
            })
        },
    );

    let config_for_due_item_delete = config.clone();
    let delete_due_item = ExecutableTool::new(
        ToolSpec::new(
            "delete_due_item",
            "Mark one active due item deleted.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("delete_due_item requires a string name");
            };
            mutate_agenda_file_from_config(
                &config_for_due_item_delete,
                name,
                mark_agenda_item_deleted,
            )
        },
    );

    let config_for_memory_update = config.clone();
    let update_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_memory",
            "Replace the body text of one active memory by exact name.",
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
                return ToolExecutionResult::error("update_memory requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_memory requires string text");
            };
            mutate_memory_file_from_config(&config_for_memory_update, name, |path| {
                update_memory_body(path, text)
            })
        },
    );

    let config_for_memory_archive = config.clone();
    let archive_memory = ExecutableTool::new(
        ToolSpec::new(
            "archive_memory",
            "Archive one active memory by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("archive_memory requires a string name");
            };
            let archive_dir = config_for_memory_archive.memory_dir.join("archive");
            archive_memory_file_from_config(&config_for_memory_archive, name, |path| {
                archive_memory_file(path, &archive_dir)
            })
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
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item_update requires a string name");
            };
            let Some(note) = arguments.get("note").and_then(Value::as_str) else {
                return ToolExecutionResult::error("add_agenda_item_update requires string note");
            };
            mutate_agenda_file_from_config(&config_for_agenda_update, name, |path| {
                append_agenda_update(path, note)
            })
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
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("complete_agenda_item requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            mutate_agenda_file_from_config(&config_for_agenda_complete, name, |path| {
                mark_agenda_item_completed(path, closing_comment)
            })
        },
    );

    let config_for_agenda_delete = config.clone();
    let delete_agenda_item = ExecutableTool::new(
        ToolSpec::new(
            "delete_agenda_item",
            "Mark one active agenda item as deleted.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("delete_agenda_item requires a string name");
            };
            mutate_agenda_file_from_config(
                &config_for_agenda_delete,
                name,
                mark_agenda_item_deleted,
            )
        },
    );

    let config_for_agenda_checklist_add = config.clone();
    let add_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "add_agenda_checklist_item",
            "Add a checklist item to one active agenda item.",
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
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "add_agenda_checklist_item requires a string name",
                );
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "add_agenda_checklist_item requires string text",
                );
            };
            let due_date = arguments.get("due_date").and_then(Value::as_str);
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_add,
                name,
                |path| {
                    let item_id = add_checklist_item(path, text, due_date)?;
                    let checklist = get_checklist(path)?;
                    Ok(json!({
                        "updated": true,
                        "checklist_item_id": item_id,
                        "checklist_count": checklist.len(),
                    })
                    .to_string())
                },
            )
        },
    );

    let config_for_agenda_checklist_edit = config.clone();
    let edit_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "edit_agenda_checklist_item",
            "Edit the text of a checklist item on one active agenda item.",
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
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires a string name",
                );
            };
            let Some(item_id) = arguments.get("item_id").and_then(Value::as_i64) else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires integer item_id",
                );
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires string text",
                );
            };
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_edit,
                name,
                |path| {
                    let updated = update_checklist_item(path, item_id, Some(text), None)?;
                    Ok(json!({
                        "updated": true,
                        "item_id": updated.id,
                        "text": updated.text,
                        "completed": updated.completed,
                    })
                    .to_string())
                },
            )
        },
    );

    let config_for_agenda_checklist_complete = config.clone();
    let complete_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "complete_agenda_checklist_item",
            "Mark a checklist item completed on one active agenda item.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("item_id", json!({"type": "integer"})),
                ],
                ["name", "item_id"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "complete_agenda_checklist_item requires a string name",
                );
            };
            let Some(item_id) = arguments.get("item_id").and_then(Value::as_i64) else {
                return ToolExecutionResult::error(
                    "complete_agenda_checklist_item requires integer item_id",
                );
            };
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_complete,
                name,
                |path| {
                    let updated = update_checklist_item(path, item_id, None, Some(true))?;
                    Ok(json!({
                        "updated": true,
                        "item_id": updated.id,
                        "text": updated.text,
                        "completed": updated.completed,
                    })
                    .to_string())
                },
            )
        },
    );

    let database_path = config.database_path.clone();
    let show_context_messages = ExecutableTool::new(
        ToolSpec::new(
            "show_context_messages",
            "Show the persisted local conversation transcript.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 20);
            let mut connection = match open_sqlite_connection(&database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            match load_context_messages(&mut connection, LOCAL_USER_TOKEN) {
                Ok(messages) => {
                    let payload = messages
                        .into_iter()
                        .rev()
                        .take(limit)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .map(|message| {
                            json!({
                                "role": format!("{:?}", message.role).to_ascii_lowercase(),
                                "content": message.content,
                                "tool_call_id": message.tool_call_id,
                                "tool_calls": message.tool_calls,
                            })
                        })
                        .collect::<Vec<_>>();
                    ToolExecutionResult::success(
                        serde_json::to_string_pretty(&payload)
                            .expect("context payload should serialize"),
                    )
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to load context messages: {error}"))
                }
            }
        },
    );

    let config_for_clear_context = config.clone();
    let clear_context_messages = ExecutableTool::new(
        ToolSpec::new(
            "clear_context_messages",
            "Clear the persisted local conversation transcript.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut connection =
                match open_sqlite_connection(&config_for_clear_context.database_path) {
                    Ok(connection) => connection,
                    Err(error) => {
                        return ToolExecutionResult::error(format!(
                            "failed to open database: {error}"
                        ));
                    }
                };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            match replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &[]) {
                Ok(()) => ToolExecutionResult::success("{\"cleared\":true}".to_string()),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to clear context messages: {error}"))
                }
            }
        },
    );

    let database_path = config.database_path.clone();
    let list_memories = ExecutableTool::new(
        ToolSpec::new(
            "list_memories",
            "List active memories available to Elroy.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let memories = list_active_memories(connection, limit)?;
                let payload = memories
                    .into_iter()
                    .map(|memory| {
                        json!({
                            "name": memory.name,
                            "file_path": memory.file_path,
                            "excerpt": excerpt(&memory.body, 180),
                            "updated_at_unix": memory.updated_at_unix,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("memory payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let search_memories = ExecutableTool::new(
        ToolSpec::new(
            "search_memories",
            "Search active memories by keyword.",
            JsonSchema::object(
                [
                    ("query", json!({"type": "string"})),
                    ("limit", json!({"type": "integer"})),
                ],
                ["query"],
            ),
        ),
        move |arguments| {
            let Some(query) = arguments.get("query").and_then(Value::as_str) else {
                return ToolExecutionResult::error("search_memories requires a string query");
            };
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let memories = search_active_memories(connection, query, limit)?;
                let payload = memories
                    .into_iter()
                    .map(|memory| {
                        json!({
                            "name": memory.name,
                            "file_path": memory.file_path,
                            "excerpt": excerpt(&memory.body, 180),
                            "updated_at_unix": memory.updated_at_unix,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("memory search payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_agenda = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda",
            "List active agenda items and reminders.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_active_plain_agenda_items(connection, limit)?;
                let payload = items
                    .into_iter()
                    .map(|item| {
                        json!({
                            "name": item.name,
                            "agenda_date": item.agenda_date,
                            "trigger_datetime": item.trigger_datetime,
                            "trigger_context": item.trigger_context,
                            "status": item.status,
                            "checklist_total": item.checklist_total,
                            "checklist_completed": item.checklist_completed,
                            "excerpt": excerpt(&item.body, 180),
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("agenda payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_due_items = ExecutableTool::new(
        ToolSpec::new(
            "list_due_items",
            "List active due items and reminders.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_active_due_items(connection, limit)?;
                let payload = items
                    .into_iter()
                    .map(|item| {
                        json!({
                            "name": item.name,
                            "agenda_date": item.agenda_date,
                            "trigger_datetime": item.trigger_datetime,
                            "trigger_context": item.trigger_context,
                            "status": item.status,
                            "checklist_total": item.checklist_total,
                            "checklist_completed": item.checklist_completed,
                            "excerpt": excerpt(&item.body, 180),
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("due item payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let show_memory = ExecutableTool::new(
        ToolSpec::new(
            "show_memory",
            "Show one active memory by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("show_memory requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name(connection, name)? else {
                    return Ok(ToolExecutionResult::error(format!(
                        "memory not found: {name}"
                    )));
                };
                Ok(ToolExecutionResult::success(
                    json!({
                        "name": memory.name,
                        "file_path": memory.file_path,
                        "body": memory.body,
                        "updated_at_unix": memory.updated_at_unix,
                    })
                    .to_string(),
                ))
            })
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
            with_tool_connection(&database_path, |connection| {
                let Some(item) = find_active_agenda_item_by_name(connection, name)? else {
                    return Ok(ToolExecutionResult::error(format!(
                        "agenda item not found: {name}"
                    )));
                };
                Ok(ToolExecutionResult::success(
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
                ))
            })
        },
    );

    ExecutableToolRegistry::new(vec![
        create_memory,
        add_agenda_item,
        create_due_item,
        update_due_item_text,
        rename_due_item,
        complete_due_item,
        delete_due_item,
        update_memory,
        archive_memory,
        add_agenda_item_update,
        complete_agenda_item,
        delete_agenda_item,
        add_agenda_checklist_item,
        edit_agenda_checklist_item,
        complete_agenda_checklist_item,
        show_context_messages,
        clear_context_messages,
        list_memories,
        search_memories,
        list_agenda,
        list_due_items,
        show_memory,
        show_agenda_item,
    ])
}

pub fn argument_limit(arguments: &Value, default_limit: usize) -> usize {
    match arguments.get("limit").and_then(Value::as_u64) {
        Some(0) | None => default_limit,
        Some(value) => value.clamp(1, 50) as usize,
    }
}

fn mutate_memory_file_from_config(
    config: &AppConfig,
    name: &str,
    operation: impl FnOnce(&Path) -> std::io::Result<()>,
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
    let memory = match find_active_memory_by_name(&connection, name) {
        Ok(Some(memory)) => memory,
        Ok(None) => return ToolExecutionResult::error(format!("memory not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&memory.file_path)).and_then(|()| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(())
    }) {
        Ok(()) => ToolExecutionResult::success(
            json!({"updated": true, "name": memory.name, "file_path": memory.file_path})
                .to_string(),
        ),
        Err(error) => ToolExecutionResult::error(format!("memory mutation failed: {error}")),
    }
}

fn archive_memory_file_from_config(
    config: &AppConfig,
    name: &str,
    operation: impl FnOnce(&Path) -> std::io::Result<PathBuf>,
) -> ToolExecutionResult {
    let connection = match open_sqlite_connection(&config.database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    let memory = match find_active_memory_by_name(&connection, name) {
        Ok(Some(memory)) => memory,
        Ok(None) => return ToolExecutionResult::error(format!("memory not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&memory.file_path)).and_then(|new_path| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(new_path)
    }) {
        Ok(new_path) => ToolExecutionResult::success(
            json!({
                "updated": true,
                "name": memory.name,
                "file_path": memory.file_path,
                "moved_to": new_path.display().to_string(),
            })
            .to_string(),
        ),
        Err(error) => ToolExecutionResult::error(format!("memory mutation failed: {error}")),
    }
}

fn mutate_agenda_file_from_config(
    config: &AppConfig,
    name: &str,
    operation: impl FnOnce(&Path) -> std::io::Result<()>,
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
    let item = match find_active_agenda_item_by_name(&connection, name) {
        Ok(Some(item)) => item,
        Ok(None) => return ToolExecutionResult::error(format!("agenda item not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&item.file_path)).and_then(|()| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(())
    }) {
        Ok(()) => ToolExecutionResult::success(
            json!({"updated": true, "name": item.name, "file_path": item.file_path}).to_string(),
        ),
        Err(error) => ToolExecutionResult::error(format!("agenda mutation failed: {error}")),
    }
}

fn mutate_agenda_file_from_config_with_result(
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
    let item = match find_active_agenda_item_by_name(&connection, name) {
        Ok(Some(item)) => item,
        Ok(None) => return ToolExecutionResult::error(format!("agenda item not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&item.file_path)).and_then(|payload| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(payload)
    }) {
        Ok(payload) => ToolExecutionResult::success(payload),
        Err(error) => ToolExecutionResult::error(format!("agenda mutation failed: {error}")),
    }
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

fn excerpt(body: &str, max_chars: usize) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let shortened = trimmed.chars().take(max_chars).collect::<String>();
    format!("{shortened}...")
}

pub fn provider_config_from_app_config(config: &AppConfig) -> Result<ProviderConfig, String> {
    match config.llm_provider() {
        LlmProvider::OpenAi => {
            let api_key = config
                .openai_api_key
                .clone()
                .ok_or_else(|| "missing OPENAI_API_KEY for OpenAI model".to_string())?;
            Ok(ProviderConfig {
                provider: Provider::OpenAi,
                model: config.chat_model.clone(),
                api_key,
                base_url: config.openai_base_url.clone(),
                anthropic_api_version: None,
                timeout_seconds: 60,
                max_output_tokens: 2048,
            })
        }
        LlmProvider::Anthropic => {
            let api_key = config
                .anthropic_api_key
                .clone()
                .ok_or_else(|| "missing ANTHROPIC_API_KEY for Anthropic model".to_string())?;
            Ok(ProviderConfig {
                provider: Provider::Anthropic,
                model: config.chat_model.clone(),
                api_key,
                base_url: config.anthropic_base_url.clone(),
                anthropic_api_version: Some(config.anthropic_api_version.clone()),
                timeout_seconds: 60,
                max_output_tokens: 2048,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use elroy_agenda::create_agenda_file;
    use elroy_config::{AppConfig, LlmProvider};
    use elroy_llm::{ConversationMessage, MessageRole, Provider};
    use elroy_memory::{create_memory_file, sanitize_filename};

    use super::{
        AppRuntime, LOCAL_USER_TOKEN, argument_limit, build_live_tool_registry,
        provider_config_from_app_config,
    };

    #[test]
    fn provider_config_uses_openai_when_model_is_not_claude() {
        let mut config = AppConfig::defaults();
        config.chat_model = "gpt-5.4".to_string();
        config.openai_api_key = Some("openai-key".to_string());

        let provider = provider_config_from_app_config(&config).expect("config should build");

        assert_eq!(config.llm_provider(), LlmProvider::OpenAi);
        assert_eq!(provider.provider, Provider::OpenAi);
        assert_eq!(provider.api_key, "openai-key");
    }

    #[test]
    fn provider_config_uses_anthropic_when_model_is_claude() {
        let mut config = AppConfig::defaults();
        config.chat_model = "claude-sonnet-4-20250514".to_string();
        config.anthropic_api_key = Some("anthropic-key".to_string());

        let provider = provider_config_from_app_config(&config).expect("config should build");

        assert_eq!(config.llm_provider(), LlmProvider::Anthropic);
        assert_eq!(provider.provider, Provider::Anthropic);
        assert_eq!(provider.api_key, "anthropic-key");
    }

    #[test]
    fn argument_limit_clamps_values() {
        assert_eq!(argument_limit(&serde_json::json!({}), 10), 10);
        assert_eq!(argument_limit(&serde_json::json!({"limit": 0}), 10), 10);
        assert_eq!(argument_limit(&serde_json::json!({"limit": 100}), 10), 50);
        assert_eq!(argument_limit(&serde_json::json!({"limit": 7}), 10), 7);
    }

    #[test]
    fn filename_sanitization_and_file_creation_are_stable() {
        assert_eq!(sanitize_filename("Runner Notes"), "runner_notes");
        assert_eq!(sanitize_filename("!!!"), "item");

        let unique = format!(
            "elroy-rs-app-files-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("root should be created");

        let memory = create_memory_file(&root, "Runner Notes", "Remember this")
            .expect("memory file should be created");
        let agenda = create_agenda_file(
            &root,
            "Doctor Visit",
            "Bring forms",
            Some("2026-05-15"),
            Some("2026-05-15T15:00:00"),
            Some("after lunch"),
        )
        .expect("agenda file should be created");

        assert!(memory.ends_with("runner_notes.md"));
        assert!(agenda.ends_with("doctor_visit.md"));
        assert!(
            fs::read_to_string(memory)
                .expect("memory file should be readable")
                .contains("Remember this")
        );
        let agenda_text = fs::read_to_string(agenda).expect("agenda file should be readable");
        assert!(agenda_text.contains("date: 2026-05-15"));
        assert!(agenda_text.contains("trigger_context: after lunch"));

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn app_runtime_loads_snapshot_and_opens_sidebar_details() {
        let unique = format!(
            "elroy-rs-app-runtime-{}",
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
            memory_dir.join("runner_notes.md"),
            "remember the hill workout\n",
        )
        .expect("memory file should be written");
        fs::write(
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\n---\n\nbring forms\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        let mut connection =
            elroy_db::open_sqlite_connection(&config.database_path).expect("db should open");
        elroy_db::run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "hello again",
            )],
        )
        .expect("messages should persist");

        let runtime = AppRuntime::new(config);
        let snapshot = runtime.load_snapshot().expect("snapshot should load");
        let detail = runtime
            .open_sidebar_item(elroy_tui::SidebarSection::Memories, "runner notes")
            .expect("memory detail should open");

        assert!(
            snapshot
                .conversation_lines
                .iter()
                .any(|line| line.contains("hello again"))
        );
        assert!(
            snapshot
                .memory_titles
                .iter()
                .any(|item| item == "runner notes")
        );
        assert!(detail.contains("remember the hill workout"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_show_exact_memory_and_agenda_items() {
        let unique = format!(
            "elroy-rs-app-show-tools-{}",
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
            memory_dir.join("runner_notes.md"),
            "remember the hill workout\n",
        )
        .expect("memory file should be written");
        fs::write(
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\n---\n\nbring forms\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let memory = registry.invoke("show_memory", "{\"name\":\"runner notes\"}");
        let agenda = registry.invoke("show_agenda_item", "{\"name\":\"doctor visit\"}");

        assert!(!memory.is_error);
        assert!(memory.content.contains("remember the hill workout"));
        assert!(!agenda.is_error);
        assert!(agenda.content.contains("bring forms"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_update_and_archive_memories() {
        let unique = format!(
            "elroy-rs-app-memory-mutations-{}",
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
        fs::write(memory_dir.join("runner_notes.md"), "old text\n")
            .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let update = registry.invoke(
            "update_memory",
            "{\"name\":\"runner notes\",\"text\":\"new text\"}",
        );
        assert!(!update.is_error);
        assert!(
            fs::read_to_string(memory_dir.join("runner_notes.md"))
                .expect("updated memory should be readable")
                .contains("new text")
        );

        let archive = registry.invoke("archive_memory", "{\"name\":\"runner notes\"}");
        assert!(!archive.is_error);
        assert!(memory_dir.join("archive").join("runner_notes.md").exists());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_update_complete_and_delete_agenda_items() {
        let unique = format!(
            "elroy-rs-app-agenda-mutations-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let update = registry.invoke(
            "add_agenda_item_update",
            "{\"name\":\"doctor visit\",\"note\":\"called ahead\"}",
        );
        assert!(!update.is_error);
        let updated_text =
            fs::read_to_string(agenda_dir.join("doctor_visit.md")).expect("agenda should read");
        assert!(updated_text.contains("## Updates"));
        assert!(updated_text.contains("called ahead"));

        let complete = registry.invoke(
            "complete_agenda_item",
            "{\"name\":\"doctor visit\",\"closing_comment\":\"done\"}",
        );
        assert!(!complete.is_error);
        let completed_text =
            fs::read_to_string(agenda_dir.join("doctor_visit.md")).expect("agenda should read");
        assert!(completed_text.contains("completed: true"));
        assert!(completed_text.contains("status: completed"));

        fs::write(
            agenda_dir.join("call_mom.md"),
            "---\ndate: 2026-05-16\ncompleted: false\nstatus: created\n---\n\ncall mom\n",
        )
        .expect("second agenda file should be written");
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        let delete = registry.invoke("delete_agenda_item", "{\"name\":\"call mom\"}");
        assert!(!delete.is_error);
        let deleted_text =
            fs::read_to_string(agenda_dir.join("call_mom.md")).expect("agenda should read");
        assert!(deleted_text.contains("status: deleted"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_manage_agenda_checklists() {
        let unique = format!(
            "elroy-rs-app-agenda-checklists-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let added = registry.invoke(
            "add_agenda_checklist_item",
            "{\"name\":\"trip\",\"text\":\"passport\",\"due_date\":\"2026-05-14\"}",
        );
        assert!(!added.is_error);
        assert!(added.content.contains("\"checklist_item_id\":1"));

        let edited = registry.invoke(
            "edit_agenda_checklist_item",
            "{\"name\":\"trip\",\"item_id\":1,\"text\":\"passport + visa\"}",
        );
        assert!(!edited.is_error);
        assert!(edited.content.contains("passport + visa"));

        let completed = registry.invoke(
            "complete_agenda_checklist_item",
            "{\"name\":\"trip\",\"item_id\":1}",
        );
        assert!(!completed.is_error);
        assert!(completed.content.contains("\"completed\":true"));

        let file_text = fs::read_to_string(agenda_dir.join("trip.md")).expect("agenda should read");
        assert!(file_text.contains("passport + visa"));
        assert!(file_text.contains("completed: true"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_create_and_list_due_items() {
        let unique = format!(
            "elroy-rs-app-due-items-{}",
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

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom tonight\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!created.is_error);
        assert!(agenda_dir.join("call_mom.md").exists());

        let listed = registry.invoke("list_due_items", "{\"limit\":10}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("call mom"));
        assert!(listed.content.contains("after dinner"));
    }

    #[test]
    fn live_tool_registry_can_update_rename_complete_and_delete_due_items() {
        let unique = format!(
            "elroy-rs-app-due-item-mutations-{}",
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
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after dinner\n---\n\nCall mom tonight\n",
        )
        .expect("due item file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let updated = registry.invoke(
            "update_due_item_text",
            "{\"name\":\"call mom\",\"text\":\"Call mom after dinner\"}",
        );
        assert!(!updated.is_error);

        let renamed = registry.invoke(
            "rename_due_item",
            "{\"name\":\"call mom\",\"new_name\":\"Call Parents\"}",
        );
        assert!(!renamed.is_error);
        assert!(agenda_dir.join("call_parents.md").exists());

        let completed = registry.invoke(
            "complete_due_item",
            "{\"name\":\"call parents\",\"closing_comment\":\"done\"}",
        );
        assert!(!completed.is_error);
        let completed_text =
            fs::read_to_string(agenda_dir.join("call_parents.md")).expect("due item should read");
        assert!(completed_text.contains("completed: true"));

        fs::write(
            agenda_dir.join("pay_bill.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: tomorrow\n---\n\nPay bill\n",
        )
        .expect("second due item file should be written");
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        let deleted = registry.invoke("delete_due_item", "{\"name\":\"pay bill\"}");
        assert!(!deleted.is_error);
        let deleted_text =
            fs::read_to_string(agenda_dir.join("pay_bill.md")).expect("due item should read");
        assert!(deleted_text.contains("status: deleted"));
    }
}
