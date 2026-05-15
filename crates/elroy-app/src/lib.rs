use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use elroy_agenda::{
    add_checklist_item, append_agenda_update, create_agenda_file, get_checklist,
    mark_agenda_item_completed, mark_agenda_item_deleted, rename_agenda_file, update_agenda_body,
    update_checklist_item,
};
use elroy_codex::{
    CodexSessionResult, dispatch_codex_session_with_bin, dispatch_codex_session_with_hook,
    get_codex_session_by_thread_id, list_recent_codex_sessions, resume_codex_session_with_bin,
    resume_codex_session_with_hook,
};
use elroy_config::{AppConfig, LlmProvider};
use elroy_core::{
    ConversationOrchestrator, LiveProviderModel, LocalToolExecutor, ModelClient,
    validated_transcript,
};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, UserPreferenceRecord, find_active_agenda_item_by_name,
    find_active_memory_by_name, list_active_due_items, list_active_memories,
    list_active_plain_agenda_items, load_context_messages, load_user_preferences,
    open_sqlite_connection, replace_context_messages, run_migrations, save_user_preferences,
    search_active_memories,
};
use elroy_llm::{
    ConversationMessage, LiveModelClient, MessageRole, Provider, ProviderConfig, StreamEvent,
    ToolCall,
};
use elroy_memory::{archive_memory_file, create_memory_file, update_memory_body};
use elroy_tasks::{
    complete_task_file, create_task_file_with_schedule, delete_task_file, find_task_by_name,
    list_active_tasks, list_due_tasks, list_today_tasks, list_triggered_tasks, rename_task_file,
    update_task_text_file,
};
use elroy_tools::{
    ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry, ToolSpec,
};
use elroy_tui::{SidebarAction, SidebarSection, TuiSnapshot};
use elroy_user::{effective_persona, effective_user_full_name, effective_user_preferred_name};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageProcessOptions {
    pub enable_tools: bool,
    pub persist_input_message: bool,
}

impl Default for MessageProcessOptions {
    fn default() -> Self {
        Self {
            enable_tools: true,
            persist_input_message: true,
        }
    }
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
        let codex_session_titles =
            list_recent_codex_sessions(&connection, LOCAL_USER_TOKEN, None, 15)?
                .into_iter()
                .map(|session| format_codex_session_title(&session))
                .collect::<Vec<_>>();

        Ok(TuiSnapshot {
            conversation_lines,
            memory_titles,
            agenda_titles,
            codex_session_titles,
            status: Some("loaded persisted transcript and sidebar data".to_string()),
        })
    }

    pub fn submit_prompt(&self, prompt: &str) -> Result<PromptRunResult, AppError> {
        self.process_message(prompt, MessageProcessOptions::default())
    }

    pub fn process_message(
        &self,
        prompt: &str,
        options: MessageProcessOptions,
    ) -> Result<PromptRunResult, AppError> {
        let mut connection = self.open_connection()?;
        let preferences = load_user_preferences(&connection, LOCAL_USER_TOKEN)?;
        let model = live_provider_model(&self.config, preferences.as_ref())?;
        let executable_tools = if options.enable_tools {
            build_live_tool_registry(&self.config)
        } else {
            ExecutableToolRegistry::new(vec![])
        };
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            prompt,
            &model,
            executable_tools,
            options.persist_input_message,
        )?;

        Ok(PromptRunResult {
            events,
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
            SidebarSection::CodexSessions => {
                let thread_id = title
                    .rsplit(' ')
                    .next()
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| {
                        AppError::Runtime(format!("invalid codex session title: {title}"))
                    })?;
                let Some(session) =
                    get_codex_session_by_thread_id(&connection, LOCAL_USER_TOKEN, thread_id)?
                else {
                    return Err(AppError::Runtime(format!(
                        "codex session not found: {thread_id}"
                    )));
                };

                let mut lines = vec![
                    format!("codex_session: {}", session.thread_id),
                    format!("repo_path: {}", session.repo_path),
                    format!("status: {}", session.status),
                ];
                if let Some(worktree_path) = session.worktree_path {
                    lines.push(format!("worktree_path: {worktree_path}"));
                }
                if let Some(session_branch) = session.session_branch {
                    lines.push(format!("session_branch: {session_branch}"));
                }
                if let Some(target_branch) = session.target_branch {
                    lines.push(format!("target_branch: {target_branch}"));
                }
                if let Some(session_file_path) = session.session_file_path {
                    lines.push(format!("session_file_path: {session_file_path}"));
                }
                lines.push(String::new());
                lines.push(
                    session
                        .latest_summary
                        .unwrap_or_else(|| "(No summary recorded.)".to_string()),
                );
                if let Some(agent_message) = session.latest_agent_message {
                    lines.push(String::new());
                    lines.push(agent_message);
                }
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
            (SidebarSection::CodexSessions, _) => {
                return Err(AppError::Runtime(
                    "codex sessions are read-only in the sidebar".to_string(),
                ));
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

fn live_provider_model(
    config: &AppConfig,
    preferences: Option<&UserPreferenceRecord>,
) -> Result<LiveProviderModel, AppError> {
    let provider_config =
        provider_config_from_app_config(config).map_err(AppError::ProviderConfig)?;
    let client = LiveModelClient::new(provider_config)
        .map_err(|error| AppError::Runtime(error.to_string()))?;
    Ok(LiveProviderModel::new(
        client,
        effective_persona(preferences, &config.assistant_name),
    ))
}

fn run_prompt_with_model_and_registry(
    connection: &mut rusqlite::Connection,
    prompt: &str,
    model: &dyn ModelClient,
    executable_tools: ExecutableToolRegistry,
    persist_input_message: bool,
) -> Result<Vec<StreamEvent>, AppError> {
    let tools = ToolRegistry::new(executable_tools.specs());
    let orchestrator = ConversationOrchestrator::new(2);
    let tool_executor = LocalToolExecutor::new(executable_tools);
    let existing_transcript =
        validated_transcript(&load_context_messages(connection, LOCAL_USER_TOKEN)?);
    let recall_context = recall_memory_context_messages(
        prompt,
        &existing_transcript,
        &list_active_memories(connection, 50)?,
    );
    let due_item_context = due_item_context_messages(&list_active_due_items(connection, 20)?);
    let mut model_transcript = existing_transcript.clone();
    model_transcript.extend(recall_context.iter().cloned());
    model_transcript.extend(due_item_context.iter().cloned());

    let turn_run = orchestrator.run_turn_with_transcript(
        model,
        tools.specs(),
        &tool_executor,
        &model_transcript,
        prompt,
    )?;
    let persisted_transcript = strip_input_message_for_persistence(
        strip_transient_context_messages(
            turn_run.transcript.clone(),
            existing_transcript.len(),
            recall_context.len() + due_item_context.len(),
        ),
        existing_transcript.len(),
        persist_input_message,
    );
    replace_context_messages(connection, LOCAL_USER_TOKEN, &persisted_transcript)?;

    let mut events = Vec::new();
    if !recall_context.is_empty() {
        events.push(StreamEvent::StatusUpdate {
            content: "fetching memories...".to_string(),
        });
    }
    if !due_item_context.is_empty() {
        events.push(StreamEvent::StatusUpdate {
            content: "surfacing due items...".to_string(),
        });
    }
    events.extend(turn_run.events);
    Ok(events)
}

fn run_background_codex_completion_followup(
    config: &AppConfig,
    result: &CodexSessionResult,
) -> Result<(), AppError> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let preferences = load_user_preferences(&connection, LOCAL_USER_TOKEN)?;
    let model = live_provider_model(config, preferences.as_ref())?;
    let prompt = codex_completion_followup_prompt(result);
    run_prompt_with_model_and_registry(
        &mut connection,
        &prompt,
        &model,
        build_live_tool_registry(config),
        false,
    )?;
    Ok(())
}

fn codex_completion_followup_prompt(result: &CodexSessionResult) -> String {
    format!(
        "A background Codex session completed.\n\nSession: {}\nRepository: {}\nWorktree: {}\nSession branch: {}\nTarget branch: {}\nStatus: {}\nSummary:\n{}\n\nRespond to the user about the outcome and decide whether any follow-up action is needed.",
        result.session_id,
        result.repo_path,
        result.worktree_path.as_deref().unwrap_or("n/a"),
        result.session_branch.as_deref().unwrap_or("n/a"),
        result.target_branch.as_deref().unwrap_or("n/a"),
        result.status,
        result.summary,
    )
}

pub fn build_live_tool_registry(config: &AppConfig) -> ExecutableToolRegistry {
    build_live_tool_registry_with_codex_bin_and_hook(config, None, None)
}

fn build_live_tool_registry_with_codex_bin_and_hook(
    config: &AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook_override: Option<Arc<dyn Fn(CodexSessionResult) + Send + Sync>>,
) -> ExecutableToolRegistry {
    let config_for_codex_completion = config.clone();
    let codex_completion_hook = codex_completion_hook_override.unwrap_or_else(|| {
        Arc::new(move |result| {
            if let Err(error) =
                run_background_codex_completion_followup(&config_for_codex_completion, &result)
            {
                eprintln!(
                    "failed to run background codex completion follow-up for {}: {}",
                    result.session_id, error
                );
            }
        })
    });

    let config_for_memory_write = config.clone();
    let create_memory = ExecutableTool::new(
        ToolSpec::new(
            "create_memory",
            "Create a new file-backed memory and rebuild derived state.",
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

    let config_for_task_write = config.clone();
    let create_task = ExecutableTool::new(
        ToolSpec::new(
            "create_task",
            "Create a new agenda-backed task and rebuild derived state.",
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
                return ToolExecutionResult::error("create_task requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_task requires string text");
            };
            let date = arguments.get("date").and_then(Value::as_str);
            let trigger_datetime = arguments.get("trigger_datetime").and_then(Value::as_str);
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);
            match create_task_file_with_schedule(
                &config_for_task_write.agenda_dir,
                name,
                text,
                date,
                trigger_datetime,
                trigger_context,
            )
            .and_then(|path| {
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_task_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            }) {
                Ok(path) => ToolExecutionResult::success(
                    json!({"created": true, "file_path": path.display().to_string()}).to_string(),
                ),
                Err(error) => ToolExecutionResult::error(format!("failed to create task: {error}")),
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
            mutate_task_file_from_config(&config_for_task_text, name, |path| {
                update_task_text_file(path, text)
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

    let config_for_task_rename = config.clone();
    let rename_task = ExecutableTool::new(
        ToolSpec::new(
            "rename_task",
            "Rename one active task.",
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
                return ToolExecutionResult::error("rename_task requires a string name");
            };
            let Some(new_name) = arguments.get("new_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("rename_task requires string new_name");
            };
            mutate_task_file_from_config_with_result(&config_for_task_rename, name, |path| {
                let renamed = rename_task_file(path, new_name)?;
                Ok(
                    json!({"updated": true, "new_file_path": renamed.display().to_string()})
                        .to_string(),
                )
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
            mutate_task_file_from_config(&config_for_task_complete, name, |path| {
                complete_task_file(path, closing_comment)
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

    let config_for_task_delete = config.clone();
    let delete_task = ExecutableTool::new(
        ToolSpec::new(
            "delete_task",
            "Mark one active task deleted.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("delete_task requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            mutate_task_file_from_config(&config_for_task_delete, name, |path| {
                delete_task_file(path, closing_comment)
            })
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
    let set_assistant_name = ExecutableTool::new(
        ToolSpec::new(
            "set_assistant_name",
            "Set the assistant name for this local user.",
            JsonSchema::object(
                [("assistant_name", json!({"type": "string"}))],
                ["assistant_name"],
            ),
        ),
        move |arguments| {
            let Some(assistant_name) = arguments.get("assistant_name").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error(
                    "set_assistant_name requires string assistant_name",
                );
            };
            mutate_user_preferences_at_path(&database_path, |record| {
                record.assistant_name = Some(assistant_name.to_string());
                Ok(format!("Assistant name updated to {assistant_name}."))
            })
        },
    );

    let database_path = config.database_path.clone();
    let set_persona = ExecutableTool::new(
        ToolSpec::new(
            "set_persona",
            "Set the system persona template for this local user.",
            JsonSchema::object(
                [("system_persona", json!({"type": "string"}))],
                ["system_persona"],
            ),
        ),
        move |arguments| {
            let Some(system_persona) = arguments.get("system_persona").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error("set_persona requires string system_persona");
            };
            let system_persona = system_persona.trim();
            if system_persona.is_empty() {
                return ToolExecutionResult::error("System persona cannot be blank.");
            }
            mutate_user_preferences_at_path(&database_path, |record| {
                if record.system_persona.as_deref() == Some(system_persona) {
                    return Ok("New system persona and old system persona are identical".into());
                }
                record.system_persona = Some(system_persona.to_string());
                Ok("System persona updated.".into())
            })
        },
    );

    let database_path = config.database_path.clone();
    let reset_system_persona = ExecutableTool::new(
        ToolSpec::new(
            "reset_system_persona",
            "Clear the persisted system persona for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            mutate_user_preferences_at_path(&database_path, |record| {
                record.system_persona = None;
                Ok("System persona cleared, will now use default persona.".into())
            })
        },
    );

    let database_path = config.database_path.clone();
    let set_user_preferred_name = ExecutableTool::new(
        ToolSpec::new(
            "set_user_preferred_name",
            "Set the preferred name for this local user.",
            JsonSchema::object(
                [
                    ("preferred_name", json!({"type": "string"})),
                    ("override_existing", json!({"type": "boolean"})),
                ],
                ["preferred_name"],
            ),
        ),
        move |arguments| {
            let Some(preferred_name) = arguments.get("preferred_name").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error(
                    "set_user_preferred_name requires string preferred_name",
                );
            };
            let override_existing = arguments
                .get("override_existing")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            mutate_user_preferences_at_path(&database_path, |record| {
                let existing = effective_user_preferred_name(Some(record));
                if existing != elroy_user::DEFAULT_USER_PREFERRED_NAME && !override_existing {
                    return Ok(format!(
                        "Preferred name already set to {}. If this should be changed, use override_existing=true.",
                        existing
                    ));
                }
                record.preferred_name = Some(preferred_name.to_string());
                Ok(format!(
                    "Set user preferred name to {}. Was {}.",
                    preferred_name, existing
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let get_user_preferred_name = ExecutableTool::new(
        ToolSpec::new(
            "get_user_preferred_name",
            "Return the preferred name for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            with_user_preferences_at_path(&database_path, |record| {
                Ok(ToolExecutionResult::success(effective_user_preferred_name(
                    record.as_ref(),
                )))
            })
        },
    );

    let database_path = config.database_path.clone();
    let set_user_full_name = ExecutableTool::new(
        ToolSpec::new(
            "set_user_full_name",
            "Set the full name for this local user.",
            JsonSchema::object(
                [
                    ("full_name", json!({"type": "string"})),
                    ("override_existing", json!({"type": "boolean"})),
                ],
                ["full_name"],
            ),
        ),
        move |arguments| {
            let Some(full_name) = arguments.get("full_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("set_user_full_name requires string full_name");
            };
            let override_existing = arguments
                .get("override_existing")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            mutate_user_preferences_at_path(&database_path, |record| {
                let existing = effective_user_full_name(Some(record));
                if existing != elroy_user::UNKNOWN_FULL_NAME && !override_existing {
                    return Ok(format!(
                        "Full name already set to {}. If this should be changed, set override_existing=true.",
                        existing
                    ));
                }
                record.full_name = Some(full_name.to_string());
                Ok(format!(
                    "Full name set to {}. Previous value was {}.",
                    full_name, existing
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let get_user_full_name = ExecutableTool::new(
        ToolSpec::new(
            "get_user_full_name",
            "Return the full name for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            with_user_preferences_at_path(&database_path, |record| {
                Ok(ToolExecutionResult::success(effective_user_full_name(
                    record.as_ref(),
                )))
            })
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

            let result = if let Some(codex_bin) = codex_bin_for_dispatch.as_deref() {
                dispatch_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    prompt,
                    repo_path.map(Path::new),
                    model,
                    codex_bin,
                    Some(codex_completion_hook_for_dispatch.clone()),
                )
            } else {
                dispatch_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    prompt,
                    repo_path.map(Path::new),
                    model,
                    Some(codex_completion_hook_for_dispatch.clone()),
                )
            };
            match result {
                Ok(result) => ToolExecutionResult::success(codex_session_result_payload(result)),
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

            let result = if let Some(codex_bin) = codex_bin_for_resume.as_deref() {
                resume_codex_session_with_bin(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    session_id,
                    prompt,
                    model,
                    codex_bin,
                    Some(codex_completion_hook_for_resume.clone()),
                )
            } else {
                resume_codex_session_with_hook(
                    &database_path,
                    LOCAL_USER_TOKEN,
                    session_id,
                    prompt,
                    model,
                    Some(codex_completion_hook_for_resume.clone()),
                )
            };
            match result {
                Ok(result) => ToolExecutionResult::success(codex_session_result_payload(result)),
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

    let database_path = config.database_path.clone();
    let list_tasks = ExecutableTool::new(
        ToolSpec::new(
            "list_tasks",
            "List active agenda-backed tasks.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
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
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
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
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
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
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
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
                        "task not found: {name}"
                    )));
                };
                Ok(ToolExecutionResult::success(task_payload(item).to_string()))
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
        create_task,
        create_due_item,
        set_assistant_name,
        set_persona,
        reset_system_persona,
        set_user_preferred_name,
        get_user_preferred_name,
        set_user_full_name,
        get_user_full_name,
        dispatch_codex_session,
        resume_codex_session,
        list_codex_sessions,
        show_codex_session,
        update_task_text,
        update_due_item_text,
        rename_task,
        rename_due_item,
        complete_task,
        complete_due_item,
        delete_task,
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
        list_tasks,
        list_triggered_tasks_tool,
        list_due_tasks_tool,
        list_today_tasks_tool,
        list_memories,
        search_memories,
        list_agenda,
        list_due_items,
        show_task,
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

fn with_user_preferences_at_path(
    database_path: &Path,
    operation: impl FnOnce(Option<UserPreferenceRecord>) -> rusqlite::Result<ToolExecutionResult>,
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
    match load_user_preferences(&connection, LOCAL_USER_TOKEN).and_then(operation) {
        Ok(result) => result,
        Err(error) => ToolExecutionResult::error(format!("database query failed: {error}")),
    }
}

fn mutate_user_preferences_at_path(
    database_path: &Path,
    operation: impl FnOnce(&mut UserPreferenceRecord) -> rusqlite::Result<String>,
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
    let mut record = load_user_preferences(&connection, LOCAL_USER_TOKEN)
        .ok()
        .flatten()
        .unwrap_or(UserPreferenceRecord {
            user_token: LOCAL_USER_TOKEN.to_string(),
            assistant_name: None,
            preferred_name: None,
            full_name: None,
            system_persona: None,
            created_at_unix: 0,
            updated_at_unix: 0,
        });

    match operation(&mut record).and_then(|message| {
        save_user_preferences(&mut connection, &record)?;
        Ok(message)
    }) {
        Ok(message) => ToolExecutionResult::success(message),
        Err(error) => ToolExecutionResult::error(format!("user preference update failed: {error}")),
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

fn mutate_task_file_from_config(
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
    let task = match find_task_by_name(&connection, name) {
        Ok(Some(task)) => task,
        Ok(None) => return ToolExecutionResult::error(format!("task not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&task.file_path)).and_then(|()| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(())
    }) {
        Ok(()) => ToolExecutionResult::success(
            json!({"updated": true, "name": task.name, "file_path": task.file_path}).to_string(),
        ),
        Err(error) => ToolExecutionResult::error(format!("task mutation failed: {error}")),
    }
}

fn mutate_task_file_from_config_with_result(
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
    let task = match find_task_by_name(&connection, name) {
        Ok(Some(task)) => task,
        Ok(None) => return ToolExecutionResult::error(format!("task not found: {name}")),
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    match operation(Path::new(&task.file_path)).and_then(|payload| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(payload)
    }) {
        Ok(payload) => ToolExecutionResult::success(payload),
        Err(error) => ToolExecutionResult::error(format!("task mutation failed: {error}")),
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

fn format_codex_session_title(session: &elroy_codex::CodexSessionRecord) -> String {
    let repo_name = Path::new(&session.repo_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(&session.repo_path);
    format!("{repo_name} ({}) {}", session.status, session.thread_id)
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

fn due_item_context_messages(items: &[AgendaItemRecord]) -> Vec<ConversationMessage> {
    if items.is_empty() {
        return Vec::new();
    }

    let payload = items
        .iter()
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
    let arguments_json = serde_json::to_string(&json!({ "limit": items.len() }))
        .expect("due item args should serialize");
    let content =
        serde_json::to_string_pretty(&payload).expect("due item payload should serialize");

    vec![
        ConversationMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "bootstrap-due-items".to_string(),
                name: "list_due_items".to_string(),
                arguments_json,
            }],
        ),
        ConversationMessage::tool_result("bootstrap-due-items", content),
    ]
}

fn recall_memory_context_messages(
    prompt: &str,
    transcript: &[ConversationMessage],
    memories: &[elroy_db::MemoryRecord],
) -> Vec<ConversationMessage> {
    if should_skip_memory_recall(prompt) {
        return Vec::new();
    }

    let recall_query = build_recall_query(prompt, transcript, 4);
    let already_recalled = recalled_memory_names(transcript);
    let recalled = select_recalled_memories(&recall_query, memories, &already_recalled, 3);
    if recalled.is_empty() {
        return Vec::new();
    }

    let payload = recalled
        .iter()
        .map(|memory| {
            json!({
                "name": memory.name,
                "file_path": memory.file_path,
                "excerpt": excerpt(&memory.body, 180),
                "updated_at_unix": memory.updated_at_unix,
            })
        })
        .collect::<Vec<_>>();
    let content =
        serde_json::to_string_pretty(&payload).expect("memory recall payload should serialize");

    vec![
        ConversationMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "bootstrap-memory-recall".to_string(),
                name: "search_memories".to_string(),
                arguments_json: json!({
                    "query": recall_query,
                    "limit": recalled.len(),
                })
                .to_string(),
            }],
        ),
        ConversationMessage::tool_result("bootstrap-memory-recall", content),
    ]
}

fn should_skip_memory_recall(prompt: &str) -> bool {
    let normalized = prompt.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return true;
    }

    const SIMPLE_SHORT: &[&str] = &[
        "ok",
        "okay",
        "yes",
        "no",
        "thanks",
        "thank you",
        "sure",
        "got it",
        "k",
        "yep",
        "nope",
    ];
    const GREETINGS: &[&str] = &[
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "goodbye",
        "bye",
    ];

    (normalized.len() < 10 && SIMPLE_SHORT.contains(&normalized.as_str()))
        || GREETINGS.contains(&normalized.as_str())
}

fn build_recall_query(prompt: &str, transcript: &[ConversationMessage], window: usize) -> String {
    let mut parts = recent_recall_context(transcript, window);
    parts.push(prompt.trim().to_string());
    parts.retain(|part| !part.is_empty());
    parts.join("\n")
}

fn recent_recall_context(transcript: &[ConversationMessage], window: usize) -> Vec<String> {
    transcript
        .iter()
        .rev()
        .filter(|message| {
            matches!(message.role, MessageRole::User | MessageRole::Assistant)
                && !message
                    .content
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty()
        })
        .take(window)
        .filter_map(|message| message.content.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn recalled_memory_names(transcript: &[ConversationMessage]) -> HashSet<String> {
    transcript
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .filter_map(|message| {
            (message.tool_call_id.as_deref() == Some("bootstrap-memory-recall"))
                .then_some(message.content.as_deref())
                .flatten()
        })
        .flat_map(parse_recalled_memory_names)
        .collect()
}

fn parse_recalled_memory_names(content: &str) -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            item.get("name")
                .and_then(serde_json::Value::as_str)
                .map(|name| name.to_ascii_lowercase())
        })
        .collect()
}

fn select_recalled_memories<'a>(
    prompt: &str,
    memories: &'a [elroy_db::MemoryRecord],
    already_recalled: &HashSet<String>,
    limit: usize,
) -> Vec<&'a elroy_db::MemoryRecord> {
    let prompt_tokens = significant_tokens(prompt);
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored = memories
        .iter()
        .filter_map(|memory| {
            if already_recalled.contains(&memory.name.to_ascii_lowercase()) {
                return None;
            }
            let mut haystack = String::with_capacity(memory.name.len() + memory.body.len() + 1);
            haystack.push_str(&memory.name);
            haystack.push(' ');
            haystack.push_str(&memory.body);
            let memory_tokens = significant_tokens(&haystack);
            let overlap = prompt_tokens.intersection(&memory_tokens).count();
            (overlap > 0).then_some((overlap, memory.updated_at_unix, memory))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
            .then_with(|| left.2.name.cmp(&right.2.name))
    });
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, memory)| memory)
        .collect()
}

fn significant_tokens(text: &str) -> HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "at", "be", "but", "by", "for", "from", "i", "if", "im", "in",
        "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the", "this", "to", "was",
        "we", "with", "you",
    ];

    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let normalized = token.trim().to_ascii_lowercase();
            if normalized.len() < 3 || STOPWORDS.contains(&normalized.as_str()) {
                None
            } else {
                Some(normalized)
            }
        })
        .collect()
}

fn strip_transient_context_messages(
    mut transcript: Vec<ConversationMessage>,
    persistent_prefix_len: usize,
    transient_len: usize,
) -> Vec<ConversationMessage> {
    if transient_len == 0 {
        return transcript;
    }
    transcript.drain(persistent_prefix_len..persistent_prefix_len + transient_len);
    transcript
}

fn strip_input_message_for_persistence(
    mut transcript: Vec<ConversationMessage>,
    persistent_prefix_len: usize,
    persist_input_message: bool,
) -> Vec<ConversationMessage> {
    if persist_input_message {
        return transcript;
    }
    if transcript
        .get(persistent_prefix_len)
        .is_some_and(|message| message.role == MessageRole::User)
    {
        transcript.remove(persistent_prefix_len);
    }
    transcript
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
    use std::{
        cell::RefCell,
        collections::HashSet,
        fs,
        path::{Path, PathBuf},
        process::Command,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use elroy_agenda::create_agenda_file;
    use elroy_codex::{
        CodexCommandRecord, CodexSessionResult, CodexSessionUpdate, upsert_codex_session,
    };
    use elroy_config::{AppConfig, LlmProvider};
    use elroy_core::{ConversationRequest, ModelClient};
    use elroy_db::{
        AgendaItemRecord, MemoryRecord, load_user_preferences, open_sqlite_connection,
        run_migrations,
    };
    use elroy_llm::{ConversationMessage, MessageRole, Provider, StreamEvent};
    use elroy_memory::{create_memory_file, sanitize_filename};
    use elroy_tools::ExecutableToolRegistry;

    use super::{
        AppRuntime, LOCAL_USER_TOKEN, argument_limit, build_live_tool_registry,
        build_live_tool_registry_with_codex_bin_and_hook, build_recall_query,
        due_item_context_messages, parse_recalled_memory_names, provider_config_from_app_config,
        recall_memory_context_messages, recalled_memory_names, recent_recall_context,
        run_prompt_with_model_and_registry, select_recalled_memories, should_skip_memory_recall,
        significant_tokens, strip_input_message_for_persistence, strip_transient_context_messages,
    };

    struct FakeModel {
        responses: RefCell<Vec<Vec<StreamEvent>>>,
    }

    impl FakeModel {
        fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
            Self {
                responses: RefCell::new(responses),
            }
        }
    }

    impl ModelClient for FakeModel {
        fn next_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            Ok(self.responses.borrow_mut().remove(0))
        }
    }

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
        upsert_codex_session(
            &mut connection,
            LOCAL_USER_TOKEN,
            "thread-123",
            &CodexSessionUpdate {
                repo_path: PathBuf::from("/tmp/sample"),
                worktree_path: Some(PathBuf::from("/tmp/.elroy-codex-worktrees/sample")),
                session_branch: Some("elroy-codex-abcd1234".to_string()),
                target_branch: Some("agent".to_string()),
                prompt: "Inspect the parser".to_string(),
                summary: "Codex inspected the parser state.".to_string(),
                agent_message: "Parser inspection complete.".to_string(),
                status: "completed".to_string(),
                commands: vec![],
                touched_paths: vec!["src/parser.rs".to_string()],
                dirty_paths_before: vec![],
                dirty_paths_after: vec![],
                session_file_path: None,
            },
        )
        .expect("codex session should persist");

        let runtime = AppRuntime::new(config);
        let snapshot = runtime.load_snapshot().expect("snapshot should load");
        let memory_detail = runtime
            .open_sidebar_item(elroy_tui::SidebarSection::Memories, "runner notes")
            .expect("memory detail should open");
        let codex_detail = runtime
            .open_sidebar_item(
                elroy_tui::SidebarSection::CodexSessions,
                "sample (completed) thread-123",
            )
            .expect("codex detail should open");

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
        assert!(
            snapshot
                .codex_session_titles
                .iter()
                .any(|item| item == "sample (completed) thread-123")
        );
        assert!(memory_detail.contains("remember the hill workout"));
        assert!(codex_detail.contains("Codex inspected the parser state."));
        assert!(codex_detail.contains("Parser inspection complete."));

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
    fn live_tool_registry_can_manage_user_preferences() {
        let unique = format!(
            "elroy-rs-app-user-preferences-{}",
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
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;

        let registry = build_live_tool_registry(&config);
        let default_name = registry.invoke("get_user_preferred_name", "{}");
        assert!(!default_name.is_error);
        assert_eq!(default_name.content, "User");

        let preferred =
            registry.invoke("set_user_preferred_name", "{\"preferred_name\":\"Jimmy\"}");
        assert!(!preferred.is_error);
        assert!(preferred.content.contains("Jimmy"));

        let duplicate =
            registry.invoke("set_user_preferred_name", "{\"preferred_name\":\"James\"}");
        assert!(!duplicate.is_error);
        assert!(duplicate.content.contains("already set"));

        let assistant = registry.invoke("set_assistant_name", "{\"assistant_name\":\"Nova\"}");
        assert!(!assistant.is_error);
        assert!(assistant.content.contains("Nova"));

        let full_name = registry.invoke("set_user_full_name", "{\"full_name\":\"James Smith\"}");
        assert!(!full_name.is_error);

        let get_full_name = registry.invoke("get_user_full_name", "{}");
        assert!(!get_full_name.is_error);
        assert_eq!(get_full_name.content, "James Smith");

        let persona = registry.invoke(
            "set_persona",
            "{\"system_persona\":\"You are $ASSISTANT_ALIAS helping $USER_ALIAS.\"}",
        );
        assert!(!persona.is_error);
        assert_eq!(persona.content, "System persona updated.");

        let connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        let persisted = load_user_preferences(&connection, LOCAL_USER_TOKEN)
            .expect("preferences should load")
            .expect("preferences should exist");
        assert_eq!(
            persisted.system_persona.as_deref(),
            Some("You are $ASSISTANT_ALIAS helping $USER_ALIAS.")
        );

        let reset = registry.invoke("reset_system_persona", "{}");
        assert!(!reset.is_error);
        assert_eq!(
            reset.content,
            "System persona cleared, will now use default persona."
        );

        let cleared = load_user_preferences(&connection, LOCAL_USER_TOKEN)
            .expect("preferences should load")
            .expect("preferences should exist");
        assert_eq!(cleared.system_persona, None);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_manage_tasks() {
        let unique = format!(
            "elroy-rs-app-tasks-{}",
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
            "create_task",
            "{\"name\":\"Job Search\",\"text\":\"Reach out to three contacts\",\"date\":\"2026-05-20\",\"trigger_context\":\"after breakfast\"}",
        );
        assert!(!created.is_error);
        assert!(agenda_dir.join("job_search.md").exists());

        let triggered = registry.invoke("list_triggered_tasks", "{\"limit\":10}");
        assert!(!triggered.is_error);
        assert!(triggered.content.contains("job search"));
        assert!(triggered.content.contains("after breakfast"));

        let today = registry.invoke("list_today_tasks", "{\"limit\":10}");
        assert!(!today.is_error);
        assert!(!today.content.contains("job search"));

        let updated = registry.invoke(
            "update_task_text",
            "{\"name\":\"job search\",\"text\":\"Reach out to four contacts\"}",
        );
        assert!(!updated.is_error);

        let renamed = registry.invoke(
            "rename_task",
            "{\"name\":\"job search\",\"new_name\":\"Career Search\"}",
        );
        assert!(!renamed.is_error);
        assert!(agenda_dir.join("career_search.md").exists());

        let listed = registry.invoke("list_tasks", "{\"limit\":10}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("career search"));

        let shown = registry.invoke("show_task", "{\"name\":\"career search\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("Reach out to four contacts"));

        let completed = registry.invoke("complete_task", "{\"name\":\"career search\"}");
        assert!(!completed.is_error);

        let deleted_created = registry.invoke(
            "create_task",
            "{\"name\":\"Inbox Zero\",\"text\":\"Clear email backlog\"}",
        );
        assert!(!deleted_created.is_error);
        let deleted = registry.invoke("delete_task", "{\"name\":\"inbox zero\"}");
        assert!(!deleted.is_error);

        let deleted_text =
            fs::read_to_string(agenda_dir.join("inbox_zero.md")).expect("task file should read");
        assert!(deleted_text.contains("status: deleted"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_list_and_show_codex_sessions() {
        let unique = format!(
            "elroy-rs-app-codex-sessions-{}",
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

        let registry = build_live_tool_registry(&config);
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

        let shown = registry.invoke("show_codex_session", "{\"session_id\":\"thread-123\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("Fix the parser"));
        assert!(shown.content.contains("cargo test"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_dispatch_and_resume_codex_sessions() {
        let unique = format!(
            "elroy-rs-app-codex-dispatch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let repo_root = root.join("development").join("sample");
        let bin_dir = root.join("bin");
        let memory_dir = root.join("memories");
        let agenda_dir = root.join("agenda");
        let database_path = root.join("elroy.db");
        fs::create_dir_all(&bin_dir).expect("bin dir should be created");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        init_test_repo(&repo_root);
        write_fake_codex_script(&bin_dir.join("codex"));

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let followup_database_path = database_path.clone();
        let followup_hook = Arc::new(move |result: CodexSessionResult| {
            let mut connection =
                open_sqlite_connection(&followup_database_path).expect("database should open");
            run_migrations(&mut connection).expect("migrations should run");
            let mut transcript = elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN)
                .expect("messages should load");
            transcript.push(ConversationMessage::new(
                MessageRole::Assistant,
                format!("Background hook for {}", result.session_id),
            ));
            elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
                .expect("messages should persist");
        });

        let registry = build_live_tool_registry_with_codex_bin_and_hook(
            &config,
            Some(bin_dir.join("codex")),
            Some(followup_hook),
        );
        let dispatched = registry.invoke(
            "dispatch_codex_session",
            &format!(
                "{{\"prompt\":\"update notes\",\"repo_path\":\"{}\"}}",
                repo_root.display()
            ),
        );
        assert!(!dispatched.is_error);
        assert!(dispatched.content.contains("\"status\":\"running\""));
        wait_for_codex_status(&database_path, "thread-123", "completed");

        let resumed = registry.invoke(
            "resume_codex_session",
            "{\"session_id\":\"thread-123\",\"prompt\":\"follow up\"}",
        );
        assert!(!resumed.is_error);
        assert!(resumed.content.contains("\"status\":\"running\""));
        wait_for_codex_status(&database_path, "thread-123", "completed");

        let shown = registry.invoke("show_codex_session", "{\"session_id\":\"thread-123\"}");
        assert!(!shown.is_error);
        assert!(shown.content.contains("resume complete"));
        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        let transcript =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert!(transcript.iter().any(|message| {
            message.role == MessageRole::Assistant
                && message.content.as_deref() == Some("Background hook for thread-123")
        }));

        let agent_head = Command::new("git")
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

    #[test]
    fn due_item_context_messages_create_synthetic_tool_context() {
        let messages = due_item_context_messages(&[AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "call mom".to_string(),
            file_path: "/tmp/call_mom.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: None,
            trigger_context: Some("after dinner".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Call mom tonight".to_string(),
            is_active: true,
            updated_at_unix: 1,
        }]);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(
            messages[0]
                .tool_calls
                .as_ref()
                .map(|calls| calls[0].name.as_str()),
            Some("list_due_items")
        );
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert!(
            messages[1]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("after dinner"))
        );
    }

    #[test]
    fn strip_transient_context_messages_removes_injected_due_item_context() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::Assistant, "earlier reply"),
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![elroy_llm::ToolCall {
                    id: "bootstrap-due-items".to_string(),
                    name: "list_due_items".to_string(),
                    arguments_json: "{\"limit\":1}".to_string(),
                }],
            ),
            ConversationMessage::tool_result("bootstrap-due-items", "[]"),
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::new(MessageRole::Assistant, "hi"),
        ];

        let stripped = strip_transient_context_messages(transcript, 1, 2);

        assert_eq!(stripped.len(), 3);
        assert_eq!(stripped[0].content.as_deref(), Some("earlier reply"));
        assert_eq!(stripped[1].role, MessageRole::User);
        assert_eq!(stripped[2].role, MessageRole::Assistant);
    }

    #[test]
    fn strip_input_message_for_persistence_can_drop_new_user_message() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::Assistant, "earlier reply"),
            ConversationMessage::new(MessageRole::User, "background follow-up"),
            ConversationMessage::new(MessageRole::Assistant, "all set"),
        ];

        let stripped = strip_input_message_for_persistence(transcript, 1, false);

        assert_eq!(stripped.len(), 2);
        assert_eq!(stripped[0].role, MessageRole::Assistant);
        assert_eq!(stripped[1].role, MessageRole::Assistant);
        assert_eq!(stripped[1].content.as_deref(), Some("all set"));
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_skip_persisting_input_message() {
        let unique = format!(
            "elroy-rs-app-background-followup-{}",
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
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "existing transcript",
            )],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Background review complete.".to_string(),
        }]]);
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "A background Codex session completed.",
            &model,
            ExecutableToolRegistry::new(vec![]),
            false,
        )
        .expect("background follow-up should succeed");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content } if content == "Background review complete."
        )));
        assert!(!stored.iter().any(|message| {
            message.role == MessageRole::User
                && message.content.as_deref() == Some("A background Codex session completed.")
        }));
        assert!(stored.iter().any(|message| {
            message.role == MessageRole::Assistant
                && message.content.as_deref() == Some("Background review complete.")
        }));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn trivial_messages_skip_memory_recall() {
        assert!(should_skip_memory_recall("hi"));
        assert!(should_skip_memory_recall("thanks"));
        assert!(!should_skip_memory_recall("I am going running tomorrow"));
    }

    #[test]
    fn significant_tokens_drop_short_words_and_stopwords() {
        let tokens = significant_tokens("I'm going to play basketball at the park");
        assert!(tokens.contains("going"));
        assert!(tokens.contains("play"));
        assert!(tokens.contains("basketball"));
        assert!(!tokens.contains("the"));
        assert!(!tokens.contains("to"));
    }

    #[test]
    fn select_recalled_memories_prefers_overlap() {
        let memories = vec![
            MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "basketball form".to_string(),
                file_path: "/tmp/basketball.md".to_string(),
                body: "Remember to follow through on your shot".to_string(),
                is_active: true,
                updated_at_unix: 10,
            },
            MemoryRecord {
                id: 2,
                legacy_frontmatter_id: None,
                name: "grocery list".to_string(),
                file_path: "/tmp/grocery.md".to_string(),
                body: "Buy apples and milk".to_string(),
                is_active: true,
                updated_at_unix: 20,
            },
        ];

        let recalled = select_recalled_memories(
            "I am heading to basketball practice",
            &memories,
            &HashSet::new(),
            3,
        );

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].name, "basketball form");
    }

    #[test]
    fn recall_memory_context_messages_create_synthetic_tool_context() {
        let messages = recall_memory_context_messages(
            "I am heading to basketball practice",
            &[],
            &[MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "basketball form".to_string(),
                file_path: "/tmp/basketball.md".to_string(),
                body: "Remember to follow through on your shot".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(
            messages[0]
                .tool_calls
                .as_ref()
                .map(|calls| calls[0].name.as_str()),
            Some("search_memories")
        );
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert!(
            messages[1]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("basketball form"))
        );
    }

    #[test]
    fn recent_recall_context_uses_recent_user_and_assistant_messages() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::System, "system"),
            ConversationMessage::new(MessageRole::User, "I am training for basketball"),
            ConversationMessage::new(MessageRole::Assistant, "How is practice going?"),
            ConversationMessage::tool_result("bootstrap-memory-recall", "[]"),
            ConversationMessage::new(MessageRole::User, "My jump shot is inconsistent"),
        ];

        let context = recent_recall_context(&transcript, 3);

        assert_eq!(context.len(), 3);
        assert_eq!(context[0], "I am training for basketball");
        assert_eq!(context[2], "My jump shot is inconsistent");
    }

    #[test]
    fn build_recall_query_includes_recent_context_and_prompt() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::User, "I am training for basketball"),
            ConversationMessage::new(MessageRole::Assistant, "How is practice going?"),
        ];

        let query = build_recall_query("What should I focus on?", &transcript, 4);

        assert!(query.contains("I am training for basketball"));
        assert!(query.contains("What should I focus on?"));
    }

    #[test]
    fn parse_and_collect_recalled_memory_names_from_transcript() {
        let transcript = vec![ConversationMessage::tool_result(
            "bootstrap-memory-recall",
            r#"[{"name":"Basketball Form"},{"name":"Sleep Routine"}]"#,
        )];

        let parsed =
            parse_recalled_memory_names(r#"[{"name":"Basketball Form"},{"name":"Sleep Routine"}]"#);
        let names = recalled_memory_names(&transcript);

        assert_eq!(parsed.len(), 2);
        assert!(names.contains("basketball form"));
        assert!(names.contains("sleep routine"));
    }

    #[test]
    fn select_recalled_memories_skips_already_recalled_names() {
        let memories = vec![
            MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "basketball form".to_string(),
                file_path: "/tmp/basketball.md".to_string(),
                body: "Remember to follow through on your shot".to_string(),
                is_active: true,
                updated_at_unix: 10,
            },
            MemoryRecord {
                id: 2,
                legacy_frontmatter_id: None,
                name: "practice plan".to_string(),
                file_path: "/tmp/practice.md".to_string(),
                body: "Warm up before basketball drills".to_string(),
                is_active: true,
                updated_at_unix: 9,
            },
        ];
        let already_recalled = HashSet::from([String::from("basketball form")]);

        let recalled = select_recalled_memories(
            "I am heading to basketball practice",
            &memories,
            &already_recalled,
            3,
        );

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].name, "practice plan");
    }

    fn init_test_repo(repo_root: &Path) {
        fs::create_dir_all(repo_root).expect("repo root should exist");
        git(repo_root, ["init"]);
        git(repo_root, ["config", "user.email", "test@example.com"]);
        git(repo_root, ["config", "user.name", "Test User"]);
        fs::write(repo_root.join("notes.txt"), "before\n").expect("notes should be written");
        git(repo_root, ["add", "notes.txt"]);
        git(repo_root, ["commit", "-m", "init"]);
    }

    fn git<const N: usize>(repo_root: &Path, args: [&str; N]) {
        let output = Command::new("git")
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
session_root="${ELROY_CODEX_SESSION_SEARCH_ROOT:-}"
for arg in "$@"; do
  if [ "$arg" = "resume" ]; then
    mode="resume"
  fi
  prompt="$arg"
done

write_session_file() {
  if [ -n "$session_root" ]; then
    mkdir -p "$session_root/nested"
    printf '{"thread_id":"thread-123"}\n' > "$session_root/nested/thread-123.jsonl"
  fi
}

if [ "$mode" = "resume" ]; then
  printf "after resume\n" > notes.txt
  write_session_file
  echo '{"type":"thread.started","thread_id":"thread-123"}'
  echo '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"resume complete"}}'
  exit 0
fi

printf "after\n" > notes.txt
pwd_out="$(pwd)"
write_session_file
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

    fn wait_for_codex_status(database_path: &Path, session_id: &str, expected_status: &str) {
        let started = Instant::now();
        loop {
            let connection = open_sqlite_connection(database_path).expect("database should open");
            let record = elroy_codex::get_codex_session_by_thread_id(
                &connection,
                LOCAL_USER_TOKEN,
                session_id,
            )
            .expect("session should query")
            .expect("session should exist");
            if record.status == expected_status {
                break;
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
