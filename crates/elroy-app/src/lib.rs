use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, TimeZone, Utc};
use elroy_agenda::{
    add_checklist_item, append_agenda_update, create_agenda_file, mark_agenda_item_completed,
    rename_agenda_file, update_agenda_body, update_checklist_item,
};
use elroy_codex::{
    CodexSessionResult, dispatch_codex_session_with_bin, dispatch_codex_session_with_hook,
    get_codex_session_by_thread_id, list_recent_codex_sessions, resume_codex_session_with_bin,
    resume_codex_session_with_hook,
};
use elroy_config::{AppConfig, LlmProvider};
use elroy_core::{
    ConversationOrchestrator, ConversationRequest, LiveProviderModel, LocalToolExecutor,
    ModelClient, StreamingModelClient, TurnEventStream, clear_background_status,
    get_background_status, set_background_status, validated_transcript,
};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, MemoryRecord, UserPreferenceRecord,
    find_active_agenda_item_by_name, get_or_create_memory_operation_tracker, list_active_due_items,
    list_active_plain_agenda_items, list_inactive_due_items, load_context_messages,
    load_messages_by_ids, load_user_preferences, open_sqlite_connection,
    record_deleted_due_item_tombstone, replace_context_messages, run_migrations,
    save_memory_operation_tracker, save_user_preferences, search_active_memories,
};
use elroy_feature_requests::{
    FeatureRequestRecord, find_best_feature_request_match, get_feature_request,
    is_active_feature_request, list_feature_requests, list_self_reflection_feature_requests,
    update_feature_request, write_new_feature_request,
};
use elroy_llm::{
    ConversationMessage, LiveModelClient, MessageRole, Provider, ProviderConfig, StreamEvent,
    ToolCall,
};
use elroy_memory::{
    archive_memory_file, create_memory_file_with_frontmatter, read_memory_parts, sanitize_filename,
    update_memory_body,
};
use elroy_self_reflection::{SelfReflectionConfig, SelfReflectionOrchestrator};
use elroy_tasks::{
    complete_task_file, create_task_file_with_schedule, delete_task_file, find_task_by_name,
    list_active_tasks, list_due_tasks, list_today_tasks, list_triggered_tasks, rename_task_file,
    update_task_text_file,
};
use elroy_tools::{
    ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry, ToolSpec,
};
use elroy_tui::{
    SidebarAction, SidebarSection, TuiCommandExecution, TuiCommandForm, TuiCommandPaletteAction,
    TuiCommandPaletteEntry, TuiCommandParameter, TuiSidebarDetail, TuiSlashCommandAction,
    TuiSnapshot,
};
use elroy_user::{effective_persona, effective_user_full_name, effective_user_preferred_name};
use serde_json::{Map, Value, json};

const LOCAL_USER_TOKEN: &str = "local-user";
const SYNTHETIC_FIRST_USER_MESSAGE: &str = "The user has begun the conversation";
const DEFAULT_MAX_LIST_ENTRIES: usize = 50;
const DEFAULT_MAX_LIST_DEPTH: usize = 2;
const DEFAULT_READ_LINE_LIMIT: usize = 200;
const CONTEXT_MESSAGE_SOURCE_TYPE: &str = "ContextMessageSet";
const MEMORY_SOURCE_TYPE: &str = "Memory";
const DEFAULT_RESTART_RESUME_PROMPT: &str =
    "Elroy just restarted. Send a brief message that you are back and ready to continue.";

static RESTART_STATE: OnceLock<Mutex<RestartState>> = OnceLock::new();

#[derive(Debug, Default)]
struct RestartState {
    supported: bool,
    pending_resume_prompt: Option<String>,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredAutoMemoryWork {
    pub existing_transcript_len: usize,
    pub transcript: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCompletion {
    pub snapshot: TuiSnapshot,
    pub deferred_auto_memory: Option<DeferredAutoMemoryWork>,
}

pub struct PromptEventStream {
    state: Option<PromptEventStreamState>,
    finalized_completion: Option<Result<PromptCompletion, AppError>>,
}

impl PromptEventStream {
    pub fn snapshot(&self) -> Option<&TuiSnapshot> {
        self.finalized_completion
            .as_ref()
            .and_then(|result| result.as_ref().ok())
            .map(|completion| &completion.snapshot)
    }

    pub fn into_completion(mut self) -> Result<PromptCompletion, AppError> {
        while self.next().is_some() {}
        self.finalized_completion
            .take()
            .unwrap_or_else(|| Err(AppError::Runtime("stream did not finalize".to_string())))
    }

    pub fn into_snapshot(self) -> Result<TuiSnapshot, AppError> {
        self.into_completion().map(|completion| completion.snapshot)
    }

    pub fn cancel(mut self) -> Result<TuiSnapshot, AppError> {
        if let Some(result) = self.finalized_completion.take() {
            return result.map(|completion| completion.snapshot);
        }
        let Some(state) = self.state.take() else {
            return Err(AppError::Runtime("stream did not finalize".to_string()));
        };
        cancel_prompt_event_stream(state)
    }
}

impl Iterator for PromptEventStream {
    type Item = StreamEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.state.as_mut()?;
        if let Some(event) = state.prelude_events.pop_front() {
            return Some(event);
        }

        match state.turn_stream.next() {
            Some(Ok(event)) => Some(event),
            Some(Err(error)) => {
                self.finalized_completion = Some(Err(AppError::from(error)));
                self.state = None;
                None
            }
            None => {
                let state = self.state.take().expect("stream state should exist");
                self.finalized_completion = Some(finalize_prompt_event_stream(state));
                None
            }
        }
    }
}

struct PromptEventStreamState {
    home_dir: PathBuf,
    bootstrap_plan: BootstrapPlan,
    connection: rusqlite::Connection,
    turn_stream: TurnEventStream,
    existing_transcript_len: usize,
    transient_context_count: usize,
    persist_input_message: bool,
    messages_between_memory: usize,
    memories_between_consolidation: usize,
    messages_between_self_reflection: usize,
    defer_auto_memory: bool,
    defer_self_reflection: bool,
    prelude_events: VecDeque<StreamEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageProcessOptions {
    pub role: MessageRole,
    pub enable_tools: bool,
    pub persist_input_message: bool,
    pub force_tool: Option<String>,
    pub defer_auto_memory: bool,
    pub defer_self_reflection: bool,
}

impl Default for MessageProcessOptions {
    fn default() -> Self {
        Self {
            role: MessageRole::User,
            enable_tools: true,
            persist_input_message: true,
            force_tool: None,
            defer_auto_memory: false,
            defer_self_reflection: false,
        }
    }
}

#[derive(Debug, Clone)]
struct PromptExecutionOptions<'a> {
    role: MessageRole,
    persist_input_message: bool,
    force_tool: Option<&'a str>,
    assistant_name: &'a str,
    ensure_alternating_roles: bool,
    home_dir: &'a Path,
    bootstrap_plan: BootstrapPlan,
    messages_between_memory: usize,
    memories_between_consolidation: usize,
    messages_between_self_reflection: usize,
    defer_auto_memory: bool,
    defer_self_reflection: bool,
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
    reflect: bool,
}

#[derive(Debug, Clone)]
pub struct AppRuntime {
    config: AppConfig,
}

impl AppRuntime {
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }

    pub fn enable_restart_support(&self) {
        enable_session_restart_support();
    }

    pub fn disable_restart_support(&self) {
        disable_session_restart_support();
    }

    pub fn consume_restart_request(&self) -> Option<String> {
        consume_session_restart_request()
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn load_snapshot(&self) -> Result<TuiSnapshot, AppError> {
        let mut connection = self.open_connection()?;
        let mut snapshot = load_snapshot_from_connection(
            &mut connection,
            &self.config.home_dir,
            &self.config.memory_dir,
        )?;
        let mut slash_command_names = build_live_tool_registry(&self.config)
            .specs()
            .into_iter()
            .map(|spec| format!("/{}", spec.name))
            .collect::<Vec<_>>();
        slash_command_names.push("/help".to_string());
        slash_command_names.sort();
        slash_command_names.dedup();
        snapshot.input_completions.extend(slash_command_names);
        snapshot.model_name = Some(self.config.chat_model.clone());
        Ok(snapshot)
    }

    pub fn load_context_messages(&self) -> Result<Vec<ConversationMessage>, AppError> {
        let mut connection = self.open_connection()?;
        load_validated_runtime_transcript(
            &mut connection,
            &self.config.assistant_name,
            self.config.llm_provider() == LlmProvider::Anthropic,
        )
    }

    pub fn load_command_palette_entries(&self) -> Result<Vec<TuiCommandPaletteEntry>, AppError> {
        let mut entries = build_live_tool_registry(&self.config)
            .specs()
            .into_iter()
            .map(|spec| TuiCommandPaletteEntry {
                title: format!("/{}", display_command_name(&spec.name)),
                description: spec.description,
                action: TuiCommandPaletteAction::ToolCommand(spec.name),
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.title.cmp(&right.title));
        Ok(entries)
    }

    pub fn launch_named_command(&self, name: &str) -> Result<TuiSlashCommandAction, AppError> {
        let registry = build_live_tool_registry(&self.config);
        let Some(spec) = registry.specs().into_iter().find(|spec| spec.name == name) else {
            return Err(AppError::Runtime(format!("Invalid command: {name}")));
        };
        let input_suggestions = self.load_command_form_suggestions()?;
        let JsonSchema::Object {
            properties,
            required,
            ..
        } = &spec.parameters;
        let parameters = ordered_command_parameters(name, properties, required, &input_suggestions);
        let required_count = parameters
            .iter()
            .filter(|parameter| !parameter.optional)
            .count();
        if required_count == 0 {
            return Ok(TuiSlashCommandAction::Execute(TuiCommandExecution {
                command_name: name.to_string(),
                display_name: display_command_name(name).to_string(),
                values: vec![],
            }));
        }
        Ok(TuiSlashCommandAction::OpenForm(TuiCommandForm {
            command_name: name.to_string(),
            description: spec.description,
            parameters,
            initial_values: vec![],
        }))
    }

    pub fn handle_slash_command(&self, prompt: &str) -> Result<TuiSlashCommandAction, AppError> {
        let trimmed = prompt.trim();
        let Some(command_text) = trimmed.strip_prefix('/') else {
            return Ok(TuiSlashCommandAction::NotHandled);
        };
        let parts = command_text.split_whitespace().collect::<Vec<_>>();
        if parts.is_empty() {
            return Err(AppError::Runtime("Invalid command: /".to_string()));
        }

        let slash_name = parts[0];
        let raw_values = &parts[1..];
        let command_name = match slash_name {
            "help" => "get_help",
            _ => slash_name,
        };
        let registry = build_live_tool_registry(&self.config);
        let Some(spec) = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == command_name)
        else {
            return Err(AppError::Runtime(format!("Invalid command: {slash_name}")));
        };
        let input_suggestions = self.load_command_form_suggestions()?;
        let JsonSchema::Object {
            properties,
            required,
            ..
        } = &spec.parameters;
        let parameters =
            ordered_command_parameters(command_name, properties, required, &input_suggestions);
        if raw_values.len() > parameters.len() {
            return Err(AppError::Runtime(format!(
                "Too many values provided for '{slash_name}'"
            )));
        }
        let required_count = parameters
            .iter()
            .filter(|parameter| !parameter.optional)
            .count();
        if raw_values.len() < required_count {
            return Ok(TuiSlashCommandAction::OpenForm(TuiCommandForm {
                command_name: command_name.to_string(),
                description: spec.description,
                parameters: parameters.clone(),
                initial_values: parameters
                    .iter()
                    .zip(raw_values.iter())
                    .map(|(parameter, value)| (parameter.name.clone(), (*value).to_string()))
                    .collect(),
            }));
        }

        Ok(TuiSlashCommandAction::Execute(TuiCommandExecution {
            command_name: command_name.to_string(),
            display_name: slash_name.to_string(),
            values: parameters
                .iter()
                .zip(raw_values.iter())
                .map(|(parameter, value)| (parameter.name.clone(), (*value).to_string()))
                .collect(),
        }))
    }

    pub fn execute_command(
        &self,
        command_name: &str,
        display_name: &str,
        values: &[(String, String)],
    ) -> Result<TuiSnapshot, AppError> {
        self.execute_command_with_values(command_name, display_name, values)
    }

    fn execute_command_with_values(
        &self,
        command_name: &str,
        slash_name: &str,
        values: &[(String, String)],
    ) -> Result<TuiSnapshot, AppError> {
        let registry = build_live_tool_registry(&self.config);
        let arguments = Value::Object(
            values
                .iter()
                .filter(|(_, value)| !value.trim().is_empty())
                .map(|(name, value)| (name.clone(), Value::String(value.clone())))
                .collect(),
        );
        let result = registry.invoke(command_name, &arguments.to_string());
        let mut snapshot = self.load_snapshot()?;
        let trimmed_content = result.content.trim();
        let use_toast_result = !result.is_error
            && command_result_target(command_name) == CommandResultTarget::Toast
            && is_short_single_line_result(trimmed_content);

        if !trimmed_content.is_empty() && !use_toast_result {
            let label = if result.is_error {
                "tool error"
            } else {
                "tool result"
            };
            snapshot
                .conversation_lines
                .push(format!("{label}: {trimmed_content}"));
        }
        snapshot.status = Some(if result.is_error {
            format!("slash command failed: /{slash_name}")
        } else if use_toast_result {
            trimmed_content.to_string()
        } else {
            format!("slash command executed: /{slash_name}")
        });
        Ok(snapshot)
    }

    pub fn refresh_context_if_needed(&self) -> Result<bool, AppError> {
        let mut connection = self.open_connection()?;
        refresh_context_if_needed(
            &mut connection,
            &self.config,
            &BootstrapPlan::from_config(&self.config),
        )
    }

    pub fn run_self_reflection_if_needed(&self) -> Result<(), AppError> {
        let mut connection = self.open_connection()?;
        let transcript = load_validated_runtime_transcript(
            &mut connection,
            &self.config.assistant_name,
            self.config.llm_provider() == LlmProvider::Anthropic,
        )?;
        run_self_reflection_if_needed(
            &self.config.home_dir,
            transcript.as_slice(),
            self.config.messages_between_self_reflection,
        )
    }

    pub fn run_auto_memory_for_transcript(
        &self,
        existing_transcript_len: usize,
        transcript: Vec<ConversationMessage>,
    ) -> Result<(), AppError> {
        set_background_status("auto-memory", "creating memory from recent conversation...");
        let result = (|| {
            let mut connection = self.open_connection()?;
            run_auto_memory_if_needed(
                &mut connection,
                &BootstrapPlan::from_config(&self.config),
                self.config.memories_between_consolidation,
                existing_transcript_len,
                transcript.as_slice(),
                self.config.messages_between_memory,
            )
        })();
        clear_background_status("auto-memory");
        result
    }

    pub fn background_status(&self) -> Option<String> {
        get_background_status()
    }

    pub fn submit_prompt(&self, prompt: &str) -> Result<PromptRunResult, AppError> {
        self.process_message(prompt, MessageProcessOptions::default())
    }

    pub fn process_message_stream(
        &self,
        prompt: &str,
        options: MessageProcessOptions,
    ) -> Result<PromptEventStream, AppError> {
        let connection = self.open_connection()?;
        let preferences = load_user_preferences(&connection, LOCAL_USER_TOKEN)?;
        let model = live_provider_model(&self.config, preferences.as_ref())?;
        let classifier_model = live_provider_model(&self.config, preferences.as_ref())?;
        let executable_tools = if options.enable_tools {
            build_live_tool_registry(&self.config)
        } else {
            ExecutableToolRegistry::new(vec![])
        };
        let force_tool = if options.enable_tools {
            options.force_tool.as_deref()
        } else {
            None
        };
        run_prompt_with_model_and_registry_stream_internal(
            connection,
            self.config.home_dir.clone(),
            prompt,
            PromptExecutionOptions {
                role: options.role,
                persist_input_message: options.persist_input_message,
                force_tool,
                assistant_name: &self.config.assistant_name,
                ensure_alternating_roles: self.config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &self.config.home_dir,
                bootstrap_plan: BootstrapPlan::from_config(&self.config),
                messages_between_memory: self.config.messages_between_memory,
                memories_between_consolidation: self.config.memories_between_consolidation,
                messages_between_self_reflection: self.config.messages_between_self_reflection,
                defer_auto_memory: options.defer_auto_memory,
                defer_self_reflection: options.defer_self_reflection,
                memory_recall_classifier_enabled: self.config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: self.config.memory_recall_classifier_window,
                reflect: self.config.reflect,
            },
            Box::new(model),
            Some(&classifier_model),
            executable_tools,
        )
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
        let force_tool = if options.enable_tools {
            options.force_tool.as_deref()
        } else {
            None
        };
        let events = run_prompt_with_model_and_registry_internal(
            &mut connection,
            prompt,
            &model,
            Some(&model),
            executable_tools,
            PromptExecutionOptions {
                role: options.role,
                persist_input_message: options.persist_input_message,
                force_tool,
                assistant_name: &self.config.assistant_name,
                ensure_alternating_roles: self.config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &self.config.home_dir,
                bootstrap_plan: BootstrapPlan::from_config(&self.config),
                messages_between_memory: self.config.messages_between_memory,
                memories_between_consolidation: self.config.memories_between_consolidation,
                messages_between_self_reflection: self.config.messages_between_self_reflection,
                defer_auto_memory: options.defer_auto_memory,
                defer_self_reflection: options.defer_self_reflection,
                memory_recall_classifier_enabled: self.config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: self.config.memory_recall_classifier_window,
                reflect: self.config.reflect,
            },
        )?;

        Ok(PromptRunResult {
            events,
            snapshot: load_snapshot_from_connection(
                &mut connection,
                &self.config.home_dir,
                &self.config.memory_dir,
            )?,
        })
    }

    pub fn startup_prompt_stream(
        &self,
        restart_resume_message: Option<&str>,
    ) -> Result<Option<PromptEventStream>, AppError> {
        if let Some(prompt) = restart_resume_message {
            return self
                .process_message_stream(
                    prompt,
                    MessageProcessOptions {
                        enable_tools: false,
                        persist_input_message: false,
                        defer_auto_memory: true,
                        ..MessageProcessOptions::default()
                    },
                )
                .map(Some);
        }

        if !self.config.enable_assistant_greeting {
            return Ok(None);
        }

        let mut connection = self.open_connection()?;
        if !should_offer_greeting(
            &load_context_messages(&mut connection, LOCAL_USER_TOKEN)?,
            self.config.min_convo_age_for_greeting_minutes,
        ) {
            return Ok(None);
        }
        drop(connection);

        self.process_message_stream(
            "<Empty user response>",
            MessageProcessOptions {
                enable_tools: false,
                defer_auto_memory: true,
                ..MessageProcessOptions::default()
            },
        )
        .map(Some)
    }

    pub fn restart_prompt_stream(
        &self,
        resume_message: &str,
    ) -> Result<PromptEventStream, AppError> {
        self.startup_prompt_stream(Some(resume_message))?
            .ok_or_else(|| AppError::Runtime("restart prompt stream should exist".to_string()))
    }

    pub fn open_sidebar_item(
        &self,
        section: SidebarSection,
        title: &str,
    ) -> Result<TuiSidebarDetail, AppError> {
        let connection = self.open_read_connection()?;
        match section {
            SidebarSection::Memories => {
                let Some(memory) = find_active_memory_by_name_in_scope(
                    &connection,
                    title,
                    &self.config.memory_dir,
                )?
                else {
                    return Err(AppError::Runtime(format!("memory not found: {title}")));
                };
                let memory_name = memory.name.clone();
                Ok(TuiSidebarDetail {
                    title: memory_name.clone(),
                    content: format!(
                        "memory: {}\npath: {}\n\n{}",
                        memory_name, memory.file_path, memory.body
                    ),
                    can_complete: false,
                    destructive_action: Some(SidebarAction::Archive),
                    destructive_label: Some("archive".to_string()),
                })
            }
            SidebarSection::Agenda => {
                let Some(item) = resolve_agenda_sidebar_item(&connection, title)? else {
                    return Err(AppError::Runtime(format!("agenda item not found: {title}")));
                };
                let can_delete = item.trigger_datetime.is_some() || item.trigger_context.is_some();
                let mut lines = vec![
                    format!("agenda: {}", item.name),
                    format!("path: {}", item.file_path),
                ];
                if let Some(date) = item.agenda_date {
                    lines.push(format!("date: {date}"));
                }
                if let Some(trigger_datetime) = item.trigger_datetime.as_ref() {
                    lines.push(format!("trigger_datetime: {trigger_datetime}"));
                }
                if let Some(trigger_context) = item.trigger_context.as_ref() {
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
                Ok(TuiSidebarDetail {
                    title: item.name,
                    content: lines.join("\n"),
                    can_complete: item.status.as_deref() == Some("created"),
                    destructive_action: can_delete.then_some(SidebarAction::Delete),
                    destructive_label: can_delete.then_some("delete".to_string()),
                })
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
                if let Some(worktree_path) = session.worktree_path.as_ref() {
                    lines.push(format!("worktree_path: {worktree_path}"));
                }
                if let Some(session_branch) = session.session_branch.as_ref() {
                    lines.push(format!("session_branch: {session_branch}"));
                }
                if let Some(target_branch) = session.target_branch.as_ref() {
                    lines.push(format!("target_branch: {target_branch}"));
                }
                if let Some(session_file_path) = session.session_file_path.as_ref() {
                    lines.push(format!("session_file_path: {session_file_path}"));
                }
                lines.push(String::new());
                lines.push(
                    session
                        .latest_summary
                        .clone()
                        .unwrap_or_else(|| "(No summary recorded.)".to_string()),
                );
                if let Some(agent_message) = session.latest_agent_message.as_ref() {
                    lines.push(String::new());
                    lines.push(agent_message.clone());
                }
                Ok(TuiSidebarDetail {
                    title: format_codex_session_title(&session),
                    content: lines.join("\n"),
                    can_complete: false,
                    destructive_action: None,
                    destructive_label: None,
                })
            }
            SidebarSection::Improvements | SidebarSection::FeatureRequests => {
                let Some(record) =
                    resolve_feature_request_sidebar_item(&self.config.home_dir, section, title)
                        .map_err(AppError::Io)?
                else {
                    return Err(AppError::Runtime(format!(
                        "feature request not found: {title}"
                    )));
                };
                Ok(TuiSidebarDetail {
                    title: record.title.clone(),
                    content: feature_request_detail_content(&record),
                    can_complete: is_active_feature_request(&record),
                    destructive_action: None,
                    destructive_label: None,
                })
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
            (SidebarSection::Agenda, SidebarAction::Complete) => {
                let connection = self.open_read_connection()?;
                let Some(item) = resolve_agenda_sidebar_item(&connection, title)? else {
                    return Err(AppError::Runtime(format!("agenda item not found: {title}")));
                };
                if item.status.as_deref() != Some("created") {
                    return Err(AppError::Runtime(format!(
                        "agenda item cannot be completed from the sidebar: {}",
                        item.name
                    )));
                }
                registry.invoke(
                    "complete_agenda_item",
                    &json!({ "name": item.name }).to_string(),
                )
            }
            (SidebarSection::Agenda, SidebarAction::Delete) => {
                let connection = self.open_read_connection()?;
                let Some(item) = resolve_agenda_sidebar_item(&connection, title)? else {
                    return Err(AppError::Runtime(format!("agenda item not found: {title}")));
                };
                if item.trigger_datetime.is_none() && item.trigger_context.is_none() {
                    return Err(AppError::Runtime(format!(
                        "agenda item is not deletable from the sidebar: {}",
                        item.name
                    )));
                }
                registry.invoke(
                    "delete_agenda_item",
                    &json!({ "name": item.name }).to_string(),
                )
            }
            (SidebarSection::CodexSessions, _) => {
                return Err(AppError::Runtime(
                    "codex sessions are read-only in the sidebar".to_string(),
                ));
            }
            (
                SidebarSection::Improvements | SidebarSection::FeatureRequests,
                SidebarAction::Complete,
            ) => {
                let Some(record) =
                    resolve_feature_request_sidebar_item(&self.config.home_dir, section, title)
                        .map_err(AppError::Io)?
                else {
                    return Err(AppError::Runtime(format!(
                        "feature request not found: {title}"
                    )));
                };
                if !is_active_feature_request(&record) {
                    return Err(AppError::Runtime(format!(
                        "feature request cannot be completed from the sidebar: {}",
                        record.title
                    )));
                }
                update_feature_request(&record, None, Some("closed"), None, None, None, None)
                    .map_err(AppError::Io)?;
                ToolExecutionResult::success("closed feature request".to_string())
            }
            (SidebarSection::Improvements | SidebarSection::FeatureRequests, _) => {
                return Err(AppError::Runtime(
                    "unsupported feature request sidebar action".to_string(),
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
        drop_old_context_messages(&mut connection, self.config.max_context_age_minutes)?;
        Ok(connection)
    }

    fn open_read_connection(&self) -> Result<rusqlite::Connection, AppError> {
        let connection = open_sqlite_connection(&self.config.database_path)?;
        Ok(connection)
    }

    fn load_command_form_suggestions(&self) -> Result<Vec<String>, AppError> {
        let bootstrap_plan = BootstrapPlan::from_config(&self.config);
        elroy_db::bootstrap_database(&bootstrap_plan)
            .map_err(|error| AppError::Runtime(error.to_string()))?;
        let connection = self.open_connection()?;
        Ok(list_active_plain_agenda_items(&connection, 50)?
            .into_iter()
            .map(|item| item.name)
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandResultTarget {
    History,
    Toast,
}

fn command_result_target(command_name: &str) -> CommandResultTarget {
    match command_name {
        "refresh_system_instructions"
        | "reset_messages"
        | "set_assistant_name"
        | "set_user_full_name"
        | "set_user_preferred_name"
        | "create_due_item"
        | "complete_due_item"
        | "delete_due_item"
        | "rename_due_item"
        | "update_due_item_text"
        | "create_memory"
        | "update_outdated_or_incorrect_memory"
        | "add_memory_to_current_context"
        | "drop_memory_from_current_context" => CommandResultTarget::Toast,
        _ => CommandResultTarget::History,
    }
}

fn is_short_single_line_result(content: &str) -> bool {
    !content.is_empty() && !content.contains('\n') && content.chars().count() <= 180
}

fn ordered_command_parameters(
    command_name: &str,
    properties: &Map<String, Value>,
    required: &[String],
    input_suggestions: &[String],
) -> Vec<TuiCommandParameter> {
    let mut names = properties.keys().cloned().collect::<Vec<_>>();
    drop_legacy_alias(&mut names, "memory_name", "name");
    drop_legacy_alias(&mut names, "item_date", "date");
    drop_legacy_alias(&mut names, "old_name", "name");
    drop_legacy_alias(&mut names, "new_text", "text");
    names.sort_by_key(|name| preferred_command_parameter_rank(command_name, name));
    names
        .into_iter()
        .map(|name| TuiCommandParameter {
            optional: !required.contains(&name)
                && !canonical_alias_field_is_required(&name, properties),
            default_text: String::new(),
            suggestions: command_parameter_suggestions(&name, input_suggestions),
            name,
        })
        .collect()
}

fn command_parameter_suggestions(name: &str, input_suggestions: &[String]) -> Vec<String> {
    if name.ends_with("name") {
        return input_suggestions.to_vec();
    }
    Vec::new()
}

fn display_command_name(name: &str) -> &str {
    if name == "get_help" { "help" } else { name }
}

fn drop_legacy_alias(names: &mut Vec<String>, canonical: &str, alias: &str) {
    if names.iter().any(|name| name == canonical) {
        names.retain(|name| name != alias);
    }
}

fn canonical_alias_field_is_required(name: &str, properties: &Map<String, Value>) -> bool {
    matches!(name, "memory_name" | "item_date" | "old_name" | "new_text")
        && match name {
            "memory_name" => properties.contains_key("name"),
            "item_date" => properties.contains_key("date"),
            "old_name" => properties.contains_key("name"),
            "new_text" => properties.contains_key("text"),
            _ => false,
        }
}

fn preferred_command_parameter_rank(command_name: &str, name: &str) -> usize {
    const ORDER: &[&str] = &[
        "name",
        "memory_name",
        "item_name",
        "old_name",
        "new_name",
        "text",
        "new_text",
        "question",
        "query",
        "item_date",
        "date",
        "trigger_time",
        "trigger_datetime",
        "trigger_context",
        "closing_comment",
        "path",
        "start_line",
        "end_line",
        "n",
    ];

    if command_name == "get_help" {
        return usize::MAX;
    }

    ORDER
        .iter()
        .position(|candidate| *candidate == name)
        .unwrap_or(ORDER.len() + name.bytes().next().unwrap_or_default() as usize)
}

fn load_snapshot_from_connection(
    connection: &mut rusqlite::Connection,
    home_dir: &Path,
    memory_dir: &Path,
) -> Result<TuiSnapshot, AppError> {
    let conversation_lines = load_context_messages(connection, LOCAL_USER_TOKEN)?
        .into_iter()
        .filter(|message| {
            !(message.role == MessageRole::User
                && message.content.as_deref() == Some(SYNTHETIC_FIRST_USER_MESSAGE))
        })
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
    let memory_titles = list_active_memories_in_scope(connection, memory_dir, 15)?
        .into_iter()
        .map(|memory| memory.name)
        .collect::<Vec<_>>();
    let now = Utc::now().naive_utc();
    let agenda_titles = list_active_tasks(connection, 15)?
        .into_iter()
        .map(|item| format_agenda_sidebar_title(&item, now))
        .collect::<Vec<_>>();
    let input_completions = list_active_plain_agenda_items(connection, 50)?
        .into_iter()
        .map(|item| item.name)
        .collect::<Vec<_>>();
    let improvement_titles = list_self_reflection_feature_requests(home_dir, true)
        .map_err(AppError::Io)?
        .into_iter()
        .take(15)
        .map(format_feature_request_sidebar_title)
        .collect::<Vec<_>>();
    let feature_request_titles = list_feature_requests(home_dir)
        .map_err(AppError::Io)?
        .into_iter()
        .take(15)
        .map(format_feature_request_sidebar_title)
        .collect::<Vec<_>>();
    let codex_session_titles = list_recent_codex_sessions(connection, LOCAL_USER_TOKEN, None, 15)?
        .into_iter()
        .map(|session| format_codex_session_title(&session))
        .collect::<Vec<_>>();

    Ok(TuiSnapshot {
        conversation_lines,
        memory_titles,
        agenda_titles,
        input_completions,
        improvement_titles,
        feature_request_titles,
        codex_session_titles,
        model_name: None,
        status: Some("loaded persisted transcript and sidebar data".to_string()),
    })
}

fn drop_old_context_messages(
    connection: &mut rusqlite::Connection,
    max_context_age_minutes: f64,
) -> Result<(), AppError> {
    let context_messages = load_context_messages(connection, LOCAL_USER_TOKEN)?;
    if context_messages.is_empty() {
        return Ok(());
    }

    let cutoff_unix = Utc::now().timestamp() - (max_context_age_minutes * 60.0) as i64;
    let first_message_id = context_messages.first().and_then(|message| message.id);
    let kept_messages = context_messages
        .iter()
        .filter(|message| {
            message.role == MessageRole::System
                || message.created_at_unix >= cutoff_unix
                || first_message_id
                    .as_ref()
                    .zip(message.id.as_ref())
                    .is_some_and(|(first_id, message_id)| first_id == message_id)
        })
        .cloned()
        .collect::<Vec<_>>();

    if kept_messages.len() != context_messages.len() {
        replace_context_messages(connection, LOCAL_USER_TOKEN, &kept_messages)?;
    }

    Ok(())
}

fn current_system_message(
    default_assistant_name: &str,
    preferences: Option<&UserPreferenceRecord>,
) -> ConversationMessage {
    ConversationMessage::new(
        MessageRole::System,
        effective_persona(preferences, default_assistant_name),
    )
}

fn repair_system_message_placement(
    raw_messages: &[ConversationMessage],
    expected_system_message: &ConversationMessage,
) -> Vec<ConversationMessage> {
    let Some(first_message) = raw_messages.first() else {
        return Vec::new();
    };

    if first_message.role == MessageRole::System
        && !raw_messages
            .iter()
            .skip(1)
            .any(|message| message.role == MessageRole::System)
        && first_message.content == expected_system_message.content
    {
        return raw_messages.to_vec();
    }

    let mut repaired = Vec::with_capacity(raw_messages.len().saturating_add(1));
    if first_message.role == MessageRole::System {
        let mut refreshed_first = first_message.clone();
        refreshed_first.content = expected_system_message.content.clone();
        refreshed_first.chat_model = None;
        refreshed_first.tool_calls = None;
        refreshed_first.tool_call_id = None;
        repaired.push(refreshed_first);
        repaired.extend(
            raw_messages
                .iter()
                .skip(1)
                .filter(|message| message.role != MessageRole::System)
                .cloned(),
        );
    } else {
        repaired.push(expected_system_message.clone());
        repaired.extend(
            raw_messages
                .iter()
                .filter(|message| message.role != MessageRole::System)
                .cloned(),
        );
    }
    repaired
}

fn repair_first_user_precedes_first_assistant(
    raw_messages: &[ConversationMessage],
    ensure_alternating_roles: bool,
) -> Vec<ConversationMessage> {
    if !ensure_alternating_roles {
        return raw_messages.to_vec();
    }

    let first_non_system_index = raw_messages
        .iter()
        .position(|message| message.role != MessageRole::System);
    let Some(index) = first_non_system_index else {
        return raw_messages.to_vec();
    };

    if raw_messages[index].role != MessageRole::Assistant {
        return raw_messages.to_vec();
    }

    let mut repaired = raw_messages.to_vec();
    repaired.insert(
        index,
        ConversationMessage::new(MessageRole::User, SYNTHETIC_FIRST_USER_MESSAGE),
    );
    repaired
}

fn load_validated_runtime_transcript(
    connection: &mut rusqlite::Connection,
    default_assistant_name: &str,
    ensure_alternating_roles: bool,
) -> Result<Vec<ConversationMessage>, AppError> {
    let raw_messages = load_context_messages(connection, LOCAL_USER_TOKEN)?;
    if raw_messages.is_empty() {
        return Ok(Vec::new());
    }

    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    let expected_system_message =
        current_system_message(default_assistant_name, preferences.as_ref());
    let repaired_messages = repair_first_user_precedes_first_assistant(
        &repair_system_message_placement(&raw_messages, &expected_system_message),
        ensure_alternating_roles,
    );
    let validated = validated_transcript(&repaired_messages);

    if validated != raw_messages {
        replace_context_messages(connection, LOCAL_USER_TOKEN, &validated)?;
        return load_context_messages(connection, LOCAL_USER_TOKEN).map_err(AppError::from);
    }

    Ok(validated)
}

fn refreshed_context_messages_with_system(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> Result<Vec<ConversationMessage>, AppError> {
    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    let system_message = current_system_message(&config.assistant_name, preferences.as_ref());
    let mut refreshed = load_context_messages(connection, LOCAL_USER_TOKEN)?
        .into_iter()
        .filter(|message| message.role != MessageRole::System)
        .collect::<Vec<_>>();
    refreshed.insert(0, system_message);
    Ok(refreshed)
}

fn refresh_persisted_system_instructions(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> Result<(), AppError> {
    let refreshed = refreshed_context_messages_with_system(connection, config)?;
    replace_context_messages(connection, LOCAL_USER_TOKEN, &refreshed)?;
    Ok(())
}

fn reset_persisted_context(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> Result<(), AppError> {
    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    let system_message = current_system_message(&config.assistant_name, preferences.as_ref());
    replace_context_messages(connection, LOCAL_USER_TOKEN, &[system_message])?;
    Ok(())
}

fn format_agenda_sidebar_title(item: &AgendaItemRecord, now: NaiveDateTime) -> String {
    let mut title = item.name.clone();
    if let Some(trigger_datetime) = item
        .trigger_datetime
        .as_deref()
        .and_then(parse_sidebar_trigger_datetime)
    {
        title = format!("{} [{}]", title, trigger_datetime.format("%Y-%m-%d %H:%M"));
        if trigger_datetime <= now {
            title.push_str(" (Due)");
        }
    }
    title
}

fn parse_sidebar_trigger_datetime(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M").ok())
}

fn resolve_agenda_sidebar_item(
    connection: &rusqlite::Connection,
    title: &str,
) -> rusqlite::Result<Option<AgendaItemRecord>> {
    if let Some(item) = find_active_agenda_item_by_name(connection, title)? {
        return Ok(Some(item));
    }

    let now = Utc::now().naive_utc();
    Ok(list_active_tasks(connection, 200)?
        .into_iter()
        .find(|item| format_agenda_sidebar_title(item, now) == title))
}

fn format_feature_request_sidebar_title(record: FeatureRequestRecord) -> String {
    format!("{} ({})", record.title, record.status)
}

fn feature_request_detail_content(record: &FeatureRequestRecord) -> String {
    let source_label = if record.source == "self_reflection" {
        "Self-reflection".to_string()
    } else {
        record
            .source
            .replace('_', " ")
            .split(' ')
            .map(capitalize_word)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut lines = vec![
        format!("Status: {}", record.status),
        format!("Source: {source_label}"),
        String::new(),
        "Summary:".to_string(),
        record.summary.clone(),
    ];
    if let Some(rationale) = &record.rationale {
        lines.push(String::new());
        lines.push("Why It Matters:".to_string());
        lines.push(rationale.clone());
    }
    if let Some(supporting_context) = &record.supporting_context {
        lines.push(String::new());
        lines.push("Supporting Context:".to_string());
        lines.push(supporting_context.clone());
    }
    lines.join("\n")
}

fn feature_request_listing_content(records: &[FeatureRequestRecord]) -> String {
    if records.is_empty() {
        return "No feature requests found.".to_string();
    }

    let mut lines = vec![
        format!("Feature requests ({}):", records.len()),
        String::new(),
    ];
    for record in records {
        let aliases = if record.aliases.is_empty() {
            String::new()
        } else {
            format!(" | aliases: {}", record.aliases.join(", "))
        };
        lines.push(format!(
            "- {} [{}] ({}){}",
            record.title,
            record.status,
            record
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
            aliases
        ));
        lines.push(format!("  Summary: {}", record.summary));
    }
    lines.join("\n")
}

fn normalize_optional_tool_string(value: Option<&str>) -> Option<&str> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

fn build_feature_request_supporting_context(
    user_token: &str,
    title: &str,
    description: &str,
    rationale: Option<&str>,
) -> String {
    let mut lines = vec![
        format!("- Captured at: {}", Utc::now().to_rfc3339()),
        format!("- Requested title: {title}"),
        format!("- Description: {}", description.trim()),
    ];
    if let Some(rationale) = rationale {
        lines.push(format!("- Rationale: {}", rationale.trim()));
    }
    lines.push(format!("- User token: {user_token}"));
    lines.join("\n")
}

fn merge_feature_request_supporting_context(existing: Option<&str>, new_context: &str) -> String {
    match normalize_optional_tool_string(existing) {
        None => new_context.to_string(),
        Some(existing) if existing.contains(new_context) => existing.to_string(),
        Some(existing) => format!("{}\n\n{}", existing.trim_end(), new_context),
    }
}

fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

fn resolve_feature_request_sidebar_item(
    home_dir: &Path,
    section: SidebarSection,
    title: &str,
) -> std::io::Result<Option<FeatureRequestRecord>> {
    let records = match section {
        SidebarSection::Improvements => list_self_reflection_feature_requests(home_dir, true)?,
        SidebarSection::FeatureRequests => list_feature_requests(home_dir)?,
        _ => return Ok(None),
    };
    Ok(records
        .into_iter()
        .find(|record| format_feature_request_sidebar_title(record.clone()) == title)
        .or_else(|| get_feature_request(home_dir, title).ok().flatten()))
}

fn should_offer_greeting(
    context_messages: &[ConversationMessage],
    min_convo_age_for_greeting_minutes: f64,
) -> bool {
    let Some(last_user_message) = context_messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::User)
    else {
        return false;
    };

    let age_seconds = Utc::now().timestamp() - last_user_message.created_at_unix;
    let min_age_seconds = (min_convo_age_for_greeting_minutes * 60.0) as i64;
    age_seconds >= min_age_seconds
}

fn finalize_prompt_event_stream(
    mut state: PromptEventStreamState,
) -> Result<PromptCompletion, AppError> {
    let turn_run = state.turn_stream.finish()?;
    let persisted_transcript = strip_input_message_for_persistence(
        strip_transient_context_messages(
            turn_run.transcript,
            state.existing_transcript_len,
            state.transient_context_count,
        ),
        state.existing_transcript_len,
        state.persist_input_message,
    );
    replace_context_messages(
        &mut state.connection,
        LOCAL_USER_TOKEN,
        &persisted_transcript,
    )?;
    let deferred_auto_memory = if state.defer_auto_memory {
        Some(DeferredAutoMemoryWork {
            existing_transcript_len: state.existing_transcript_len,
            transcript: persisted_transcript.clone(),
        })
    } else {
        run_auto_memory_if_needed(
            &mut state.connection,
            &state.bootstrap_plan,
            state.memories_between_consolidation,
            state.existing_transcript_len,
            persisted_transcript.as_slice(),
            state.messages_between_memory,
        )?;
        None
    };
    if !state.defer_self_reflection {
        run_self_reflection_if_needed(
            &state.home_dir,
            persisted_transcript.as_slice(),
            state.messages_between_self_reflection,
        )?;
    }
    Ok(PromptCompletion {
        snapshot: load_snapshot_from_connection(
            &mut state.connection,
            &state.home_dir,
            &state.bootstrap_plan.memory_dir,
        )?,
        deferred_auto_memory,
    })
}

fn cancel_prompt_event_stream(mut state: PromptEventStreamState) -> Result<TuiSnapshot, AppError> {
    load_snapshot_from_connection(
        &mut state.connection,
        &state.home_dir,
        &state.bootstrap_plan.memory_dir,
    )
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

fn best_effort_provider_model(
    provider_config: Option<&ProviderConfig>,
    assistant_name: &str,
) -> Option<LiveProviderModel> {
    let provider_config = provider_config.cloned()?;
    let client = LiveModelClient::new(provider_config).ok()?;
    Some(LiveProviderModel::new(
        client,
        effective_persona(None, assistant_name),
    ))
}

fn run_prompt_with_model_and_registry_internal(
    connection: &mut rusqlite::Connection,
    prompt: &str,
    model: &dyn ModelClient,
    classifier_model: Option<&dyn ModelClient>,
    executable_tools: ExecutableToolRegistry,
    options: PromptExecutionOptions<'_>,
) -> Result<Vec<StreamEvent>, AppError> {
    let tools = ToolRegistry::new(executable_tools.specs());
    if let Some(force_tool) = options.force_tool
        && !tools.specs().iter().any(|tool| tool.name == force_tool)
    {
        return Err(AppError::Runtime(format!(
            "Requested tool {force_tool} not available."
        )));
    }
    let orchestrator = ConversationOrchestrator::new(2);
    let tool_executor = LocalToolExecutor::new(executable_tools);
    let existing_transcript = load_validated_runtime_transcript(
        connection,
        options.assistant_name,
        options.ensure_alternating_roles,
    )?;
    let memory_recall_decision = determine_memory_recall_decision(
        options.memory_recall_classifier_enabled,
        options.memory_recall_classifier_window,
        prompt,
        &existing_transcript,
        classifier_model,
    );
    let all_due_items = list_active_due_items(connection, 20)?;
    let recall_context = recall_memory_context_messages_with_decision(
        options.memory_recall_classifier_window,
        options.reflect,
        prompt,
        memory_recall_decision.needs_recall,
        classifier_model,
        RecallContext {
            transcript: &existing_transcript,
            memories: &list_active_memories_in_scope(
                connection,
                &options.bootstrap_plan.memory_dir,
                50,
            )?,
            due_items: &all_due_items,
            agenda_items: &list_active_plain_agenda_items(connection, 20)?,
        },
    );
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let timed_due_item_context =
        due_item_context_messages(&list_due_tasks(connection, 20, &now_iso)?);
    let contextual_due_item_context =
        recall_due_item_context_messages(prompt, &existing_transcript, &all_due_items, &now_iso);
    let mut model_transcript = existing_transcript.clone();
    model_transcript.extend(contextual_due_item_context.iter().cloned());
    let persisted_transcript_start_len = model_transcript.len();
    model_transcript.extend(recall_context.iter().cloned());
    model_transcript.extend(timed_due_item_context.iter().cloned());

    let turn_run = orchestrator.run_turn_with_transcript_and_options(
        model,
        tools.specs(),
        &tool_executor,
        &model_transcript,
        elroy_core::ConversationOptions {
            role: options.role,
            force_tool: options.force_tool,
        },
        prompt,
    )?;
    let persisted_transcript = strip_input_message_for_persistence(
        strip_transient_context_messages(
            turn_run.transcript.clone(),
            persisted_transcript_start_len,
            recall_context.len() + timed_due_item_context.len(),
        ),
        persisted_transcript_start_len,
        options.persist_input_message,
    );
    replace_context_messages(connection, LOCAL_USER_TOKEN, &persisted_transcript)?;
    run_auto_memory_if_needed(
        connection,
        &options.bootstrap_plan,
        options.memories_between_consolidation,
        persisted_transcript_start_len,
        persisted_transcript.as_slice(),
        options.messages_between_memory,
    )?;
    if !options.defer_self_reflection {
        run_self_reflection_if_needed(
            options.home_dir,
            persisted_transcript.as_slice(),
            options.messages_between_self_reflection,
        )?;
    }

    let mut events = prompt_prelude_status_updates_with_decision(
        memory_recall_decision.used_llm,
        !recall_context.is_empty(),
        !(timed_due_item_context.is_empty() && contextual_due_item_context.is_empty()),
    );
    events.extend(turn_run.events);
    Ok(events)
}

#[cfg(test)]
fn run_prompt_with_model_and_registry(
    connection: &mut rusqlite::Connection,
    prompt: &str,
    model: &dyn ModelClient,
    executable_tools: ExecutableToolRegistry,
    options: PromptExecutionOptions<'_>,
) -> Result<Vec<StreamEvent>, AppError> {
    run_prompt_with_model_and_registry_internal(
        connection,
        prompt,
        model,
        None,
        executable_tools,
        options,
    )
}

fn run_prompt_with_model_and_registry_stream_internal(
    mut connection: rusqlite::Connection,
    home_dir: PathBuf,
    prompt: &str,
    options: PromptExecutionOptions<'_>,
    model: Box<dyn StreamingModelClient>,
    classifier_model: Option<&dyn ModelClient>,
    executable_tools: ExecutableToolRegistry,
) -> Result<PromptEventStream, AppError> {
    let tools = ToolRegistry::new(executable_tools.specs());
    if let Some(force_tool) = options.force_tool
        && !tools.specs().iter().any(|tool| tool.name == force_tool)
    {
        return Err(AppError::Runtime(format!(
            "Requested tool {force_tool} not available."
        )));
    }
    let orchestrator = ConversationOrchestrator::new(2);
    let tool_executor = Box::new(LocalToolExecutor::new(executable_tools));
    let existing_transcript = load_validated_runtime_transcript(
        &mut connection,
        options.assistant_name,
        options.ensure_alternating_roles,
    )?;
    let memory_recall_decision = determine_memory_recall_decision(
        options.memory_recall_classifier_enabled,
        options.memory_recall_classifier_window,
        prompt,
        &existing_transcript,
        classifier_model,
    );
    let all_due_items = list_active_due_items(&connection, 20)?;
    let recall_context = recall_memory_context_messages_with_decision(
        options.memory_recall_classifier_window,
        options.reflect,
        prompt,
        memory_recall_decision.needs_recall,
        classifier_model,
        RecallContext {
            transcript: &existing_transcript,
            memories: &list_active_memories_in_scope(
                &connection,
                &options.bootstrap_plan.memory_dir,
                50,
            )?,
            due_items: &all_due_items,
            agenda_items: &list_active_plain_agenda_items(&connection, 20)?,
        },
    );
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let timed_due_item_context =
        due_item_context_messages(&list_due_tasks(&connection, 20, &now_iso)?);
    let contextual_due_item_context =
        recall_due_item_context_messages(prompt, &existing_transcript, &all_due_items, &now_iso);
    let mut model_transcript = existing_transcript.clone();
    model_transcript.extend(contextual_due_item_context.iter().cloned());
    let persisted_transcript_start_len = model_transcript.len();
    model_transcript.extend(recall_context.iter().cloned());
    model_transcript.extend(timed_due_item_context.iter().cloned());

    let turn_stream = orchestrator.stream_turn_with_transcript_and_options(
        model,
        tools.specs().to_vec(),
        tool_executor,
        &model_transcript,
        elroy_core::ConversationOptions {
            role: options.role,
            force_tool: options.force_tool,
        },
        prompt,
    )?;

    let prelude_events = VecDeque::from(prompt_prelude_status_updates_with_decision(
        memory_recall_decision.used_llm,
        !recall_context.is_empty(),
        !(timed_due_item_context.is_empty() && contextual_due_item_context.is_empty()),
    ));

    Ok(PromptEventStream {
        state: Some(PromptEventStreamState {
            home_dir,
            bootstrap_plan: options.bootstrap_plan,
            connection,
            turn_stream,
            existing_transcript_len: persisted_transcript_start_len,
            transient_context_count: recall_context.len() + timed_due_item_context.len(),
            persist_input_message: options.persist_input_message,
            messages_between_memory: options.messages_between_memory,
            memories_between_consolidation: options.memories_between_consolidation,
            messages_between_self_reflection: options.messages_between_self_reflection,
            defer_auto_memory: options.defer_auto_memory,
            defer_self_reflection: options.defer_self_reflection,
            prelude_events,
        }),
        finalized_completion: None,
    })
}

#[cfg(test)]
fn run_prompt_with_model_and_registry_stream(
    connection: rusqlite::Connection,
    home_dir: PathBuf,
    prompt: &str,
    options: PromptExecutionOptions<'_>,
    model: Box<dyn StreamingModelClient>,
    executable_tools: ExecutableToolRegistry,
) -> Result<PromptEventStream, AppError> {
    run_prompt_with_model_and_registry_stream_internal(
        connection,
        home_dir,
        prompt,
        options,
        model,
        None,
        executable_tools,
    )
}

fn run_self_reflection_if_needed(
    home_dir: &Path,
    transcript: &[ConversationMessage],
    messages_between_self_reflection: usize,
) -> Result<(), AppError> {
    set_background_status("self-reflection", "reflecting on recent conversation...");
    let result = SelfReflectionOrchestrator::new(SelfReflectionConfig {
        messages_between_self_reflection,
    })
    .run(home_dir, transcript);
    clear_background_status("self-reflection");
    result?;
    Ok(())
}

fn refresh_context_if_needed(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
    bootstrap_plan: &BootstrapPlan,
) -> Result<bool, AppError> {
    let transcript = load_validated_runtime_transcript(
        connection,
        &config.assistant_name,
        config.llm_provider() == LlmProvider::Anthropic,
    )?;
    if !is_context_refresh_needed(&transcript, config.max_tokens) {
        return Ok(false);
    }

    set_background_status("context-refresh", "refreshing context...");

    let result = (|| {
        let compressed = compress_context_messages(
            &transcript,
            config.context_refresh_target_tokens(),
            config.max_context_age_minutes,
        );
        if transcript
            .iter()
            .any(|message| message.role == MessageRole::User)
        {
            let (name, text) = formulate_memory_from_transcript(&transcript);
            create_memory_file_from_context_messages(
                &bootstrap_plan.memory_dir,
                &name,
                &text,
                &transcript,
            )?;
            elroy_db::bootstrap_database(bootstrap_plan)
                .map_err(|error| AppError::Runtime(error.to_string()))?;
            *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
            record_memory_creation_and_maybe_consolidate(
                connection,
                bootstrap_plan,
                config.memories_between_consolidation,
            )?;
        }

        let mut refreshed_transcript = compressed;
        if transcript
            .iter()
            .any(|message| message.role == MessageRole::User)
        {
            let tool_call_id = format!(
                "context-summary-{}",
                Utc::now()
                    .timestamp_nanos_opt()
                    .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000)
            );
            refreshed_transcript.extend(synthetic_tool_context_messages(
                tool_call_id,
                "context_summary",
                "{}",
                build_context_summary_message(connection, config, &transcript),
            ));
        }

        replace_context_messages(connection, LOCAL_USER_TOKEN, &refreshed_transcript)?;
        Ok(true)
    })();

    clear_background_status("context-refresh");
    result
}

fn run_auto_memory_if_needed(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
    existing_transcript_len: usize,
    transcript: &[ConversationMessage],
    messages_between_memory: usize,
) -> Result<(), AppError> {
    if messages_between_memory == 0 {
        return Ok(());
    }

    let new_message_count = transcript
        .iter()
        .skip(existing_transcript_len)
        .filter(|message| {
            matches!(message.role, MessageRole::User | MessageRole::Assistant)
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| !content.trim().is_empty())
        })
        .count() as i64;
    if new_message_count == 0 {
        return Ok(());
    }

    let mut tracker = get_or_create_memory_operation_tracker(connection, LOCAL_USER_TOKEN)?;
    tracker.messages_since_memory += new_message_count;
    tracker.updated_at_unix = Utc::now().timestamp();

    if tracker.messages_since_memory < messages_between_memory as i64 {
        save_memory_operation_tracker(connection, &tracker)?;
        return Ok(());
    }

    let (name, text) = formulate_memory_from_transcript(transcript);
    let persisted_context_messages = load_context_messages(connection, LOCAL_USER_TOKEN)?;
    create_memory_file_from_context_messages(
        &bootstrap_plan.memory_dir,
        &name,
        &text,
        &persisted_context_messages,
    )?;
    elroy_db::bootstrap_database(bootstrap_plan)
        .map_err(|error| AppError::Runtime(error.to_string()))?;
    *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
    record_memory_creation_and_maybe_consolidate(
        connection,
        bootstrap_plan,
        memories_between_consolidation,
    )?;
    Ok(())
}

fn record_memory_creation_and_maybe_consolidate(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
) -> Result<(), AppError> {
    *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
    run_migrations(connection)?;
    let mut tracker = get_or_create_memory_operation_tracker(connection, LOCAL_USER_TOKEN)?;
    tracker.messages_since_memory = 0;
    tracker.memories_since_consolidation += 1;
    tracker.updated_at_unix = Utc::now().timestamp();

    if memories_between_consolidation == 0 {
        save_memory_operation_tracker(connection, &tracker)?;
        return Ok(());
    }

    if tracker.memories_since_consolidation < memories_between_consolidation as i64 {
        save_memory_operation_tracker(connection, &tracker)?;
        return Ok(());
    }

    (|| {
        consolidate_exact_duplicate_memories(connection, bootstrap_plan)?;
        tracker.memories_since_consolidation = 0;
        tracker.updated_at_unix = Utc::now().timestamp();
        save_memory_operation_tracker(connection, &tracker)?;
        Ok(())
    })()
}

fn consolidate_exact_duplicate_memories(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
) -> Result<(), AppError> {
    let active_memories =
        list_active_memories_in_scope(connection, &bootstrap_plan.memory_dir, 500)?;
    let mut groups = HashMap::<String, Vec<MemoryRecord>>::new();
    for memory in active_memories {
        let normalized = normalize_memory_body(&memory.body);
        if normalized.is_empty() {
            continue;
        }
        groups.entry(normalized).or_default().push(memory);
    }

    let duplicate_groups = groups
        .into_values()
        .filter(|group| group.len() >= 2)
        .collect::<Vec<_>>();

    for group in duplicate_groups {
        let canonical = group.first().ok_or_else(|| {
            AppError::Runtime("duplicate memory group was unexpectedly empty".to_string())
        })?;
        let source_names = group
            .iter()
            .map(|memory| memory.name.as_str())
            .collect::<Vec<_>>();
        create_consolidated_memory_from_plan(
            bootstrap_plan,
            &canonical.name,
            &canonical.body,
            &source_names,
        )
        .map_err(AppError::Io)?;
        *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
        run_migrations(connection)?;
    }

    Ok(())
}

fn normalize_memory_body(body: &str) -> String {
    body.to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn create_memory_file_from_context_messages(
    memory_dir: &Path,
    name: &str,
    text: &str,
    context_messages: &[ConversationMessage],
) -> std::io::Result<PathBuf> {
    let frontmatter = context_message_source_frontmatter(context_messages);
    create_memory_file_with_frontmatter(memory_dir, name, text, frontmatter.as_deref())
}

fn context_message_source_frontmatter(context_messages: &[ConversationMessage]) -> Option<String> {
    let message_ids = context_messages
        .iter()
        .filter_map(|message| message.id)
        .collect::<Vec<_>>();
    if message_ids.is_empty() {
        return None;
    }
    let message_ids_json = serde_json::to_string(&message_ids).ok()?;
    Some(format!(
        "source_type: {CONTEXT_MESSAGE_SOURCE_TYPE}\nmessage_ids_json: {message_ids_json}"
    ))
}

fn parse_context_message_source_ids(frontmatter: Option<&str>) -> Option<Vec<i64>> {
    let frontmatter = frontmatter?;
    let mut source_type = None;
    let mut message_ids_json = None;
    for line in frontmatter.lines() {
        if let Some(value) = line.strip_prefix("source_type:") {
            source_type = Some(value.trim().to_string());
        }
        if let Some(value) = line.strip_prefix("message_ids_json:") {
            message_ids_json = Some(value.trim().to_string());
        }
    }
    if source_type.as_deref() != Some(CONTEXT_MESSAGE_SOURCE_TYPE) {
        return None;
    }
    serde_json::from_str(message_ids_json?.as_str()).ok()
}

fn memory_source_frontmatter(memory_sources: &[(&str, &Path)]) -> Option<String> {
    if memory_sources.is_empty() {
        return None;
    }
    let names = memory_sources
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect::<Vec<_>>();
    let paths = memory_sources
        .iter()
        .map(|(_, path)| path.display().to_string())
        .collect::<Vec<_>>();
    let source_memory_names_json = serde_json::to_string(&names).ok()?;
    let source_memory_paths_json = serde_json::to_string(&paths).ok()?;
    Some(format!(
        "source_type: {MEMORY_SOURCE_TYPE}\nsource_memory_names_json: {source_memory_names_json}\nsource_memory_paths_json: {source_memory_paths_json}"
    ))
}

fn parse_memory_sources(frontmatter: Option<&str>) -> Option<Vec<(String, String)>> {
    let frontmatter = frontmatter?;
    let mut source_type = None;
    let mut names_json = None;
    let mut paths_json = None;
    for line in frontmatter.lines() {
        if let Some(value) = line.strip_prefix("source_type:") {
            source_type = Some(value.trim().to_string());
        }
        if let Some(value) = line.strip_prefix("source_memory_names_json:") {
            names_json = Some(value.trim().to_string());
        }
        if let Some(value) = line.strip_prefix("source_memory_paths_json:") {
            paths_json = Some(value.trim().to_string());
        }
    }
    if source_type.as_deref() != Some(MEMORY_SOURCE_TYPE) {
        return None;
    }
    let names = serde_json::from_str::<Vec<String>>(names_json?.as_str()).ok()?;
    let paths = serde_json::from_str::<Vec<String>>(paths_json?.as_str()).ok()?;
    if names.len() != paths.len() {
        return None;
    }
    Some(names.into_iter().zip(paths).collect())
}

fn list_memory_sources(frontmatter: Option<&str>) -> Vec<(String, String)> {
    if let Some(memory_sources) = parse_memory_sources(frontmatter) {
        return memory_sources
            .into_iter()
            .map(|(name, _)| (MEMORY_SOURCE_TYPE.to_string(), name))
            .collect();
    }
    if let Some(message_ids) = parse_context_message_source_ids(frontmatter) {
        let source_name = message_ids
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        return vec![(CONTEXT_MESSAGE_SOURCE_TYPE.to_string(), source_name)];
    }
    Vec::new()
}

fn format_context_message_source_content(messages: &[ConversationMessage]) -> String {
    messages
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            message
                .content
                .as_deref()
                .map(|content| format!("{role}: {content}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_memory_file_source_content(source_name: &str, source_body: &str) -> String {
    format!("#{source_name}\n{source_body}")
}

fn format_due_item_detail(item: &AgendaItemRecord) -> String {
    let mut lines = vec![format!("Due item '{}':", item.name)];
    if let Some(trigger_datetime) = &item.trigger_datetime {
        let formatted = parse_sidebar_trigger_datetime(trigger_datetime)
            .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| trigger_datetime.clone());
        lines.push(format!("Trigger Time: {formatted}"));
    }
    if let Some(trigger_context) = &item.trigger_context {
        lines.push(format!("Context: {trigger_context}"));
    }
    lines.push(format!("Text: {}", item.body));
    lines.join("\n")
}

fn format_due_item_listing(items: &[AgendaItemRecord], active: bool) -> String {
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

fn format_memory_detail(memory: &MemoryRecord) -> String {
    format!("#{}\n{}", memory.name, memory.body)
}

fn format_agenda_item_recall_detail(item: &AgendaItemRecord) -> String {
    if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
        let formatted = parse_sidebar_trigger_datetime(trigger_datetime)
            .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| trigger_datetime.to_string());
        return format!("#{} (Timed: {formatted})\n{}", item.name, item.body.trim());
    }
    if let Some(trigger_context) = item.trigger_context.as_deref() {
        return format!(
            "#{} (Context: {})\n{}",
            item.name,
            trigger_context,
            item.body.trim()
        );
    }
    format!("#Agenda: {}\n{}", item.name, item.body.trim())
}

fn format_memory_examination(memory: &MemoryRecord) -> String {
    format!(
        "# Memory: {}\n\n*to view the source content this memory is based on, call tool `get_source_content_for_memory({}, idx)`\n\n{}",
        memory.name,
        memory.name,
        memory.body.trim()
    )
}

fn format_memory_listing(memories: &[MemoryRecord]) -> String {
    if memories.is_empty() {
        return "No memories found.".to_string();
    }

    let mut lines = vec!["Memories".to_string()];
    for memory in memories.iter().rev() {
        lines.push(format!(
            "- {} | Text: {}",
            memory.name,
            excerpt(&memory.body, 180)
        ));
    }
    lines.join("\n")
}

fn format_memory_search_results(
    memories: &[&MemoryRecord],
    due_items: &[&AgendaItemRecord],
    agenda_items: &[&AgendaItemRecord],
) -> String {
    if memories.is_empty() && due_items.is_empty() && agenda_items.is_empty() {
        return "No relevant memories found".to_string();
    }

    let mut lines = vec!["Search Results".to_string()];
    for memory in memories {
        lines.push(format!(
            "- Memory | {} | {}",
            memory.name,
            excerpt(&memory.body, 180)
        ));
    }
    for item in due_items {
        lines.push(format!(
            "- DueItem | {} | {}",
            item.name,
            excerpt(&item.body, 180)
        ));
    }
    for item in agenda_items {
        lines.push(format!(
            "- AgendaItem | {} | {}",
            item.name,
            excerpt(&item.body, 180)
        ));
    }
    lines.join("\n")
}

fn derive_agenda_item_name(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or(text)
        .trim()
        .chars()
        .take(60)
        .collect()
}

fn formulate_memory_from_transcript(transcript: &[ConversationMessage]) -> (String, String) {
    let messages = transcript
        .iter()
        .filter(|message| {
            matches!(message.role, MessageRole::User | MessageRole::Assistant)
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| !content.trim().is_empty())
        })
        .rev()
        .take(6)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();

    let title_seed = messages
        .iter()
        .find(|message| message.role == MessageRole::User)
        .and_then(|message| message.content.as_deref())
        .unwrap_or("Conversation memory");
    let title_words = title_seed
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ");
    let title = if title_words.is_empty() {
        format!(
            "Conversation memory {}",
            Utc::now().format("%Y-%m-%d %H:%M")
        )
    } else {
        format!("Conversation memory: {title_words}")
    };

    let body = messages
        .into_iter()
        .map(|message| {
            let role = match message.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                _ => unreachable!("filtered to user/assistant"),
            };
            format!("{role}: {}", message.content.unwrap_or_default().trim())
        })
        .collect::<Vec<_>>()
        .join("\n");

    (title, body)
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
    run_prompt_with_model_and_registry_internal(
        &mut connection,
        &prompt,
        &model,
        Some(&model),
        build_live_tool_registry(config),
        PromptExecutionOptions {
            role: MessageRole::User,
            persist_input_message: false,
            force_tool: None,
            assistant_name: &config.assistant_name,
            ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
            home_dir: &config.home_dir,
            bootstrap_plan: BootstrapPlan::from_config(config),
            messages_between_memory: config.messages_between_memory,
            memories_between_consolidation: config.memories_between_consolidation,
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: config.reflect,
        },
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

fn codex_background_status_key(session_id: &str) -> String {
    format!("codex-session-{session_id}")
}

fn codex_background_status_message(session_id: &str) -> String {
    format!("codex session {session_id} running...")
}

fn codex_completion_followup_status_message(session_id: &str) -> String {
    format!("processing codex session {session_id} completion...")
}

fn context_refresh_summary_system_prompt(assistant_name: &str) -> String {
    format!(
        "Your job is to summarize a history of previous messages in a conversation between an AI persona and a human.\nThe conversation you are given is from a fixed context window and may not be complete.\nMessages sent by the AI are marked with the 'assistant' role.\nSummarize what happened in the conversation from the perspective of {} (use the first person).\nNote not only the content of the messages but also the context and relationship between the entities mentioned.\nAlso take note of the overall tone of the conversation.\nOnly output the summary, and keep it concise.",
        assistant_name
    )
}

fn format_context_messages_for_summary(
    messages: &[ConversationMessage],
    user_name: &str,
    assistant_name: &str,
) -> String {
    messages
        .iter()
        .filter_map(|message| match message.role {
            MessageRole::System => None,
            MessageRole::User => message
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map(|content| format!("{user_name}: {}", excerpt(content, 400))),
            MessageRole::Assistant => {
                let mut lines = Vec::new();
                if let Some(content) = message.content.as_deref().map(str::trim)
                    && !content.is_empty()
                {
                    lines.push(format!("{assistant_name}: {}", excerpt(content, 400)));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    lines.extend(tool_calls.iter().map(|call| {
                        format!(
                            "{assistant_name} Tool Call: {} {}",
                            call.name,
                            excerpt(&call.arguments_json, 200)
                        )
                    }));
                }
                (!lines.is_empty()).then_some(lines.join("\n"))
            }
            MessageRole::Tool => message
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map(|content| format!("Tool Result: {}", excerpt(content, 400))),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn summarize_context_messages_with_model(
    model: &dyn ModelClient,
    assistant_name: &str,
    user_name: &str,
    messages: &[ConversationMessage],
) -> Result<String, AppError> {
    let prompt = format_context_messages_for_summary(messages, user_name, assistant_name);
    if prompt.trim().is_empty() {
        return Err(AppError::Runtime(
            "cannot summarize empty context-refresh transcript".to_string(),
        ));
    }

    let summary = model
        .next_events(ConversationRequest {
            user_message: &prompt,
            tools: &[],
            transcript: &[ConversationMessage::new(MessageRole::User, prompt.clone())],
            force_tool: None,
        })?
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>()
        .trim()
        .to_string();

    if summary.is_empty() {
        return Err(AppError::Runtime(
            "summary model returned no assistant text".to_string(),
        ));
    }

    Ok(format!("Recent conversation summary: {summary}"))
}

fn build_context_summary_message(
    connection: &rusqlite::Connection,
    config: &AppConfig,
    dropped_messages: &[ConversationMessage],
) -> String {
    let deterministic = format_context_summary_message(dropped_messages);
    if dropped_messages.is_empty() {
        return deterministic;
    }

    let Ok(preferences) = load_user_preferences(connection, LOCAL_USER_TOKEN) else {
        return deterministic;
    };
    let Ok(provider_config) = provider_config_from_app_config(config) else {
        return deterministic;
    };
    let Ok(client) = LiveModelClient::new(provider_config) else {
        return deterministic;
    };

    let preferred_user_name = effective_user_preferred_name(preferences.as_ref());
    let user_name = if preferred_user_name.trim().is_empty() {
        "User"
    } else {
        preferred_user_name.as_str()
    };
    let assistant_name = &config.assistant_name;
    let model = LiveProviderModel::new(
        client,
        context_refresh_summary_system_prompt(assistant_name),
    );

    summarize_context_messages_with_model(&model, assistant_name, user_name, dropped_messages)
        .unwrap_or(deterministic)
}

fn restart_state() -> &'static Mutex<RestartState> {
    RESTART_STATE.get_or_init(|| Mutex::new(RestartState::default()))
}

fn enable_session_restart_support() {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.supported = true;
}

fn disable_session_restart_support() {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.supported = false;
    state.pending_resume_prompt = None;
}

fn request_session_restart(resume_prompt: &str) -> Result<(), &'static str> {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    if !state.supported {
        return Err("Session restart is not available in this Elroy runtime.");
    }
    state.pending_resume_prompt = Some(resume_prompt.to_string());
    Ok(())
}

fn consume_session_restart_request() -> Option<String> {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.pending_resume_prompt.take()
}

fn redact_secret(value: Option<&str>) -> String {
    match value {
        Some(value) if !value.trim().is_empty() => "********".to_string(),
        _ => "None (May be read from env vars)".to_string(),
    }
}

fn render_plain_text_table(title: &str, headers: &[&str], rows: &[Vec<String>]) -> String {
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

pub fn build_live_tool_registry(config: &AppConfig) -> ExecutableToolRegistry {
    build_live_tool_registry_with_codex_bin_and_hook(config, None, None)
}

fn build_live_tool_registry_with_codex_bin_and_hook(
    config: &AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook_override: Option<Arc<dyn Fn(CodexSessionResult) + Send + Sync>>,
) -> ExecutableToolRegistry {
    if !config.include_base_tools {
        return ExecutableToolRegistry::new(vec![]);
    }

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
            match request_session_restart(resume_message) {
                Ok(()) => ToolExecutionResult::success(
                    "Restart scheduled. Elroy will restart after this response completes."
                        .to_string(),
                ),
                Err(error) => ToolExecutionResult::error(error.to_string()),
            }
        },
    );

    let config_for_print_config = config.clone();
    let print_config = ExecutableTool::new(
        ToolSpec::new(
            "print_config",
            "Print the current Elroy configuration in a formatted report.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let rows = vec![
                vec![
                    "System Information".to_string(),
                    "Config Path".to_string(),
                    config_for_print_config.config_path.display().to_string(),
                ],
                vec![
                    String::new(),
                    "Home Dir".to_string(),
                    config_for_print_config.home_dir.display().to_string(),
                ],
                vec![
                    "Basic Configuration".to_string(),
                    "Default Assistant Name".to_string(),
                    config_for_print_config.assistant_name.clone(),
                ],
                vec![
                    String::new(),
                    "Reflect".to_string(),
                    config_for_print_config.reflect.to_string(),
                ],
                vec![
                    String::new(),
                    "Database URL".to_string(),
                    format!(
                        "sqlite:///{}",
                        config_for_print_config.database_path.display()
                    ),
                ],
                vec![
                    String::new(),
                    "Include Base Tools".to_string(),
                    config_for_print_config.include_base_tools.to_string(),
                ],
                vec![
                    String::new(),
                    "Exclude Tools".to_string(),
                    if config_for_print_config.exclude_tools.is_empty() {
                        "(none)".to_string()
                    } else {
                        config_for_print_config.exclude_tools.join(", ")
                    },
                ],
                vec![
                    String::new(),
                    "Async Runtime".to_string(),
                    config_for_print_config.async_runtime_enabled.to_string(),
                ],
                vec![
                    "Model Configuration".to_string(),
                    "Chat Model".to_string(),
                    config_for_print_config.chat_model.clone(),
                ],
                vec![
                    String::new(),
                    "Max Tokens".to_string(),
                    config_for_print_config.max_tokens.to_string(),
                ],
                vec![
                    String::new(),
                    "Context Refresh Target Tokens".to_string(),
                    config_for_print_config
                        .context_refresh_target_tokens()
                        .to_string(),
                ],
                vec![
                    "API Configuration".to_string(),
                    "Chat API Base".to_string(),
                    config_for_print_config.openai_base_url.clone(),
                ],
                vec![
                    String::new(),
                    "Chat API Key".to_string(),
                    redact_secret(config_for_print_config.openai_api_key.as_deref()),
                ],
                vec![
                    String::new(),
                    "Anthropic API Base".to_string(),
                    config_for_print_config.anthropic_base_url.clone(),
                ],
                vec![
                    String::new(),
                    "Anthropic API Key".to_string(),
                    redact_secret(config_for_print_config.anthropic_api_key.as_deref()),
                ],
                vec![
                    String::new(),
                    "Anthropic API Version".to_string(),
                    config_for_print_config.anthropic_api_version.clone(),
                ],
                vec![
                    "Context Management".to_string(),
                    "Assistant Greeting Enabled".to_string(),
                    config_for_print_config
                        .enable_assistant_greeting
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Min Convo Age For Greeting Minutes".to_string(),
                    config_for_print_config
                        .min_convo_age_for_greeting_minutes
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Max Context Age Minutes".to_string(),
                    config_for_print_config.max_context_age_minutes.to_string(),
                ],
                vec![
                    "Memory And Recall".to_string(),
                    "Messages Between Memory".to_string(),
                    config_for_print_config.messages_between_memory.to_string(),
                ],
                vec![
                    String::new(),
                    "Memories Between Consolidation".to_string(),
                    config_for_print_config
                        .memories_between_consolidation
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Memory Cluster Similarity".to_string(),
                    config_for_print_config
                        .memory_cluster_similarity_threshold
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "L2 Memory Relevance Distance Threshold".to_string(),
                    config_for_print_config
                        .l2_memory_relevance_distance_threshold
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Max Memory Cluster Size".to_string(),
                    config_for_print_config.max_memory_cluster_size.to_string(),
                ],
                vec![
                    String::new(),
                    "Min Memory Cluster Size".to_string(),
                    config_for_print_config.min_memory_cluster_size.to_string(),
                ],
                vec![
                    String::new(),
                    "Messages Between Self Reflection".to_string(),
                    config_for_print_config
                        .messages_between_self_reflection
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Memory Recall Classifier Enabled".to_string(),
                    config_for_print_config
                        .memory_recall_classifier_enabled
                        .to_string(),
                ],
                vec![
                    String::new(),
                    "Memory Recall Classifier Window".to_string(),
                    config_for_print_config
                        .memory_recall_classifier_window
                        .to_string(),
                ],
                vec![
                    "Paths".to_string(),
                    "Memory Dir".to_string(),
                    config_for_print_config.memory_dir.display().to_string(),
                ],
                vec![
                    String::new(),
                    "Agenda Dir".to_string(),
                    config_for_print_config.agenda_dir.display().to_string(),
                ],
            ];
            ToolExecutionResult::success(render_plain_text_table(
                "Elroy Configuration",
                &["Section", "Setting", "Value"],
                &rows,
            ))
        },
    );

    let log_dir = config.home_dir.join("logs");
    let tail_elroy_logs = ExecutableTool::new(
        ToolSpec::new(
            "tail_elroy_logs",
            "Return the last lines of the Elroy log file.",
            JsonSchema::object([("lines", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let lines = arguments
                .get("lines")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(10);
            let log_path = log_dir.join("elroy.log");
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

    let config_for_get_help = config.clone();
    let get_help = ExecutableTool::new(
        ToolSpec::new(
            "get_help",
            "Print the available system commands.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut commands = build_live_tool_registry(&config_for_get_help)
                .specs()
                .into_iter()
                .map(|spec| (spec.name, spec.description))
                .collect::<Vec<_>>();
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
            let created = (|| -> Result<PathBuf, std::io::Error> {
                let mut connection = open_sqlite_connection(&config_for_memory_write.database_path)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                run_migrations(&mut connection)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let context_messages = load_context_messages(&mut connection, LOCAL_USER_TOKEN)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let path = create_memory_file_from_context_messages(
                    &config_for_memory_write.memory_dir,
                    name,
                    text,
                    &context_messages,
                )?;
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_memory_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                record_memory_creation_and_maybe_consolidate(
                    &mut connection,
                    &BootstrapPlan::from_config(&config_for_memory_write),
                    config_for_memory_write.memories_between_consolidation,
                )
                .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            })();
            match created {
                Ok(_path) => ToolExecutionResult::success(format!("New memory created: {name}")),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create memory: {error}"))
                }
            }
        },
    );

    let config_for_consolidated_memory_write = config.clone();
    let create_consolidated_memory = ExecutableTool::new(
        ToolSpec::new(
            "create_consolidated_memory",
            "Create a consolidated memory from one or more existing active memories.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    (
                        "source_names",
                        json!({
                            "type": "array",
                            "items": {"type": "string"}
                        }),
                    ),
                ],
                ["name", "text", "source_names"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires a string name",
                );
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires string text",
                );
            };
            let Some(source_names) = arguments.get("source_names").and_then(Value::as_array) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires array source_names",
                );
            };
            let source_names = source_names
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>();
            let Some(source_names) = source_names else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires string source_names entries",
                );
            };
            if source_names.is_empty() {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires at least one source memory",
                );
            }
            match create_consolidated_memory_from_config(
                &config_for_consolidated_memory_write,
                name,
                text,
                &source_names,
            ) {
                Ok(path) => ToolExecutionResult::success(
                    json!({
                        "created": true,
                        "file_path": path.display().to_string(),
                    })
                    .to_string(),
                ),
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to create consolidated memory: {error}"
                )),
            }
        },
    );

    let get_fast_recall = ExecutableTool::new(
        ToolSpec::new(
            "get_fast_recall",
            "No-op tool used to acknowledge synthetic recall context.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| ToolExecutionResult::success("OK".to_string()),
    );

    let get_reflective_recall = ExecutableTool::new(
        ToolSpec::new(
            "get_reflective_recall",
            "No-op tool used to acknowledge synthetic reflective recall context.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| ToolExecutionResult::success("OK".to_string()),
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
            let trigger_datetime = arguments.get("trigger_datetime").and_then(Value::as_str);
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);

            match (|| -> Result<PathBuf, std::io::Error> {
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
                sync_task_context_after_mutation(&config_for_agenda_write, name, Some(name))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            })() {
                Ok(path) => ToolExecutionResult::success(format!(
                    "Agenda item added for {}: {}",
                    effective_date,
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or(name)
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

    let config_for_due_item_write = config.clone();
    let create_due_item = ExecutableTool::new(
        ToolSpec::new(
            "create_due_item",
            "Create a new file-backed due item and rebuild derived state.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("trigger_time", json!({"type": "string"})),
                    ("trigger_context", json!({"type": "string"})),
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
            if name.trim().is_empty() {
                return ToolExecutionResult::error("Due item name cannot be empty");
            }
            let trigger_time = arguments
                .get("trigger_time")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("trigger_datetime").and_then(Value::as_str));
            let trigger_context = arguments.get("trigger_context").and_then(Value::as_str);
            if trigger_time.is_none() && trigger_context.is_none() {
                return ToolExecutionResult::error(
                    "Either trigger_time or trigger_context must be provided for due items",
                );
            }
            if let Some(trigger_time) = trigger_time {
                match parse_trigger_datetime_for_validation(trigger_time) {
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
            let date = arguments.get("date").and_then(Value::as_str);
            match (|| -> Result<(), std::io::Error> {
                let mut connection =
                    open_sqlite_connection(&config_for_due_item_write.database_path)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                run_migrations(&mut connection)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                if find_active_agenda_item_by_name(&connection, name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                    .is_some()
                {
                    let item_type = if trigger_time.is_some() {
                        "Timed"
                    } else {
                        "Contextual"
                    };
                    return Err(std::io::Error::other(format!(
                        "{item_type} due item '{name}' already exists"
                    )));
                }
                let path = create_agenda_file(
                    &config_for_due_item_write.agenda_dir,
                    name,
                    text,
                    date,
                    trigger_time,
                    trigger_context,
                )?;
                let logical_due_item_name = sanitize_filename(name).replace('_', " ");
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(
                    &config_for_due_item_write,
                ))
                .map_err(|error| std::io::Error::other(error.to_string()))?;
                connection
                    .execute(
                        "UPDATE agenda_items
                         SET name = ?1
                         WHERE file_path = ?2
                           AND is_active = 1",
                        rusqlite::params![logical_due_item_name, path.to_string_lossy().as_ref()],
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let due_item = find_active_agenda_item_by_name(&connection, &logical_due_item_name)
                    .map_err(|error| std::io::Error::other(error.to_string()))?
                    .filter(|item| {
                        item.trigger_datetime.is_some() || item.trigger_context.is_some()
                    });
                if let Some(due_item) = due_item {
                    let mut transcript = load_validated_runtime_transcript(
                        &mut connection,
                        &config_for_due_item_write.assistant_name,
                        config_for_due_item_write.llm_provider() == LlmProvider::Anthropic,
                    )
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                    if !transcript_contains_context_due_item(&transcript, &due_item.name) {
                        transcript.extend(context_due_item_tool_messages(&due_item));
                        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
                            .map_err(|error| std::io::Error::other(error.to_string()))?;
                    }
                }
                Ok(())
            })() {
                Ok(_) => {
                    let message = match (trigger_time, trigger_context) {
                        (Some(trigger_time), Some(trigger_context)) => format!(
                            "Hybrid due item '{name}' has been created for {trigger_time} and context: {trigger_context}."
                        ),
                        (Some(trigger_time), None) => {
                            format!("Timed due item '{name}' has been created for {trigger_time}.")
                        }
                        (None, Some(_)) => {
                            format!("Contextual due item '{name}' has been created.")
                        }
                        (None, None) => unreachable!("validated above"),
                    };
                    ToolExecutionResult::success(message)
                }
                Err(error)
                    if error.to_string().contains(" due item '")
                        && error.to_string().ends_with(" already exists") =>
                {
                    ToolExecutionResult::error(error.to_string())
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create due item: {error}"))
                }
            }
        },
    );

    let home_dir = config.home_dir.clone();
    let list_feature_requests_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_feature_requests",
            "List the current markdown-backed feature requests.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| match list_feature_requests(&home_dir) {
            Ok(records) => ToolExecutionResult::success(feature_request_listing_content(&records)),
            Err(error) => {
                ToolExecutionResult::error(format!("failed to list feature requests: {error}"))
            }
        },
    );

    let home_dir = config.home_dir.clone();
    let make_feature_request = ExecutableTool::new(
        ToolSpec::new(
            "make_feature_request",
            "Create or merge a markdown feature request for future product work.",
            JsonSchema::object(
                [
                    ("title", json!({"type": "string"})),
                    ("description", json!({"type": "string"})),
                    ("rationale", json!({"type": "string"})),
                ],
                ["title", "description"],
            ),
        ),
        move |arguments| {
            let Some(title) = arguments.get("title").and_then(Value::as_str) else {
                return ToolExecutionResult::error("make_feature_request requires a string title");
            };
            let Some(description) = arguments.get("description").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "make_feature_request requires a string description",
                );
            };
            let title = title.trim();
            let description = description.trim();
            let rationale =
                normalize_optional_tool_string(arguments.get("rationale").and_then(Value::as_str));
            let supporting_context = build_feature_request_supporting_context(
                LOCAL_USER_TOKEN,
                title,
                description,
                rationale,
            );

            match find_best_feature_request_match(&home_dir, title, description) {
                Ok(Some(matched)) => {
                    let mut aliases = matched.record.aliases.clone();
                    if title != matched.record.title && !aliases.iter().any(|alias| alias == title)
                    {
                        aliases.push(title.to_string());
                        aliases.sort();
                    }
                    match update_feature_request(
                        &matched.record,
                        None,
                        None,
                        Some(&aliases),
                        None,
                        None,
                        Some(Some(&merge_feature_request_supporting_context(
                            matched.record.supporting_context.as_deref(),
                            &supporting_context,
                        ))),
                    ) {
                        Ok(updated) => ToolExecutionResult::success(format!(
                            "Merged into existing feature request: {} ({}; match reason: {}).",
                            updated.title,
                            updated
                                .path
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or_default(),
                            matched.reason
                        )),
                        Err(error) => ToolExecutionResult::error(format!(
                            "failed to update feature request: {error}"
                        )),
                    }
                }
                Ok(None) => match write_new_feature_request(
                    &home_dir,
                    title,
                    description,
                    rationale,
                    Some(&supporting_context),
                    "user_request",
                ) {
                    Ok(created) => ToolExecutionResult::success(format!(
                        "Created feature request: {} ({}).",
                        created.title,
                        created
                            .path
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or_default()
                    )),
                    Err(error) => ToolExecutionResult::error(format!(
                        "failed to create feature request: {error}"
                    )),
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to match feature request: {error}"))
                }
            }
        },
    );

    let home_dir = config.home_dir.clone();
    let edit_feature_request = ExecutableTool::new(
        ToolSpec::new(
            "edit_feature_request",
            "Edit an existing markdown feature request.",
            JsonSchema::object(
                [
                    ("identifier", json!({"type": "string"})),
                    ("title", json!({"type": "string"})),
                    ("description", json!({"type": "string"})),
                    ("rationale", json!({"type": "string"})),
                    ("status", json!({"type": "string"})),
                ],
                ["identifier"],
            ),
        ),
        move |arguments| {
            let Some(identifier) = arguments.get("identifier").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "edit_feature_request requires a string identifier",
                );
            };
            let title =
                normalize_optional_tool_string(arguments.get("title").and_then(Value::as_str));
            let description = normalize_optional_tool_string(
                arguments.get("description").and_then(Value::as_str),
            );
            let rationale =
                normalize_optional_tool_string(arguments.get("rationale").and_then(Value::as_str));
            let status =
                normalize_optional_tool_string(arguments.get("status").and_then(Value::as_str));

            let record = match get_feature_request(&home_dir, identifier.trim()) {
                Ok(Some(record)) => record,
                Ok(None) => {
                    return ToolExecutionResult::success(format!(
                        "Feature request '{}' not found.",
                        identifier
                    ));
                }
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load feature request: {error}"
                    ));
                }
            };
            let edit_context = format!(
                "- Edited at: {}\n- Edited by user token: {}",
                Utc::now().to_rfc3339(),
                LOCAL_USER_TOKEN
            );
            match update_feature_request(
                &record,
                title,
                status,
                None,
                description,
                Some(rationale),
                Some(Some(&merge_feature_request_supporting_context(
                    record.supporting_context.as_deref(),
                    &edit_context,
                ))),
            ) {
                Ok(updated) => ToolExecutionResult::success(format!(
                    "Updated feature request: {} ({}).",
                    updated.title,
                    updated
                        .path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or_default()
                )),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to update feature request: {error}"))
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
                        sorted_names.join(",")
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
                    mark_agenda_item_completed(path, closing_comment)?;
                    Ok(match closing_comment {
                        Some(closing_comment) => format!(
                            "Due item '{name}' has been marked as completed. Comment: {closing_comment}"
                        ),
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

    let config_for_due_item_delete = config.clone();
    let delete_due_item = ExecutableTool::new(
        ToolSpec::new(
            "delete_due_item",
            "Mark one active due item deleted.",
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
                return ToolExecutionResult::error("delete_due_item requires a string name");
            };
            let closing_comment = arguments.get("closing_comment").and_then(Value::as_str);
            let due_item_before_delete = (|| -> Result<Option<AgendaItemRecord>, String> {
                let mut connection =
                    open_sqlite_connection(&config_for_due_item_delete.database_path)
                        .map_err(|error| format!("failed to open database: {error}"))?;
                run_migrations(&mut connection)
                    .map_err(|error| format!("failed to run migrations: {error}"))?;
                let items = list_active_due_items(&connection, 1_000)
                    .map_err(|error| format!("database query failed: {error}"))?;
                Ok(items.into_iter().find(|item| item.name == name))
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
                        Some(closing_comment) => {
                            format!(
                                "Due item '{name}' has been deleted. Comment: {closing_comment}"
                            )
                        }
                        None => format!("Due item '{name}' has been deleted."),
                    })
                },
            );
            if result.is_error {
                return result;
            }
            match due_item_before_delete {
                Ok(Some(item)) => {
                    let persisted = (|| -> Result<(), String> {
                        let mut connection =
                            open_sqlite_connection(&config_for_due_item_delete.database_path)
                                .map_err(|error| format!("failed to open database: {error}"))?;
                        run_migrations(&mut connection)
                            .map_err(|error| format!("failed to run migrations: {error}"))?;
                        record_deleted_due_item_tombstone(&connection, &item, closing_comment)
                            .map_err(|error| format!("database query failed: {error}"))?;
                        Ok(())
                    })();
                    if let Err(error) = persisted {
                        return ToolExecutionResult::error(format!(
                            "failed to persist deleted due item history: {error}"
                        ));
                    }
                }
                Ok(None) => {}
                Err(error) => return ToolExecutionResult::error(error),
            }
            let tool_call_id = context_due_item_tool_call_id(name);
            match remove_context_tool_messages_by_id(&config_for_due_item_delete, &tool_call_id) {
                Ok(()) => result,
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to remove due item from context: {error}"
                )),
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

    let config_for_memory_update = config.clone();
    let update_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_memory",
            "Replace the body text of one active memory by exact name.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                ],
                ["text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
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

    let config_for_outdated_memory_update = config.clone();
    let update_outdated_or_incorrect_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_outdated_or_incorrect_memory",
            "Replace one active memory with an updated version while preserving the old content as a source.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("update_text", json!({"type": "string"})),
                ],
                ["memory_name", "update_text"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "update_outdated_or_incorrect_memory requires string memory_name",
                );
            };
            let Some(update_text) = arguments.get("update_text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "update_outdated_or_incorrect_memory requires string update_text",
                );
            };
            let mut connection =
                match open_sqlite_connection(&config_for_outdated_memory_update.database_path) {
                    Ok(connection) => connection,
                    Err(error) => {
                        return ToolExecutionResult::error(format!(
                            "failed to open database: {error}"
                        ));
                    }
                };
            let memory = match find_active_memory_by_name_in_scope(
                &connection,
                memory_name,
                &config_for_outdated_memory_update.memory_dir,
            ) {
                Ok(Some(memory)) => memory,
                Ok(None) => {
                    return ToolExecutionResult::success(format!(
                        "Memory '{memory_name}' not found"
                    ));
                }
                Err(error) => {
                    return ToolExecutionResult::error(format!("database query failed: {error}"));
                }
            };
            let path = Path::new(&memory.file_path);
            let existing = match read_memory_parts(path) {
                Ok((_, body)) => body,
                Err(error) => {
                    return ToolExecutionResult::error(format!("memory mutation failed: {error}"));
                }
            };
            let mut updated = existing;
            if !updated.is_empty() {
                updated.push_str("\n\n");
            }
            updated.push_str(&format!(
                "Update ({}):\n{}",
                Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                update_text.trim()
            ));
            let archive_dir = config_for_outdated_memory_update.memory_dir.join("archive");
            let archived_path = match archive_memory_file(path, &archive_dir) {
                Ok(path) => path,
                Err(error) => {
                    return ToolExecutionResult::error(format!("memory mutation failed: {error}"));
                }
            };
            let frontmatter = memory_source_frontmatter(&[(&memory.name, archived_path.as_path())]);
            if let Err(error) = create_memory_file_with_frontmatter(
                &config_for_outdated_memory_update.memory_dir,
                &memory.name,
                &updated,
                frontmatter.as_deref(),
            ) {
                return ToolExecutionResult::error(format!("memory mutation failed: {error}"));
            }
            if let Err(error) = elroy_db::bootstrap_database(&BootstrapPlan::from_config(
                &config_for_outdated_memory_update,
            )) {
                return ToolExecutionResult::error(format!("memory mutation failed: {error}"));
            }
            if let Err(error) = record_memory_creation_and_maybe_consolidate(
                &mut connection,
                &BootstrapPlan::from_config(&config_for_outdated_memory_update),
                config_for_outdated_memory_update.memories_between_consolidation,
            ) {
                return ToolExecutionResult::error(format!("memory mutation failed: {error}"));
            }
            ToolExecutionResult::success(format!("Memory '{memory_name}' has been updated"))
        },
    );

    let config_for_memory_archive = config.clone();
    let archive_memory = ExecutableTool::new(
        ToolSpec::new(
            "archive_memory",
            "Archive one active memory by exact name.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
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
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("note", json!({"type": "string"})),
                ],
                ["note"],
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
                return ToolExecutionResult::error("add_agenda_item_update requires string note");
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
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("closing_comment", json!({"type": "string"})),
                ],
                [] as [&str; 0],
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
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                ],
                [] as [&str; 0],
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

    let config_for_agenda_checklist_add = config.clone();
    let add_agenda_checklist_item = ExecutableTool::new(
        ToolSpec::new(
            "add_agenda_checklist_item",
            "Add a checklist item to one active agenda item.",
            JsonSchema::object(
                [
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    ("due_date", json!({"type": "string"})),
                ],
                ["text"],
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
                    Ok(due_date) => due_date,
                    Err(error) => return ToolExecutionResult::error(error),
                };
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_add,
                name,
                |path| {
                    let item_id = add_checklist_item(path, text, due_date.as_deref())?;
                    Ok(format!("Checklist item {item_id} added to '{name}'."))
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
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("checklist_item_id", json!({"type": "integer"})),
                    ("item_id", json!({"type": "integer"})),
                    ("new_text", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                ],
                [] as [&str; 0],
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
            let Some(item_id) = arguments
                .get("checklist_item_id")
                .and_then(Value::as_i64)
                .or_else(|| arguments.get("item_id").and_then(Value::as_i64))
            else {
                return ToolExecutionResult::error(
                    "edit_agenda_checklist_item requires integer item_id",
                );
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
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_edit,
                name,
                |path| {
                    let updated = match update_checklist_item(path, item_id, Some(text), None) {
                        Ok(updated) => updated,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            return Err(std::io::Error::other(format!(
                                "Agenda item '{}' has no checklist item {}.",
                                path.file_stem()
                                    .and_then(|value| value.to_str())
                                    .unwrap_or(name),
                                item_id
                            )));
                        }
                        Err(error) => return Err(error),
                    };
                    Ok(format!(
                        "Checklist item {} on '{}' updated.",
                        updated.id, name
                    ))
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
                    ("item_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("checklist_item_id", json!({"type": "integer"})),
                    ("item_id", json!({"type": "integer"})),
                ],
                [] as [&str; 0],
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
            let Some(item_id) = arguments
                .get("checklist_item_id")
                .and_then(Value::as_i64)
                .or_else(|| arguments.get("item_id").and_then(Value::as_i64))
            else {
                return ToolExecutionResult::error(
                    "complete_agenda_checklist_item requires integer item_id",
                );
            };
            mutate_agenda_file_from_config_with_result(
                &config_for_agenda_checklist_complete,
                name,
                |path| {
                    let updated = match update_checklist_item(path, item_id, None, Some(true)) {
                        Ok(updated) => updated,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            return Err(std::io::Error::other(format!(
                                "Agenda item '{}' has no checklist item {}.",
                                path.file_stem()
                                    .and_then(|value| value.to_str())
                                    .unwrap_or(name),
                                item_id
                            )));
                        }
                        Err(error) => return Err(error),
                    };
                    Ok(format!(
                        "Checklist item {} on '{}' marked as completed.",
                        updated.id, name
                    ))
                },
            )
        },
    );

    let database_path = config.database_path.clone();
    let config_for_show_context = config.clone();
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
            match load_validated_runtime_transcript(
                &mut connection,
                &config_for_show_context.assistant_name,
                config_for_show_context.llm_provider() == LlmProvider::Anthropic,
            ) {
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

    let database_path = config.database_path.clone();
    let config_for_add_memory_to_context = config.clone();
    let add_memory_to_current_context = ExecutableTool::new(
        ToolSpec::new(
            "add_memory_to_current_context",
            "Add one active memory to the persisted local conversation context.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "add_memory_to_current_context requires string memory_name",
                );
            };
            let mut connection = match open_sqlite_connection(&database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            let memory = match find_active_memory_by_name_in_scope(
                &connection,
                memory_name,
                &config_for_add_memory_to_context.memory_dir,
            ) {
                Ok(Some(memory)) => memory,
                Ok(None) => {
                    return ToolExecutionResult::success(format!(
                        "Memory '{memory_name}' not found."
                    ));
                }
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to load memory: {error}"));
                }
            };
            let mut transcript = match load_validated_runtime_transcript(
                &mut connection,
                &config_for_add_memory_to_context.assistant_name,
                config_for_add_memory_to_context.llm_provider() == LlmProvider::Anthropic,
            ) {
                Ok(messages) => messages,
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load context messages: {error}"
                    ));
                }
            };
            if !transcript_contains_context_memory(&transcript, &memory.name) {
                transcript.extend(context_memory_tool_messages(&memory));
                if let Err(error) =
                    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
                {
                    return ToolExecutionResult::error(format!(
                        "failed to persist context messages: {error}"
                    ));
                }
            }
            ToolExecutionResult::success(format!("Memory '{}' added to context.", memory.name))
        },
    );

    let database_path = config.database_path.clone();
    let config_for_drop_memory_from_context = config.clone();
    let drop_memory_from_current_context = ExecutableTool::new(
        ToolSpec::new(
            "drop_memory_from_current_context",
            "Drop one explicitly added memory from the persisted local conversation context.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "drop_memory_from_current_context requires string memory_name",
                );
            };
            let mut connection = match open_sqlite_connection(&database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            let Some(memory) = (match find_active_memory_by_name_in_scope(
                &connection,
                memory_name,
                &config_for_drop_memory_from_context.memory_dir,
            ) {
                Ok(memory) => memory,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to load memory: {error}"));
                }
            }) else {
                return ToolExecutionResult::success(format!("Memory '{memory_name}' not found."));
            };
            let transcript = match load_validated_runtime_transcript(
                &mut connection,
                &config_for_drop_memory_from_context.assistant_name,
                config_for_drop_memory_from_context.llm_provider() == LlmProvider::Anthropic,
            ) {
                Ok(messages) => messages,
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load context messages: {error}"
                    ));
                }
            };
            let context_memory_tool_call_id = context_memory_tool_call_id(&memory.name);
            let updated_transcript = transcript
                .into_iter()
                .filter(|message| {
                    !message_matches_context_memory(message, &context_memory_tool_call_id)
                })
                .collect::<Vec<_>>();
            if let Err(error) =
                replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)
            {
                return ToolExecutionResult::error(format!(
                    "failed to persist context messages: {error}"
                ));
            }
            ToolExecutionResult::success(format!("Memory '{}' dropped from context.", memory.name))
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

    let config_for_reset_context = config.clone();
    let reset_messages = ExecutableTool::new(
        ToolSpec::new(
            "reset_messages",
            "Reset the persisted context for the local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut connection =
                match open_sqlite_connection(&config_for_reset_context.database_path) {
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
            match reset_persisted_context(&mut connection, &config_for_reset_context) {
                Ok(()) => ToolExecutionResult::success("Context reset complete".to_string()),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to reset context: {error}"))
                }
            }
        },
    );

    let config_for_refresh_system = config.clone();
    let refresh_system_instructions = ExecutableTool::new(
        ToolSpec::new(
            "refresh_system_instructions",
            "Refresh the effective system instructions for the local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut connection =
                match open_sqlite_connection(&config_for_refresh_system.database_path) {
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
            match refresh_persisted_system_instructions(&mut connection, &config_for_refresh_system)
            {
                Ok(()) => {
                    ToolExecutionResult::success("System instruction refresh complete".to_string())
                }
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh system instructions: {error}"
                )),
            }
        },
    );

    let config_for_assistant_name = config.clone();
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
            mutate_user_preferences_in_config(&config_for_assistant_name, |record| {
                record.assistant_name = Some(assistant_name.to_string());
                Ok(format!("Assistant name updated to {assistant_name}."))
            })
        },
    );

    let config_for_persona = config.clone();
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
            mutate_user_preferences_in_config(&config_for_persona, |record| {
                if record.system_persona.as_deref() == Some(system_persona) {
                    return Ok("New system persona and old system persona are identical".into());
                }
                record.system_persona = Some(system_persona.to_string());
                Ok("System persona updated.".into())
            })
        },
    );

    let config_for_reset_system_persona = config.clone();
    let reset_system_persona = ExecutableTool::new(
        ToolSpec::new(
            "reset_system_persona",
            "Clear the persisted system persona for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            mutate_user_preferences_in_config(&config_for_reset_system_persona, |record| {
                record.system_persona = None;
                Ok("System persona cleared, will now use default persona.".into())
            })
        },
    );

    let config_for_preferred_name = config.clone();
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
            mutate_user_preferences_in_config(&config_for_preferred_name, |record| {
                let existing = effective_user_preferred_name(Some(record));
                if existing != elroy_user::DEFAULT_USER_PREFERRED_NAME && !override_existing {
                    return Ok(format!(
                        "Preferred name already set to {}. If this should be changed, use override_existing=True.",
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

    let config_for_full_name = config.clone();
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
            mutate_user_preferences_in_config(&config_for_full_name, |record| {
                let existing = effective_user_full_name(Some(record));
                if existing != elroy_user::UNKNOWN_FULL_NAME && !override_existing {
                    return Ok(format!(
                        "Full name already set to {}. If this should be changed, set override_existing=True.",
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
    let memory_dir_for_list_memories = config.memory_dir.clone();
    let list_memories = ExecutableTool::new(
        ToolSpec::new(
            "list_memories",
            "List active memories available to Elroy.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let memories = list_active_memories_in_scope(
                    connection,
                    &memory_dir_for_list_memories,
                    limit,
                )?;
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
    let memory_dir_for_print_memories = config.memory_dir.clone();
    let print_memories = ExecutableTool::new(
        ToolSpec::new(
            "print_memories",
            "List active memories available to Elroy.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let memories = list_active_memories_in_scope(
                    connection,
                    &memory_dir_for_print_memories,
                    limit,
                )?;
                Ok(ToolExecutionResult::success(format_memory_listing(
                    &memories,
                )))
            })
        },
    );

    let database_path = config.database_path.clone();
    let memory_dir_for_search_memories = config.memory_dir.clone();
    let provider_config_for_search_memories = provider_config_from_app_config(config).ok();
    let assistant_name_for_search_memories = config.assistant_name.clone();
    let search_memories = ExecutableTool::new(
        ToolSpec::new(
            "search_memories",
            "Search active memories by keyword.",
            JsonSchema::object([("query", json!({"type": "string"}))], ["query"]),
        ),
        move |arguments| {
            let Some(query) = arguments.get("query").and_then(Value::as_str) else {
                return ToolExecutionResult::error("search_memories requires a string query");
            };
            let recall_limit = argument_limit(&arguments, 10).min(2);
            let relevance_model = best_effort_provider_model(
                provider_config_for_search_memories.as_ref(),
                &assistant_name_for_search_memories,
            );
            with_tool_connection(&database_path, |connection| {
                let memories = search_active_memories_in_scope(
                    connection,
                    &memory_dir_for_search_memories,
                    query,
                    recall_limit * 3,
                )?;
                let due_items = list_active_due_items(connection, recall_limit * 3)?;
                let agenda_items = list_active_plain_agenda_items(connection, recall_limit * 3)?;
                let relevant_memories = select_relevant_recall_memories(
                    query,
                    &memories,
                    &HashSet::new(),
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );
                let relevant_due_items = select_relevant_recall_due_items(
                    query,
                    &due_items,
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );
                let relevant_agenda_items = select_relevant_recall_agenda_items(
                    query,
                    &agenda_items,
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );
                Ok(ToolExecutionResult::success(format_memory_search_results(
                    &relevant_memories,
                    &relevant_due_items,
                    &relevant_agenda_items,
                )))
            })
        },
    );

    let database_path = config.database_path.clone();
    let memory_dir_for_examine_memories = config.memory_dir.clone();
    let provider_config_for_examine_memories = provider_config_from_app_config(config).ok();
    let assistant_name_for_examine_memories = config.assistant_name.clone();
    let examine_memories = ExecutableTool::new(
        ToolSpec::new(
            "examine_memories",
            "Search memories and due items for the answer to a question.",
            JsonSchema::object([("question", json!({"type": "string"}))], ["question"]),
        ),
        move |arguments| {
            let Some(question) = arguments.get("question").and_then(Value::as_str) else {
                return ToolExecutionResult::error("examine_memories requires a string question");
            };
            let recall_limit = argument_limit(&arguments, 10).min(2);
            let relevance_model = best_effort_provider_model(
                provider_config_for_examine_memories.as_ref(),
                &assistant_name_for_examine_memories,
            );
            with_tool_connection(&database_path, |connection| {
                let memories = list_active_memories_in_scope(
                    connection,
                    &memory_dir_for_examine_memories,
                    recall_limit * 3,
                )?;
                let due_items = list_active_due_items(connection, recall_limit * 3)?;
                let agenda_items = list_active_plain_agenda_items(connection, recall_limit * 3)?;
                let relevant_memories = select_relevant_recall_memories(
                    question,
                    &memories,
                    &HashSet::new(),
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );
                let relevant_due_items = select_relevant_recall_due_items(
                    question,
                    &due_items,
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );
                let relevant_agenda_items = select_relevant_recall_agenda_items(
                    question,
                    &agenda_items,
                    recall_limit,
                    relevance_model
                        .as_ref()
                        .map(|model| model as &dyn ModelClient),
                );

                let mut sections = relevant_memories
                    .into_iter()
                    .map(format_memory_examination)
                    .collect::<Vec<_>>();
                sections.extend(relevant_due_items.into_iter().map(|item| {
                    let mut text = format!("# Due Item: {}\n\n{}", item.name, item.body.trim());
                    if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
                        text.push_str(&format!("\n\nScheduled for: {trigger_datetime}"));
                    }
                    if let Some(trigger_context) = item.trigger_context.as_deref() {
                        text.push_str(&format!("\nTrigger context: {trigger_context}"));
                    }
                    text
                }));
                sections.extend(relevant_agenda_items.into_iter().map(|item| {
                    let mut text = format!("# Agenda Item: {}\n\n{}", item.name, item.body.trim());
                    if let Some(agenda_date) = item.agenda_date.as_deref() {
                        text.push_str(&format!("\n\nAgenda date: {agenda_date}"));
                    }
                    if item.checklist_total > 0 {
                        text.push_str(&format!(
                            "\nChecklist progress: {}/{}",
                            item.checklist_completed, item.checklist_total
                        ));
                    }
                    text
                }));

                if sections.is_empty() {
                    Ok(ToolExecutionResult::success(
                        "No relevant memories found".to_string(),
                    ))
                } else {
                    Ok(ToolExecutionResult::success(sections.join("\n\n")))
                }
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
    let list_agenda_items = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda_items",
            "List agenda items for a given date, defaulting to today.",
            JsonSchema::object([("item_date", json!({"type": "string"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let target_date =
                match parse_agenda_item_date(arguments.get("item_date").and_then(Value::as_str)) {
                    Ok(date) => date,
                    Err(error) => return ToolExecutionResult::error(error),
                };
            with_tool_connection(&database_path, |connection| {
                let items = list_active_plain_agenda_items(connection, 1000)?
                    .into_iter()
                    .filter(|item| item.agenda_date.as_deref() == Some(target_date.as_str()))
                    .map(|item| {
                        json!({
                            "name": item.name,
                            "text": item.body,
                            "checklist_completed": item.checklist_completed,
                            "checklist_total": item.checklist_total,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    json!({
                        "item_date": target_date,
                        "items": items,
                    })
                    .to_string(),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_agenda_items_cmd = ExecutableTool::new(
        ToolSpec::new(
            "list_agenda_items_cmd",
            "List agenda items for a given date as formatted text, defaulting to today.",
            JsonSchema::object([("item_date", json!({"type": "string"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let target_date =
                match parse_agenda_item_date(arguments.get("item_date").and_then(Value::as_str)) {
                    Ok(date) => date,
                    Err(error) => return ToolExecutionResult::error(error),
                };
            with_tool_connection(&database_path, |connection| {
                let items = list_active_plain_agenda_items(connection, 1000)?
                    .into_iter()
                    .filter(|item| item.agenda_date.as_deref() == Some(target_date.as_str()))
                    .collect::<Vec<_>>();
                if items.is_empty() {
                    return Ok(ToolExecutionResult::success(format!(
                        "No agenda items for {target_date}."
                    )));
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
                    lines.push(format!("- {}: {}{}", item.name, item.body, checklist_info));
                }
                Ok(ToolExecutionResult::success(lines.join("\n")))
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
    let print_active_due_items = ExecutableTool::new(
        ToolSpec::new(
            "print_active_due_items",
            "List active due items and reminders.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_active_due_items(connection, limit)?;
                Ok(ToolExecutionResult::success(format_due_item_listing(
                    &items, true,
                )))
            })
        },
    );

    let database_path = config.database_path.clone();
    let list_inactive_due_items_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_inactive_due_items",
            "List inactive due items and reminders.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_inactive_due_items(connection, limit)?;
                let payload = items
                    .into_iter()
                    .map(|item| {
                        json!({
                            "name": item.name,
                            "agenda_date": item.agenda_date,
                            "trigger_datetime": item.trigger_datetime,
                            "trigger_context": item.trigger_context,
                            "status": item.status,
                            "closing_comment": item.closing_comment,
                            "excerpt": excerpt(&item.body, 180),
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("inactive due item payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let print_inactive_due_items = ExecutableTool::new(
        ToolSpec::new(
            "print_inactive_due_items",
            "List inactive due items and reminders.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let items = list_inactive_due_items(connection, limit)?;
                Ok(ToolExecutionResult::success(format_due_item_listing(
                    &items, false,
                )))
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

    let database_path = config.database_path.clone();
    let show_due_item = ExecutableTool::new(
        ToolSpec::new(
            "show_due_item",
            "Show one active due item by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("show_due_item requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(item) = find_active_agenda_item_by_name(connection, name)? else {
                    return Ok(ToolExecutionResult::error(format!(
                        "due item not found: {name}"
                    )));
                };
                if item.trigger_datetime.is_none() && item.trigger_context.is_none() {
                    return Ok(ToolExecutionResult::error(format!(
                        "due item not found: {name}"
                    )));
                }
                Ok(ToolExecutionResult::success(
                    json!({
                        "name": item.name,
                        "trigger_datetime": item.trigger_datetime,
                        "trigger_context": item.trigger_context,
                        "status": item.status,
                        "closing_comment": item.closing_comment,
                        "body": item.body,
                    })
                    .to_string(),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let print_due_item = ExecutableTool::new(
        ToolSpec::new(
            "print_due_item",
            "Show one active due item by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("print_due_item requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(item) = find_active_agenda_item_by_name(connection, name)? else {
                    let valid_due_items = list_active_due_items(connection, 1_000)?
                        .into_iter()
                        .map(|item| item.name)
                        .collect::<Vec<_>>();
                    return Ok(ToolExecutionResult::error(format!(
                        "Due item '{name}' not found. Valid items: {}",
                        valid_due_items.join(",")
                    )));
                };
                if item.trigger_datetime.is_none() && item.trigger_context.is_none() {
                    let valid_due_items = list_active_due_items(connection, 1_000)?
                        .into_iter()
                        .filter(|item| {
                            item.trigger_datetime.is_some() || item.trigger_context.is_some()
                        })
                        .map(|item| item.name)
                        .collect::<Vec<_>>();
                    return Ok(ToolExecutionResult::error(format!(
                        "Due item '{name}' not found. Valid items: {}",
                        valid_due_items.join(",")
                    )));
                }
                Ok(ToolExecutionResult::success(format_due_item_detail(&item)))
            })
        },
    );

    let database_path = config.database_path.clone();
    let memory_dir_for_show_memory = config.memory_dir.clone();
    let show_memory = ExecutableTool::new(
        ToolSpec::new(
            "show_memory",
            "Show one active memory by exact name.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("show_memory requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name_in_scope(
                    connection,
                    name,
                    &memory_dir_for_show_memory,
                )?
                else {
                    return Ok(ToolExecutionResult::error(format!(
                        "Memory '{name}' not found for the current user."
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
    let memory_dir_for_print_memory = config.memory_dir.clone();
    let print_memory = ExecutableTool::new(
        ToolSpec::new(
            "print_memory",
            "Show one active memory by exact name.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("print_memory requires a string name");
            };
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name_in_scope(
                    connection,
                    name,
                    &memory_dir_for_print_memory,
                )?
                else {
                    return Ok(ToolExecutionResult::success(format!(
                        "Memory '{name}' not found for the current user."
                    )));
                };
                Ok(ToolExecutionResult::success(format_memory_detail(&memory)))
            })
        },
    );

    let database_path = config.database_path.clone();
    let memory_dir_for_source_list = config.memory_dir.clone();
    let get_source_list_for_memory = ExecutableTool::new(
        ToolSpec::new(
            "get_source_list_for_memory",
            "List available sources for one active memory.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "get_source_list_for_memory requires string memory_name",
                );
            };
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name_in_scope(
                    connection,
                    memory_name,
                    &memory_dir_for_source_list,
                )?
                else {
                    return Ok(ToolExecutionResult::error(format!(
                        "Memory '{memory_name}' not found for the current user."
                    )));
                };
                let path = Path::new(&memory.file_path);
                let sources = match read_memory_parts(path) {
                    Ok((frontmatter, _)) => list_memory_sources(frontmatter.as_deref()),
                    Err(_) => Vec::new(),
                };
                Ok(ToolExecutionResult::success(
                    json!(
                        sources
                            .into_iter()
                            .map(|(source_type, name)| json!([source_type, name]))
                            .collect::<Vec<_>>()
                    )
                    .to_string(),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let memory_dir_for_source_content = config.memory_dir.clone();
    let get_source_content_for_memory = ExecutableTool::new(
        ToolSpec::new(
            "get_source_content_for_memory",
            "Show the file-backed source content for one active memory.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("index", json!({"type": "integer"})),
                ],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "get_source_content_for_memory requires string memory_name",
                );
            };
            let index = arguments.get("index").and_then(Value::as_i64).unwrap_or(0);
            if index < 0 {
                return ToolExecutionResult::error(format!(
                    "Index {index} out of range. Available indices: [0]"
                ));
            }
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name_in_scope(
                    connection,
                    memory_name,
                    &memory_dir_for_source_content,
                )?
                else {
                    return Ok(ToolExecutionResult::error(format!(
                        "Memory '{memory_name}' not found for the current user."
                    )));
                };
                let path = Path::new(&memory.file_path);
                let Ok((frontmatter, body)) = read_memory_parts(path) else {
                    return Ok(ToolExecutionResult::success(format!(
                        "No sources found for memory '{memory_name}'"
                    )));
                };
                if let Some(memory_sources) = parse_memory_sources(frontmatter.as_deref()) {
                    if index as usize >= memory_sources.len() {
                        return Ok(ToolExecutionResult::error(format!(
                            "Index {index} out of range. Available indices: {:?}",
                            (0..memory_sources.len()).collect::<Vec<_>>()
                        )));
                    }
                    let (source_name, source_path) = &memory_sources[index as usize];
                    let Ok((_, source_body)) = read_memory_parts(Path::new(source_path)) else {
                        return Ok(ToolExecutionResult::success(format!(
                            "Source not found with type: {MEMORY_SOURCE_TYPE}, name: {source_name}"
                        )));
                    };
                    return Ok(ToolExecutionResult::success(format!(
                        "# Source content for memory: {} ({} / {})\n\n{}",
                        memory.name,
                        index,
                        memory_sources.len() - 1,
                        format_memory_file_source_content(source_name, &source_body)
                    )));
                }
                if let Some(message_ids) = parse_context_message_source_ids(frontmatter.as_deref())
                {
                    if index > 0 {
                        return Ok(ToolExecutionResult::error(format!(
                            "Index {index} out of range. Available indices: [0]"
                        )));
                    }
                    let source_messages = load_messages_by_ids(connection, &message_ids)?;
                    return Ok(ToolExecutionResult::success(format!(
                        "# Source content for memory: {} (0 / 0)\n\n{}",
                        memory.name,
                        format_context_message_source_content(&source_messages)
                    )));
                }
                let _ = body;
                Ok(ToolExecutionResult::success(format!(
                    "No sources found for memory '{}'",
                    memory.name
                )))
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
                let item = match find_matching_active_agenda_item(connection, name) {
                    Ok(item) => item,
                    Err(error) => return Ok(ToolExecutionResult::error(error)),
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

    let excluded_tools = config.exclude_tools.iter().cloned().collect::<HashSet<_>>();
    let tools = vec![
        get_current_date,
        pwd,
        ls,
        read_file,
        restart_session,
        get_help,
        print_config,
        tail_elroy_logs,
        create_memory,
        create_consolidated_memory,
        get_fast_recall,
        get_reflective_recall,
        add_agenda_item,
        create_task,
        create_due_item,
        list_feature_requests_tool,
        make_feature_request,
        edit_feature_request,
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
        update_outdated_or_incorrect_memory,
        archive_memory,
        add_agenda_item_update,
        complete_agenda_item,
        delete_agenda_item,
        add_agenda_checklist_item,
        edit_agenda_checklist_item,
        complete_agenda_checklist_item,
        show_context_messages,
        add_memory_to_current_context,
        drop_memory_from_current_context,
        clear_context_messages,
        reset_messages,
        refresh_system_instructions,
        list_tasks,
        list_triggered_tasks_tool,
        list_due_tasks_tool,
        list_today_tasks_tool,
        list_memories,
        print_memories,
        search_memories,
        examine_memories,
        list_agenda,
        list_agenda_items,
        list_agenda_items_cmd,
        list_due_items,
        print_active_due_items,
        list_inactive_due_items_tool,
        print_inactive_due_items,
        show_task,
        show_due_item,
        print_due_item,
        show_memory,
        print_memory,
        get_source_list_for_memory,
        get_source_content_for_memory,
        show_agenda_item,
    ];

    ExecutableToolRegistry::new(
        tools
            .into_iter()
            .filter(|tool| !excluded_tools.contains(&tool.spec().name))
            .collect(),
    )
}

pub fn argument_limit(arguments: &Value, default_limit: usize) -> usize {
    match arguments
        .get("limit")
        .and_then(Value::as_u64)
        .or_else(|| arguments.get("n").and_then(Value::as_u64))
    {
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

fn mutate_user_preferences_in_config(
    config: &AppConfig,
    operation: impl FnOnce(&mut UserPreferenceRecord) -> rusqlite::Result<String>,
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
        refresh_persisted_system_instructions(&mut connection, config)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
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
    let memory = match find_active_memory_by_name_in_scope(&connection, name, &config.memory_dir) {
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
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
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

fn mutate_due_item_file_from_config_with_result(
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
    let memory = match find_active_memory_by_name_in_scope(&connection, name, &config.memory_dir) {
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

fn create_consolidated_memory_from_config(
    config: &AppConfig,
    name: &str,
    text: &str,
    source_names: &[&str],
) -> std::io::Result<PathBuf> {
    create_consolidated_memory_from_plan(
        &BootstrapPlan::from_config(config),
        name,
        text,
        source_names,
    )
}

fn create_consolidated_memory_from_plan(
    bootstrap_plan: &BootstrapPlan,
    name: &str,
    text: &str,
    source_names: &[&str],
) -> std::io::Result<PathBuf> {
    let mut connection = open_sqlite_connection(&bootstrap_plan.database_path)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    run_migrations(&mut connection).map_err(|error| std::io::Error::other(error.to_string()))?;

    let mut source_memories = Vec::new();
    for source_name in source_names {
        let memory = find_active_memory_by_name_in_scope(
            &connection,
            source_name,
            &bootstrap_plan.memory_dir,
        )
        .map_err(|error| std::io::Error::other(error.to_string()))?
        .ok_or_else(|| std::io::Error::other(format!("memory not found: {source_name}")))?;
        source_memories.push(memory);
    }

    let archive_dir = bootstrap_plan.memory_dir.join("archive");
    let mut archived_sources = Vec::new();
    for memory in source_memories {
        let archived_path = archive_memory_file(Path::new(&memory.file_path), &archive_dir)?;
        archived_sources.push((memory.name, archived_path));
    }

    let frontmatter = memory_source_frontmatter(
        &archived_sources
            .iter()
            .map(|(source_name, path)| (source_name.as_str(), path.as_path()))
            .collect::<Vec<_>>(),
    );
    let created = create_memory_file_with_frontmatter(
        &bootstrap_plan.memory_dir,
        name,
        text,
        frontmatter.as_deref(),
    )?;
    elroy_db::bootstrap_database(bootstrap_plan)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    Ok(created)
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

fn find_matching_active_agenda_item(
    connection: &rusqlite::Connection,
    item_name: &str,
) -> Result<AgendaItemRecord, String> {
    let items = list_active_tasks(connection, 1_000)
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

fn parse_agenda_due_date(raw: Option<&str>) -> Result<Option<String>, String> {
    match raw {
        Some(raw) => NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
            .map(|date| Some(date.format("%Y-%m-%d").to_string()))
            .map_err(|_| format!("Invalid due_date format '{raw}'. Use YYYY-MM-DD.")),
        None => Ok(None),
    }
}

fn parse_optional_line_number_argument(
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

    let lines = items
        .iter()
        .filter_map(|item| {
            let trigger_datetime = item.trigger_datetime.as_deref()?;
            let formatted_trigger_datetime = parse_sidebar_trigger_datetime(trigger_datetime)
                .map(|datetime| datetime.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| trigger_datetime.to_string());
            Some(format!(
                "⏰ DUE ITEM: '{}' - {}\n\nThis item was scheduled for {} and is now due. Please inform the user about it and then use the delete_due_item tool to remove it from active due items.",
                item.name,
                item.body,
                formatted_trigger_datetime,
            ))
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return Vec::new();
    }

    synthetic_tool_context_messages(
        "bootstrap-due-items",
        "get_due_items",
        "{}",
        lines.join("\n\n"),
    )
}

fn synthetic_tool_context_messages(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments_json: impl Into<String>,
    content: impl Into<String>,
) -> Vec<ConversationMessage> {
    let tool_call_id = tool_call_id.into();
    vec![
        ConversationMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: tool_call_id.clone(),
                name: tool_name.into(),
                arguments_json: arguments_json.into(),
            }],
        ),
        ConversationMessage::tool_result(tool_call_id, content),
    ]
}

fn context_memory_tool_call_id(name: &str) -> String {
    format!("context-memory:{}", name.to_ascii_lowercase())
}

fn context_due_item_tool_call_id(name: &str) -> String {
    format!("context-due-item:{}", name.to_ascii_lowercase())
}

fn context_task_tool_call_id(name: &str) -> String {
    format!("context-task:{}", name.to_ascii_lowercase())
}

fn find_active_memory_by_name_in_scope(
    connection: &rusqlite::Connection,
    name: &str,
    memory_dir: &Path,
) -> rusqlite::Result<Option<elroy_db::MemoryRecord>> {
    Ok(
        list_active_memories_in_scope(connection, memory_dir, 10_000)?
            .into_iter()
            .find(|memory| memory.name.eq_ignore_ascii_case(name)),
    )
}

fn list_active_memories_in_scope(
    connection: &rusqlite::Connection,
    memory_dir: &Path,
    limit: usize,
) -> rusqlite::Result<Vec<elroy_db::MemoryRecord>> {
    Ok(elroy_db::list_active_memories(connection, 10_000)?
        .into_iter()
        .filter(|memory| Path::new(&memory.file_path).starts_with(memory_dir))
        .take(limit)
        .collect())
}

fn search_active_memories_in_scope(
    connection: &rusqlite::Connection,
    memory_dir: &Path,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<elroy_db::MemoryRecord>> {
    Ok(search_active_memories(connection, query, 10_000)?
        .into_iter()
        .filter(|memory| Path::new(&memory.file_path).starts_with(memory_dir))
        .take(limit)
        .collect())
}

fn context_memory_tool_messages(memory: &elroy_db::MemoryRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&json!({
        "content": format!("MEMORY: '{}' - {}", memory.name, memory.body),
        "recall_metadata": [{
            "memory_type": "Memory",
            "memory_id": memory.id,
            "name": memory.name,
        }],
        "memories": [{
            "type": "memory",
            "name": memory.name,
            "file_path": memory.file_path,
            "excerpt": excerpt(&memory.body, 180),
            "updated_at_unix": memory.updated_at_unix,
        }],
    }))
    .expect("context-memory payload should serialize");
    synthetic_tool_context_messages(
        context_memory_tool_call_id(&memory.name),
        "get_fast_recall",
        "{}",
        content,
    )
}

fn context_due_item_tool_messages(item: &AgendaItemRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&json!({
        "content": format!("DUE ITEM: '{}' - {}", item.name, item.body),
        "recall_metadata": [{
            "memory_type": "AgendaItem",
            "memory_id": item.id,
            "name": item.name,
        }],
        "due_items": [{
            "type": "due_item",
            "name": item.name,
            "trigger_datetime": item.trigger_datetime,
            "trigger_context": item.trigger_context,
            "status": item.status,
            "closing_comment": item.closing_comment,
            "excerpt": excerpt(&item.body, 180),
        }],
    }))
    .expect("context-due-item payload should serialize");
    synthetic_tool_context_messages(
        context_due_item_tool_call_id(&item.name),
        "get_fast_recall",
        "{}",
        content,
    )
}

fn context_task_tool_messages(item: &AgendaItemRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&json!({
        "content": format!("TASK: '{}' - {}", item.name, item.body),
        "recall_metadata": [{
            "memory_type": "AgendaItem",
            "memory_id": item.id,
            "name": item.name,
        }],
        "tasks": [{
            "type": "task",
            "name": item.name,
            "agenda_date": item.agenda_date,
            "trigger_datetime": item.trigger_datetime,
            "trigger_context": item.trigger_context,
            "status": item.status,
            "closing_comment": item.closing_comment,
            "excerpt": excerpt(&item.body, 180),
        }],
    }))
    .expect("context-task payload should serialize");
    synthetic_tool_context_messages(
        context_task_tool_call_id(&item.name),
        "get_fast_recall",
        "{}",
        content,
    )
}

fn transcript_contains_context_memory(
    transcript: &[ConversationMessage],
    memory_name: &str,
) -> bool {
    let tool_call_id = context_memory_tool_call_id(memory_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

fn transcript_contains_context_due_item(
    transcript: &[ConversationMessage],
    due_item_name: &str,
) -> bool {
    let tool_call_id = context_due_item_tool_call_id(due_item_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

fn transcript_contains_context_task(transcript: &[ConversationMessage], task_name: &str) -> bool {
    let tool_call_id = context_task_tool_call_id(task_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

fn message_matches_context_memory(message: &ConversationMessage, tool_call_id: &str) -> bool {
    message_matches_tool_call_id(message, tool_call_id)
}

fn message_matches_tool_call_id(message: &ConversationMessage, tool_call_id: &str) -> bool {
    message.tool_call_id.as_deref() == Some(tool_call_id)
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| tool_calls.iter().any(|call| call.id == tool_call_id))
}

fn remove_context_tool_messages_by_id(
    config: &AppConfig,
    tool_call_id: &str,
) -> Result<(), AppError> {
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

fn sync_task_context_after_mutation(
    config: &AppConfig,
    old_name: &str,
    current_name: Option<&str>,
) -> Result<(), AppError> {
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

fn sync_due_item_context_after_mutation(
    config: &AppConfig,
    old_name: &str,
    current_name: Option<&str>,
) -> Result<(), AppError> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let transcript = load_validated_runtime_transcript(
        &mut connection,
        &config.assistant_name,
        config.llm_provider() == LlmProvider::Anthropic,
    )?;
    let old_tool_call_id = context_due_item_tool_call_id(old_name);
    let mut updated_transcript = transcript
        .into_iter()
        .filter(|message| !message_matches_tool_call_id(message, &old_tool_call_id))
        .collect::<Vec<_>>();

    if let Some(current_name) = current_name
        && let Some(due_item) = find_active_agenda_item_by_name(&connection, current_name)?
            .filter(|item| item.trigger_datetime.is_some() || item.trigger_context.is_some())
        && !transcript_contains_context_due_item(&updated_transcript, &due_item.name)
    {
        updated_transcript.extend(context_due_item_tool_messages(&due_item));
    }

    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

fn format_context_summary_message(messages: &[ConversationMessage]) -> String {
    let lines = messages
        .iter()
        .filter_map(|message| {
            let mut parts = Vec::new();
            let timestamp = format_context_summary_timestamp(message.created_at_unix);
            match message.role {
                MessageRole::System => return None,
                MessageRole::User => {
                    let content = message.content.as_deref()?.trim();
                    if content.is_empty() {
                        return None;
                    }
                    parts.push(format!("User ({timestamp}): {}", excerpt(content, 160)));
                }
                MessageRole::Assistant => {
                    if let Some(content) = message.content.as_deref().map(str::trim)
                        && !content.is_empty()
                    {
                        parts.push(format!(
                            "Assistant ({timestamp}): {}",
                            excerpt(content, 160)
                        ));
                    }
                    if let Some(tool_calls) = &message.tool_calls {
                        parts.extend(tool_calls.iter().map(|call| {
                            format!(
                                "Assistant Tool Call ({timestamp}): {} {}",
                                call.name,
                                excerpt(&call.arguments_json, 120)
                            )
                        }));
                    }
                }
                MessageRole::Tool => {
                    let content = message.content.as_deref()?.trim();
                    if content.is_empty() {
                        return None;
                    }
                    parts.push(format!(
                        "Tool Result ({timestamp}): {}",
                        excerpt(content, 160)
                    ));
                }
            }

            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        })
        .collect::<Vec<_>>();
    let conversation_range = messages
        .iter()
        .filter(|message| message.role == MessageRole::User)
        .map(|message| message.created_at_unix)
        .collect::<Vec<_>>();
    let summary_lines = if let (Some(min), Some(max)) = (
        conversation_range.iter().min().copied(),
        conversation_range.iter().max().copied(),
    ) {
        let mut lines_with_range = lines;
        lines_with_range.push(format!(
            "Messages from {} to {}",
            format_context_summary_timestamp(min),
            format_context_summary_timestamp(max)
        ));
        lines_with_range
    } else {
        lines
    };

    if summary_lines.is_empty() {
        "Recent conversation summary: (No earlier conversation summary available.)".to_string()
    } else {
        format!("Recent conversation summary: {}", summary_lines.join("\n"))
    }
}

fn format_context_summary_timestamp(unix_seconds: i64) -> String {
    Local
        .timestamp_opt(unix_seconds, 0)
        .single()
        .unwrap_or_else(Local::now)
        .format("%A, %B %d, %Y %I:%M %p %Z")
        .to_string()
}

struct RecallContext<'a> {
    transcript: &'a [ConversationMessage],
    memories: &'a [elroy_db::MemoryRecord],
    due_items: &'a [AgendaItemRecord],
    agenda_items: &'a [AgendaItemRecord],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoryRecallDecision {
    needs_recall: bool,
    reasoning: String,
    used_llm: bool,
}

fn recall_memory_context_messages_with_decision(
    memory_recall_classifier_window: usize,
    reflect: bool,
    prompt: &str,
    should_recall: bool,
    reflective_model: Option<&dyn ModelClient>,
    context: RecallContext<'_>,
) -> Vec<ConversationMessage> {
    if !should_recall {
        return Vec::new();
    }

    let recall_query =
        build_recall_query(prompt, context.transcript, memory_recall_classifier_window);
    let relevance_model = if reflect { None } else { reflective_model };
    let already_recalled_memories = recalled_item_names_by_type(context.transcript, "Memory");
    let recalled = select_relevant_recall_memories(
        &recall_query,
        context.memories,
        &already_recalled_memories,
        2,
        relevance_model,
    );
    let already_recalled_agenda_items =
        recalled_item_names_by_type(context.transcript, "AgendaItem");
    let fast_due_items = if reflect {
        Vec::new()
    } else {
        select_relevant_recall_due_items(&recall_query, context.due_items, 2, relevance_model)
            .into_iter()
            .filter(|item| !already_recalled_agenda_items.contains(&item.name))
            .collect()
    };
    let fast_agenda_items = if reflect {
        Vec::new()
    } else {
        select_relevant_recall_agenda_items(&recall_query, context.agenda_items, 2, relevance_model)
            .into_iter()
            .filter(|item| !already_recalled_agenda_items.contains(&item.name))
            .collect()
    };
    let reflective_due_items = if reflect {
        select_due_items_by_overlap(&recall_query, context.due_items, 2, None)
            .into_iter()
            .filter(|item| !already_recalled_agenda_items.contains(&item.name))
            .collect()
    } else {
        Vec::new()
    };
    let reflective_agenda_items = if reflect {
        select_agenda_items_by_overlap(&recall_query, context.agenda_items, 2)
            .into_iter()
            .filter(|item| !already_recalled_agenda_items.contains(&item.name))
            .collect()
    } else {
        Vec::new()
    };
    if recalled.is_empty()
        && fast_due_items.is_empty()
        && fast_agenda_items.is_empty()
        && reflective_due_items.is_empty()
        && reflective_agenda_items.is_empty()
    {
        return Vec::new();
    }

    if reflect {
        let recent_context = recent_recall_context(context.transcript, 3);
        let fallback_content = build_reflective_recall_content(
            &recalled,
            &reflective_due_items,
            &reflective_agenda_items,
            prompt,
            &recent_context,
        );
        let Some(content) = build_reflective_recall_content_with_model(
            reflective_model,
            &recalled,
            &reflective_due_items,
            &reflective_agenda_items,
            prompt,
            &recent_context,
            &fallback_content,
        ) else {
            return Vec::new();
        };
        let content = serde_json::to_string_pretty(&json!({
            "content": content,
            "recall_metadata": recalled.iter().map(|memory| {
                json!({
                    "memory_type": "Memory",
                    "memory_id": memory.id,
                    "name": memory.name,
                })
            }).chain(reflective_due_items.iter().map(|item| {
                json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            })).chain(reflective_agenda_items.iter().map(|item| {
                json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            })).collect::<Vec<_>>(),
        }))
        .expect("reflective memory recall payload should serialize");

        return synthetic_tool_context_messages(
            "bootstrap-memory-recall",
            "get_reflective_recall",
            "{}",
            content,
        );
    }

    let content = serde_json::to_string_pretty(&json!({
        "content": recalled
            .iter()
            .map(|memory| format_memory_detail(memory))
            .chain(fast_due_items.iter().map(|item| format_agenda_item_recall_detail(item)))
            .chain(fast_agenda_items.iter().map(|item| format_agenda_item_recall_detail(item)))
            .collect::<Vec<_>>()
            .join("\n\n"),
        "recall_metadata": recalled
            .iter()
            .map(|memory| {
                json!({
                    "memory_type": "Memory",
                    "memory_id": memory.id,
                    "name": memory.name,
                })
            })
            .chain(fast_due_items.iter().map(|item| {
                json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            }))
            .chain(fast_agenda_items.iter().map(|item| {
                json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            }))
            .collect::<Vec<_>>(),
    }))
    .expect("memory recall payload should serialize");

    synthetic_tool_context_messages("bootstrap-memory-recall", "get_fast_recall", "{}", content)
}

#[cfg(test)]
fn recall_memory_context_messages(
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
    reflect: bool,
    prompt: &str,
    context: RecallContext<'_>,
) -> Vec<ConversationMessage> {
    let should_recall = !memory_recall_classifier_enabled || !should_skip_memory_recall(prompt);
    recall_memory_context_messages_with_decision(
        memory_recall_classifier_window,
        reflect,
        prompt,
        should_recall,
        None,
        context,
    )
}

fn recall_due_item_context_messages(
    prompt: &str,
    transcript: &[ConversationMessage],
    due_items: &[AgendaItemRecord],
    now_iso: &str,
) -> Vec<ConversationMessage> {
    let recall_query = build_recall_query(prompt, transcript, 6);
    let recalled = select_recalled_due_items(&recall_query, due_items, now_iso, 2);
    if recalled.is_empty() {
        return Vec::new();
    }
    recalled
        .into_iter()
        .filter(|item| !transcript_contains_context_due_item(transcript, &item.name))
        .flat_map(context_due_item_tool_messages)
        .collect()
}

fn memory_recall_status_updates_with_decision(
    used_llm_classifier: bool,
    fetched_memories: bool,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    if used_llm_classifier {
        events.push(StreamEvent::StatusUpdate {
            content: "classifying recall...".to_string(),
        });
    }
    if fetched_memories {
        events.push(StreamEvent::StatusUpdate {
            content: "fetching memories...".to_string(),
        });
    }
    events
}

#[cfg(test)]
fn memory_recall_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
) -> Vec<StreamEvent> {
    let used_llm_classifier =
        memory_recall_classifier_enabled && !should_skip_memory_recall(prompt);
    memory_recall_status_updates_with_decision(used_llm_classifier, fetched_memories)
}

fn prompt_prelude_status_updates_with_decision(
    used_llm_classifier: bool,
    fetched_memories: bool,
    surfaced_due_items: bool,
) -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::StatusUpdate {
        content: "loading context...".to_string(),
    }];
    events.extend(memory_recall_status_updates_with_decision(
        used_llm_classifier,
        fetched_memories,
    ));
    if surfaced_due_items {
        events.push(StreamEvent::StatusUpdate {
            content: "surfacing due items...".to_string(),
        });
    }
    events.push(StreamEvent::StatusUpdate {
        content: "thinking...".to_string(),
    });
    events
}

#[cfg(test)]
fn prompt_prelude_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
    surfaced_due_items: bool,
) -> Vec<StreamEvent> {
    let used_llm_classifier =
        memory_recall_classifier_enabled && !should_skip_memory_recall(prompt);
    prompt_prelude_status_updates_with_decision(
        used_llm_classifier,
        fetched_memories,
        surfaced_due_items,
    )
}

fn should_skip_memory_recall(prompt: &str) -> bool {
    let normalized = prompt
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
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
    const CLARIFICATIONS: &[&str] = &["what", "huh", "pardon", "sorry", "excuse me"];

    (normalized.len() < 10 && SIMPLE_SHORT.contains(&normalized.as_str()))
        || GREETINGS.contains(&normalized.as_str())
        || CLARIFICATIONS.contains(&normalized.as_str())
}

fn parse_memory_recall_decision(response: &str) -> Option<(bool, String)> {
    let trimmed = response.trim();
    let json_text = if trimmed.starts_with("```") {
        trimmed
            .lines()
            .skip(1)
            .take_while(|line| !line.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        trimmed.to_string()
    };
    let value: Value = serde_json::from_str(json_text.trim()).ok()?;
    let needs_recall = value.get("needs_recall")?.as_bool()?;
    let reasoning = value
        .get("reasoning")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Some((needs_recall, reasoning))
}

fn parse_relevance_filter_response(response: &str) -> Option<Vec<bool>> {
    let trimmed = response.trim();
    let json_text = if trimmed.starts_with("```") {
        trimmed
            .lines()
            .skip(1)
            .take_while(|line| !line.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        trimmed.to_string()
    };
    let value: Value = serde_json::from_str(json_text.trim()).ok()?;
    value
        .get("answers")?
        .as_array()?
        .iter()
        .map(Value::as_bool)
        .collect()
}

fn filter_candidates_for_relevance<'a, T>(
    model: Option<&dyn ModelClient>,
    query: &str,
    candidates: Vec<&'a T>,
    extraction_fn: impl Fn(&T) -> String,
) -> Vec<&'a T> {
    let Some(model) = model else {
        return candidates;
    };
    if candidates.is_empty() {
        return candidates;
    }

    let responses = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| format!("{index}. {}", extraction_fn(candidate)))
        .collect::<Vec<_>>()
        .join("\n\n");
    let prompt = format!(
        "Your job is to determine which candidate recall items are relevant to a query.\n\
Return exactly one JSON object with keys `answers` (array of booleans, one per candidate, in order) \
and `reasoning` (string).\n\n\
Query: {query}\nResponses:\n{responses}"
    );
    let Ok(events) = model.next_events(ConversationRequest {
        user_message: &prompt,
        tools: &[],
        transcript: &[ConversationMessage::new(MessageRole::User, prompt.clone())],
        force_tool: None,
    }) else {
        return candidates;
    };
    let response = events
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>();
    let Some(answers) = parse_relevance_filter_response(&response) else {
        return candidates;
    };
    if answers.len() != candidates.len() {
        return candidates;
    }

    candidates
        .into_iter()
        .zip(answers)
        .filter_map(|(candidate, is_relevant)| is_relevant.then_some(candidate))
        .collect()
}

fn select_relevant_recall_memories<'a>(
    query: &str,
    memories: &'a [MemoryRecord],
    already_recalled: &HashSet<String>,
    limit: usize,
    relevance_model: Option<&dyn ModelClient>,
) -> Vec<&'a MemoryRecord> {
    let candidate_limit = limit.saturating_mul(3).max(limit);
    let overlap_candidates =
        select_recalled_memories(query, memories, already_recalled, candidate_limit);
    let candidates = if overlap_candidates.is_empty() && relevance_model.is_some() {
        recent_memory_candidates(memories, already_recalled, candidate_limit)
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(relevance_model, query, candidates, |memory| {
        format!("# {}\n{}", memory.name, memory.body.trim())
    })
    .into_iter()
    .take(limit)
    .collect()
}

fn select_relevant_recall_due_items<'a>(
    query: &str,
    due_items: &'a [AgendaItemRecord],
    limit: usize,
    relevance_model: Option<&dyn ModelClient>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = limit.saturating_mul(3).max(limit);
    let overlap_candidates = select_due_items_by_overlap(query, due_items, candidate_limit, None);
    let candidates = if overlap_candidates.is_empty() && relevance_model.is_some() {
        recent_due_item_candidates(due_items, candidate_limit)
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(relevance_model, query, candidates, |item| {
        let mut text = format!("# {}\n{}", item.name, item.body.trim());
        if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
            text.push_str(&format!("\ntrigger_datetime: {trigger_datetime}"));
        }
        if let Some(trigger_context) = item.trigger_context.as_deref() {
            text.push_str(&format!("\ntrigger_context: {trigger_context}"));
        }
        text
    })
    .into_iter()
    .take(limit)
    .collect()
}

fn select_relevant_recall_agenda_items<'a>(
    query: &str,
    agenda_items: &'a [AgendaItemRecord],
    limit: usize,
    relevance_model: Option<&dyn ModelClient>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = limit.saturating_mul(3).max(limit);
    let overlap_candidates = select_agenda_items_by_overlap(query, agenda_items, candidate_limit);
    let candidates = if overlap_candidates.is_empty() && relevance_model.is_some() {
        recent_agenda_item_candidates(agenda_items, candidate_limit)
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(relevance_model, query, candidates, |item| {
        format!("# {}\n{}", item.name, item.body.trim())
    })
    .into_iter()
    .take(limit)
    .collect()
}

fn recent_memory_candidates<'a>(
    memories: &'a [MemoryRecord],
    already_recalled: &HashSet<String>,
    limit: usize,
) -> Vec<&'a MemoryRecord> {
    let mut candidates = memories
        .iter()
        .filter(|memory| !already_recalled.contains(&memory.name.to_ascii_lowercase()))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .updated_at_unix
            .cmp(&left.updated_at_unix)
            .then_with(|| left.name.cmp(&right.name))
    });
    candidates.into_iter().take(limit).collect()
}

fn recent_due_item_candidates(
    due_items: &[AgendaItemRecord],
    limit: usize,
) -> Vec<&AgendaItemRecord> {
    let mut candidates = due_items
        .iter()
        .filter(|item| item.trigger_datetime.is_some() || item.trigger_context.is_some())
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .updated_at_unix
            .cmp(&left.updated_at_unix)
            .then_with(|| left.name.cmp(&right.name))
    });
    candidates.into_iter().take(limit).collect()
}

fn recent_agenda_item_candidates(
    agenda_items: &[AgendaItemRecord],
    limit: usize,
) -> Vec<&AgendaItemRecord> {
    let mut candidates = agenda_items
        .iter()
        .filter(|item| item.trigger_datetime.is_none() && item.trigger_context.is_none())
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .updated_at_unix
            .cmp(&left.updated_at_unix)
            .then_with(|| left.name.cmp(&right.name))
    });
    candidates.into_iter().take(limit).collect()
}

fn classify_memory_recall_with_model(
    model: &dyn ModelClient,
    current_message: &str,
    recent_messages: &[ConversationMessage],
    window_size: usize,
) -> Result<MemoryRecallDecision, AppError> {
    let conversation_context = recent_recall_context(recent_messages, window_size).join("\n");
    let prompt = format!(
        "Analyze if this message requires recalling information from long-term memory (including due items).\n\nRecent conversation:\n{conversation_context}\n\nCurrent message: {current_message}\n\nMemory recall is NEEDED if (almost always):\n- Message mentions ANY specific topic, activity, person, place, or thing\n- Message references ANY past topics, events, or context\n- Message contains substantive content beyond pure acknowledgment\n- Message mentions activities, hobbies, tasks, or appointments that commonly have due items\n- Message is a follow-up question or statement\n- Message asks about preferences, goals, or history\n- When in doubt - ALWAYS prefer recall\n\nMemory recall is NOT needed ONLY if:\n- Message is ONLY a simple greeting with no other content (hi, hello, bye)\n- Message is ONLY a simple acknowledgment with no other content (ok, thanks, yes, no)\n- Message is ONLY a clarification question with no topic content (what?, huh?)\n\nCRITICAL: If the message mentions ANY topic, activity, or substantive content, memory recall is NEEDED because there may be relevant due items or memories. Be VERY conservative - prefer false positives over false negatives.\n\nReturn exactly one JSON object with keys `needs_recall` (boolean) and `reasoning` (string)."
    );
    let response = model
        .next_events(ConversationRequest {
            user_message: &prompt,
            tools: &[],
            transcript: &[ConversationMessage::new(MessageRole::User, prompt.clone())],
            force_tool: None,
        })?
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>();
    let Some((needs_recall, reasoning)) = parse_memory_recall_decision(&response) else {
        return Err(AppError::Runtime(
            "memory recall classifier returned invalid JSON".to_string(),
        ));
    };
    Ok(MemoryRecallDecision {
        needs_recall,
        reasoning,
        used_llm: true,
    })
}

fn determine_memory_recall_decision(
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
    prompt: &str,
    transcript: &[ConversationMessage],
    classifier_model: Option<&dyn ModelClient>,
) -> MemoryRecallDecision {
    if !memory_recall_classifier_enabled {
        return MemoryRecallDecision {
            needs_recall: true,
            reasoning: "classifier disabled".to_string(),
            used_llm: false,
        };
    }
    if should_skip_memory_recall(prompt) {
        return MemoryRecallDecision {
            needs_recall: false,
            reasoning: "Simple greeting/acknowledgment/clarification detected by heuristic"
                .to_string(),
            used_llm: false,
        };
    }
    if let Some(model) = classifier_model
        && let Ok(decision) = classify_memory_recall_with_model(
            model,
            prompt,
            transcript,
            memory_recall_classifier_window,
        )
    {
        return decision;
    }
    MemoryRecallDecision {
        needs_recall: true,
        reasoning: "classifier unavailable; falling back to conservative recall".to_string(),
        used_llm: false,
    }
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
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                _ => return None,
            };
            message
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map(|content| format!("{role}: {content}"))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn build_reflective_recall_content(
    memories: &[&elroy_db::MemoryRecord],
    due_items: &[&AgendaItemRecord],
    agenda_items: &[&AgendaItemRecord],
    prompt: &str,
    recent_context: &[String],
) -> String {
    let mut sections = Vec::new();
    if !memories.is_empty() {
        let memory_lines = memories
            .iter()
            .map(|memory| format!("- {}: {}", memory.name, excerpt(&memory.body, 180)))
            .collect::<Vec<_>>();
        sections.push(format!(
            "I remember these memory details may be relevant to the current conversation:\n{}",
            memory_lines.join("\n")
        ));
    }
    if !due_items.is_empty() {
        let due_item_lines = due_items
            .iter()
            .map(|item| {
                let mut line = format!("- {}: {}", item.name, excerpt(&item.body, 180));
                if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
                    line.push_str(&format!(" (scheduled for {trigger_datetime})"));
                }
                if let Some(trigger_context) = item.trigger_context.as_deref() {
                    line.push_str(&format!(" (trigger context: {trigger_context})"));
                }
                line
            })
            .collect::<Vec<_>>();
        sections.push(format!(
            "I also recall these due items may matter:\n{}",
            due_item_lines.join("\n")
        ));
    }
    if !agenda_items.is_empty() {
        let agenda_item_lines = agenda_items
            .iter()
            .map(|item| format!("- {}: {}", item.name, excerpt(&item.body, 180)))
            .collect::<Vec<_>>();
        sections.push(format!(
            "I also recall these agenda items may matter:\n{}",
            agenda_item_lines.join("\n")
        ));
    }
    if !recent_context.is_empty() {
        sections.push(format!(
            "Recent conversation context:\n{}",
            recent_context.join("\n")
        ));
    }
    sections.push(format!(
        "The latest user message is: {}",
        excerpt(prompt.trim(), 160)
    ));
    sections.push(
        "I should use the recalled details only if they help answer the user clearly.".to_string(),
    );
    sections.join("\n\n")
}

fn build_reflective_recall_prompt(
    memories: &[&elroy_db::MemoryRecord],
    due_items: &[&AgendaItemRecord],
    agenda_items: &[&AgendaItemRecord],
    prompt: &str,
    recent_context: &[String],
) -> String {
    let recalled_memory_facts = memories
        .iter()
        .map(|memory| format!("# {}\n{}", memory.name, memory.body.trim()))
        .chain(due_items.iter().map(|item| {
            let mut fact = format!("# {}\n{}", item.name, item.body.trim());
            if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
                fact.push_str(&format!("\ntrigger_datetime: {trigger_datetime}"));
            }
            if let Some(trigger_context) = item.trigger_context.as_deref() {
                fact.push_str(&format!("\ntrigger_context: {trigger_context}"));
            }
            fact
        }))
        .chain(
            agenda_items
                .iter()
                .map(|item| format!("# {}\n{}", item.name, item.body.trim())),
        )
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut body = String::from("Recalled Memory Content\n\n");
    body.push_str(&recalled_memory_facts);
    body.push_str("\n\n#Conversation Transcript:\n");
    if recent_context.is_empty() {
        body.push_str(&format!("user: {}", prompt.trim()));
    } else {
        body.push_str(&recent_context.join("\n"));
        body.push('\n');
        body.push_str(&format!("user: {}", prompt.trim()));
    }
    body
}

fn parse_reflective_recall_model_response(response: &str) -> Option<(bool, Option<String>)> {
    let trimmed = response.trim();
    let json_text = if trimmed.starts_with("```") {
        trimmed
            .lines()
            .skip(1)
            .take_while(|line| !line.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        trimmed.to_string()
    };
    let value: Value = serde_json::from_str(json_text.trim()).ok()?;
    let is_relevant = value.get("is_relevant")?.as_bool()?;
    let content = value
        .get("content")
        .and_then(|content| content.as_str().map(str::trim).map(str::to_string));
    Some((is_relevant, content))
}

fn build_reflective_recall_content_with_model(
    model: Option<&dyn ModelClient>,
    memories: &[&elroy_db::MemoryRecord],
    due_items: &[&AgendaItemRecord],
    agenda_items: &[&AgendaItemRecord],
    prompt: &str,
    recent_context: &[String],
    fallback_content: &str,
) -> Option<String> {
    let Some(model) = model else {
        return Some(fallback_content.to_string());
    };
    let model_prompt =
        build_reflective_recall_prompt(memories, due_items, agenda_items, prompt, recent_context);
    let response = model
        .next_events(ConversationRequest {
            user_message: &model_prompt,
            tools: &[],
            transcript: &[ConversationMessage::new(
                MessageRole::User,
                model_prompt.clone(),
            )],
            force_tool: None,
        })
        .ok()?
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>();
    match parse_reflective_recall_model_response(&response) {
        Some((false, _)) => None,
        Some((true, Some(content))) if !content.trim().is_empty() => Some(content),
        _ => Some(fallback_content.to_string()),
    }
}

#[cfg(test)]
fn recalled_memory_names(transcript: &[ConversationMessage]) -> HashSet<String> {
    recalled_item_names_by_type(transcript, "Memory")
}

fn recalled_item_names_by_type(
    transcript: &[ConversationMessage],
    memory_type: &str,
) -> HashSet<String> {
    transcript
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .filter_map(|message| message.content.as_deref())
        .flat_map(|content| parse_recalled_item_names(content, memory_type))
        .collect()
}

fn select_due_items_by_overlap<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    limit: usize,
    skip_time_due_before: Option<&str>,
) -> Vec<&'a AgendaItemRecord> {
    let prompt_tokens = significant_tokens(prompt);
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored = due_items
        .iter()
        .filter_map(|item| {
            let trigger_context = item.trigger_context.as_deref()?;
            if skip_time_due_before.is_some_and(|now_iso| {
                item.trigger_datetime
                    .as_deref()
                    .is_some_and(|trigger_datetime| trigger_datetime <= now_iso)
            }) {
                return None;
            }

            let mut haystack = String::with_capacity(
                item.name.len() + item.body.len() + trigger_context.len() + 2,
            );
            haystack.push_str(&item.name);
            haystack.push(' ');
            haystack.push_str(&item.body);
            haystack.push(' ');
            haystack.push_str(trigger_context);
            let due_item_tokens = significant_tokens(&haystack);
            let overlap = prompt_tokens.intersection(&due_item_tokens).count();
            (overlap > 0).then_some((overlap, item.updated_at_unix, item))
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
        .map(|(_, _, item)| item)
        .collect()
}

fn select_agenda_items_by_overlap<'a>(
    prompt: &str,
    agenda_items: &'a [AgendaItemRecord],
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    let prompt_tokens = significant_tokens(prompt);
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored = agenda_items
        .iter()
        .filter_map(|item| {
            let mut haystack = String::with_capacity(
                item.name.len()
                    + item.body.len()
                    + item.agenda_date.as_deref().map_or(0, str::len)
                    + 2,
            );
            haystack.push_str(&item.name);
            haystack.push(' ');
            haystack.push_str(&item.body);
            if let Some(agenda_date) = item.agenda_date.as_deref() {
                haystack.push(' ');
                haystack.push_str(agenda_date);
            }
            let agenda_item_tokens = significant_tokens(&haystack);
            let overlap = prompt_tokens.intersection(&agenda_item_tokens).count();
            (overlap > 0).then_some((overlap, item.updated_at_unix, item))
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
        .map(|(_, _, item)| item)
        .collect()
}

fn select_recalled_due_items<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    select_due_items_by_overlap(prompt, due_items, limit, Some(now_iso))
}

#[cfg(test)]
fn parse_recalled_memory_names(content: &str) -> Vec<String> {
    parse_recalled_item_names(content, "Memory")
}

fn parse_recalled_item_names(content: &str, desired_memory_type: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
        return Vec::new();
    };

    if let Some(items) = value.as_array() {
        return items
            .iter()
            .filter_map(|item| {
                item.get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| name.to_ascii_lowercase())
            })
            .collect();
    }

    value
        .get("recall_metadata")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let memory_type = item
                .get("memory_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("Memory");
            (memory_type == desired_memory_type).then(|| {
                item.get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| name.to_ascii_lowercase())
            })?
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

fn approximate_message_token_count(message: &ConversationMessage) -> usize {
    let content_tokens = message
        .content
        .as_deref()
        .map(|content| content.split_whitespace().count())
        .unwrap_or(0);
    let tool_call_tokens = message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    call.name.split_whitespace().count()
                        + call.arguments_json.split_whitespace().count()
                })
                .sum::<usize>()
        })
        .unwrap_or(0);
    let tool_result_tokens = message
        .tool_call_id
        .as_deref()
        .map(|tool_call_id| tool_call_id.split_whitespace().count())
        .unwrap_or(0);

    content_tokens + tool_call_tokens + tool_result_tokens
}

fn count_context_tokens(context_messages: &[ConversationMessage]) -> usize {
    context_messages
        .iter()
        .map(approximate_message_token_count)
        .sum()
}

fn is_context_refresh_needed(context_messages: &[ConversationMessage], max_tokens: usize) -> bool {
    if !context_messages
        .iter()
        .any(|message| message.role == MessageRole::User)
    {
        return false;
    }

    count_context_tokens(context_messages) > max_tokens
}

fn compress_context_messages(
    context_messages: &[ConversationMessage],
    context_refresh_target_tokens: usize,
    max_context_age_minutes: f64,
) -> Vec<ConversationMessage> {
    if context_messages.is_empty() {
        return Vec::new();
    }

    let system_message = context_messages[0].clone();
    let previous_messages = &context_messages[1..];
    if previous_messages.is_empty() {
        return vec![system_message];
    }

    let system_tokens = approximate_message_token_count(&system_message);
    let remaining_budget = context_refresh_target_tokens.saturating_sub(system_tokens);
    let cutoff_unix = Utc::now().timestamp() - (max_context_age_minutes * 60.0) as i64;

    let mut cutoff_index = 0usize;
    let mut current_token_count = 0usize;
    let mut idx = previous_messages.len();

    while idx > 0 {
        idx -= 1;
        let message = &previous_messages[idx];

        if message.role == MessageRole::Tool
            && idx > 0
            && previous_messages[idx - 1].role == MessageRole::Assistant
        {
            let pair_tokens = approximate_message_token_count(&previous_messages[idx - 1])
                + approximate_message_token_count(message);
            if current_token_count + pair_tokens > remaining_budget
                || previous_messages[idx - 1].created_at_unix < cutoff_unix
            {
                cutoff_index = idx + 1;
                break;
            }
            current_token_count += pair_tokens;
            idx -= 1;
            continue;
        }

        let message_tokens = approximate_message_token_count(message);
        if current_token_count + message_tokens > remaining_budget
            || message.created_at_unix < cutoff_unix
        {
            cutoff_index = idx + 1;
            break;
        }

        current_token_count += message_tokens;
    }

    let mut compressed = vec![system_message];
    compressed.extend(previous_messages[cutoff_index..].iter().cloned());
    compressed
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
    use crate::RecallContext;
    use chrono::{Local, Utc};
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

    use super::{
        AppRuntime, LOCAL_USER_TOKEN, PromptExecutionOptions, SYNTHETIC_FIRST_USER_MESSAGE,
        argument_limit, build_live_tool_registry, build_live_tool_registry_with_codex_bin_and_hook,
        build_recall_query, classify_memory_recall_with_model, codex_background_status_key,
        compress_context_messages, consolidate_exact_duplicate_memories,
        context_due_item_tool_call_id, context_due_item_tool_messages,
        context_memory_tool_messages, context_task_tool_messages, count_context_tokens,
        determine_memory_recall_decision, drop_old_context_messages, due_item_context_messages,
        format_context_messages_for_summary, format_context_summary_message,
        is_context_refresh_needed, memory_recall_status_updates, message_matches_tool_call_id,
        parse_memory_recall_decision, parse_recalled_item_names, parse_recalled_memory_names,
        parse_reflective_recall_model_response, parse_relevance_filter_response,
        prompt_prelude_status_updates, provider_config_from_app_config,
        recall_due_item_context_messages, recall_memory_context_messages,
        recall_memory_context_messages_with_decision, recalled_item_names_by_type,
        recalled_memory_names, recent_recall_context, refresh_context_if_needed,
        run_prompt_with_model_and_registry, run_prompt_with_model_and_registry_internal,
        run_prompt_with_model_and_registry_stream, select_due_items_by_overlap,
        select_recalled_due_items, select_recalled_memories, select_relevant_recall_agenda_items,
        select_relevant_recall_due_items, select_relevant_recall_memories, should_offer_greeting,
        should_skip_memory_recall, significant_tokens, strip_input_message_for_persistence,
        strip_transient_context_messages, summarize_context_messages_with_model,
    };
    use elroy_agenda::create_agenda_file;
    use elroy_codex::{
        CodexCommandRecord, CodexSessionResult, CodexSessionUpdate, upsert_codex_session,
    };
    use elroy_config::{AppConfig, LlmProvider};
    use elroy_core::{
        ConversationRequest, ModelClient, StreamingModelClient, clear_background_status,
        get_background_status,
    };
    use elroy_db::{
        AgendaItemRecord, BootstrapPlan, MemoryRecord, list_active_due_items,
        load_memory_operation_tracker, load_user_preferences, open_sqlite_connection,
        run_migrations,
    };
    use elroy_feature_requests::{list_feature_requests, write_new_feature_request};
    use elroy_llm::ToolCall;
    use elroy_llm::{ConversationMessage, MessageRole, Provider, StreamEvent};
    use elroy_memory::{create_memory_file, sanitize_filename};
    use elroy_tools::ExecutableToolRegistry;
    use elroy_tui::{TuiCommandExecution, TuiCommandPaletteAction, TuiSlashCommandAction};

    fn seed_competing_memory_record(
        connection: &rusqlite::Connection,
        memory_path: &Path,
        stored_name: &str,
        body: &str,
        updated_at_unix: i64,
    ) {
        connection
            .execute(
                "INSERT INTO bootstrap_documents (
                    kind,
                    path,
                    stem,
                    frontmatter_id,
                    agenda_date,
                    is_completed,
                    status,
                    body,
                    updated_at_unix,
                    trigger_datetime,
                    trigger_context,
                    closing_comment,
                    checklist_total,
                    checklist_completed
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                rusqlite::params![
                    "memory",
                    memory_path.display().to_string(),
                    memory_path.file_stem().and_then(|value| value.to_str()),
                    Option::<i64>::None,
                    Option::<String>::None,
                    0_i64,
                    Option::<String>::None,
                    body,
                    updated_at_unix,
                    Option::<String>::None,
                    Option::<String>::None,
                    Option::<String>::None,
                    0_i64,
                    0_i64,
                ],
            )
            .expect("competing bootstrap document should insert");
        let bootstrap_document_id: i64 = connection
            .query_row(
                "SELECT id FROM bootstrap_documents WHERE path = ?1",
                [memory_path.display().to_string()],
                |row| row.get(0),
            )
            .expect("competing bootstrap document should load");
        connection
            .execute(
                "INSERT INTO memories (
                    bootstrap_document_id,
                    legacy_frontmatter_id,
                    name,
                    file_path,
                    body,
                    is_active,
                    updated_at_unix
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    bootstrap_document_id,
                    Option::<i64>::None,
                    stored_name,
                    memory_path.display().to_string(),
                    body,
                    1_i64,
                    updated_at_unix,
                ],
            )
            .expect("competing memory row should insert");
    }

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

    impl StreamingModelClient for FakeModel {
        fn stream_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.responses.borrow_mut().remove(0);
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct DueItemSurfacingModel {
        round: RefCell<usize>,
    }

    impl DueItemSurfacingModel {
        fn new() -> Self {
            Self {
                round: RefCell::new(0),
            }
        }
    }

    impl ModelClient for DueItemSurfacingModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let mut round = self.round.borrow_mut();
            let current_round = *round;
            *round += 1;

            match current_round {
                0 => {
                    assert_eq!(request.user_message, "Hi, how are you doing today?");
                    assert!(request.transcript.iter().any(|message| {
                        message.role == MessageRole::Tool
                            && message.content.as_deref().is_some_and(|content| {
                                content.contains("Take your daily medicine")
                                    && content.contains("delete_due_item")
                            })
                    }));
                    Ok(vec![StreamEvent::ToolCallRequested(ToolCall {
                        id: "call-delete-due-item".to_string(),
                        name: "delete_due_item".to_string(),
                        arguments_json: "{\"name\":\"medicine reminder\"}".to_string(),
                    })])
                }
                1 => {
                    assert!(request.transcript.iter().any(|message| {
                        message.role == MessageRole::Tool
                            && message.tool_call_id.as_deref() == Some("call-delete-due-item")
                            && message.content.as_deref().is_some_and(|content| {
                                content.contains("Due item 'medicine reminder' has been deleted.")
                            })
                    }));
                    Ok(vec![StreamEvent::AssistantResponse {
                        content: "You had a reminder to take your daily medicine, and I've cleared it for you.".to_string(),
                    }])
                }
                _ => panic!("unexpected extra model round"),
            }
        }
    }

    impl StreamingModelClient for DueItemSurfacingModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct NoDueItemContextModel;

    impl ModelClient for NoDueItemContextModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "How's the weather today?");
            assert!(!request.transcript.iter().any(|message| {
                message.content.as_deref().is_some_and(|content| {
                    content.contains("future reminder")
                        || content.contains("This is for tomorrow")
                        || content.contains("⏰ DUE ITEM")
                })
            }));
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Weather looks calm today.".to_string(),
            }])
        }
    }

    impl StreamingModelClient for NoDueItemContextModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct MultipleDueItemsModel;

    impl ModelClient for MultipleDueItemsModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What's on my schedule today?");
            let tool_messages = request
                .transcript
                .iter()
                .filter_map(|message| {
                    (message.role == MessageRole::Tool)
                        .then_some(message.content.as_deref())
                        .flatten()
                })
                .collect::<Vec<_>>();
            assert!(tool_messages.iter().any(|content| {
                content.contains("First due reminder") && content.contains("⏰ DUE ITEM")
            }));
            assert!(tool_messages.iter().any(|content| {
                content.contains("Second due reminder") && content.contains("⏰ DUE ITEM")
            }));
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You have two reminders due: First due reminder and Second due reminder."
                    .to_string(),
            }])
        }
    }

    impl StreamingModelClient for MultipleDueItemsModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct HybridDueItemModel;

    impl ModelClient for HybridDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What's happening?");
            assert!(request.transcript.iter().any(|message| {
                message.role == MessageRole::Tool
                    && message.content.as_deref().is_some_and(|content| {
                        content.contains("Hybrid reminder text")
                            && content.contains("⏰ DUE ITEM")
                            && content.contains("delete_due_item")
                    })
            }));
            Ok(vec![StreamEvent::AssistantResponse {
                content: "A hybrid reminder is due: Hybrid reminder text.".to_string(),
            }])
        }
    }

    impl StreamingModelClient for HybridDueItemModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct ContextualDueItemModel;

    impl ModelClient for ContextualDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert!(
                request.user_message == "I just got the payroll email."
                    || request.user_message == "I'm following up after that payroll email now."
            );
            assert!(request.transcript.iter().any(|message| {
                message.role == MessageRole::Tool
                    && message.content.as_deref().is_some_and(|content| {
                        content.contains("DUE ITEM")
                            && content.contains("Reply to payroll")
                            && content.contains("\"trigger_context\": \"after payroll email\"")
                    })
            }));
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You have a relevant reminder: Reply to payroll.".to_string(),
            }])
        }
    }

    impl StreamingModelClient for ContextualDueItemModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct MemoryRecallScopeModel;

    impl ModelClient for MemoryRecallScopeModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What preference did I mention?");
            assert!(request.transcript.iter().any(|message| {
                message.role == MessageRole::Tool
                    && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                    && message.content.as_deref().is_some_and(|content| {
                        content.contains("Current preference is tea")
                            && !content.contains("Current preference is coffee")
                    })
            }));
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You prefer tea.".to_string(),
            }])
        }
    }

    impl StreamingModelClient for MemoryRecallScopeModel {
        fn stream_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, elroy_core::ModelClientError>>>,
            elroy_core::ModelClientError,
        > {
            let events = self.next_events(request)?;
            Ok(Box::new(events.into_iter().map(Ok)))
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
        assert_eq!(argument_limit(&serde_json::json!({"n": 0}), 10), 10);
        assert_eq!(argument_limit(&serde_json::json!({"limit": 100}), 10), 50);
        assert_eq!(argument_limit(&serde_json::json!({"limit": 7}), 10), 7);
        assert_eq!(argument_limit(&serde_json::json!({"n": 7}), 10), 7);
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
            "---\ndate: 2026-05-15\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\n---\n\nbring forms\n",
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
        let agenda_detail = runtime
            .open_sidebar_item(
                elroy_tui::SidebarSection::Agenda,
                "doctor visit [2000-01-01 09:00] (Due)",
            )
            .expect("agenda detail should open");
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
                .agenda_titles
                .iter()
                .any(|item| item == "doctor visit [2000-01-01 09:00] (Due)")
        );
        assert!(
            snapshot
                .codex_session_titles
                .iter()
                .any(|item| item == "sample (completed) thread-123")
        );
        assert_eq!(memory_detail.title, "runner notes");
        assert_eq!(memory_detail.destructive_label.as_deref(), Some("archive"));
        assert!(memory_detail.content.contains("remember the hill workout"));
        assert_eq!(agenda_detail.title, "doctor visit");
        assert!(agenda_detail.can_complete);
        assert_eq!(agenda_detail.destructive_label.as_deref(), Some("delete"));
        assert!(
            agenda_detail
                .content
                .contains("trigger_datetime: 2000-01-01T09:00:00")
        );
        assert!(
            codex_detail
                .content
                .contains("Codex inspected the parser state.")
        );
        assert!(codex_detail.content.contains("Parser inspection complete."));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn app_runtime_loads_feature_request_sidebar_sections_and_can_close_improvement() {
        let unique = format!(
            "elroy-rs-app-feature-request-sidebar-{}",
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

        write_new_feature_request(
            &home,
            "Improve correction handling",
            "Recover more directly after user corrections.",
            Some("Reflection found a correction handling gap."),
            Some("- Reflected at: 2026-05-12T00:00:00+00:00\n- Trigger phrase: correction\n- Recent user feedback: please fix corrections"),
            "self_reflection",
        )
        .expect("improvement should be created");
        write_new_feature_request(
            &home,
            "General export feature",
            "Export notes to markdown.",
            Some("Users want portable notes."),
            None,
            "user_request",
        )
        .expect("feature request should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let runtime = AppRuntime::new(config.clone());
        let snapshot = runtime.load_snapshot().expect("snapshot should load");
        let improvement_detail = runtime
            .open_sidebar_item(
                elroy_tui::SidebarSection::Improvements,
                "Improve correction handling (open)",
            )
            .expect("improvement detail should open");
        let feature_request_detail = runtime
            .open_sidebar_item(
                elroy_tui::SidebarSection::FeatureRequests,
                "General export feature (open)",
            )
            .expect("feature request detail should open");

        assert_eq!(
            snapshot.improvement_titles,
            vec!["Improve correction handling (open)".to_string()]
        );
        assert_eq!(
            snapshot.feature_request_titles,
            vec![
                "General export feature (open)".to_string(),
                "Improve correction handling (open)".to_string()
            ]
        );
        assert!(improvement_detail.can_complete);
        assert!(
            improvement_detail
                .content
                .contains("Source: Self-reflection")
        );
        assert!(
            feature_request_detail
                .content
                .contains("Source: User Request")
        );

        let refreshed = runtime
            .mutate_sidebar_item(
                elroy_tui::SidebarSection::Improvements,
                "Improve correction handling (open)",
                elroy_tui::SidebarAction::Complete,
            )
            .expect("improvement should close");

        assert!(refreshed.improvement_titles.is_empty());
        let records = list_feature_requests(&home).expect("feature requests should list");
        let improvement = records
            .into_iter()
            .find(|record| record.title == "Improve correction handling")
            .expect("improvement should remain on disk");
        assert_eq!(improvement.status, "closed");

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
        let memory = registry.invoke("show_memory", "{\"memory_name\":\"runner notes\"}");
        let printed_memory = registry.invoke("print_memory", "{\"memory_name\":\"runner notes\"}");
        let missing_printed_memory =
            registry.invoke("print_memory", "{\"memory_name\":\"missing\"}");
        let agenda = registry.invoke("show_agenda_item", "{\"name\":\"doctor visit\"}");
        let substring_agenda = registry.invoke("show_agenda_item", "{\"name\":\"visit\"}");
        let missing_agenda = registry.invoke("show_agenda_item", "{\"name\":\"dentist\"}");
        let printed_memories = registry.invoke("print_memories", "{\"n\":10}");

        assert!(!memory.is_error);
        assert!(memory.content.contains("remember the hill workout"));
        assert!(!printed_memory.is_error);
        assert!(printed_memory.content.starts_with("#runner notes\n"));
        assert!(printed_memory.content.contains("remember the hill workout"));
        assert!(!missing_printed_memory.is_error);
        assert_eq!(
            missing_printed_memory.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(!agenda.is_error);
        assert!(agenda.content.contains("bring forms"));
        assert!(!substring_agenda.is_error);
        assert!(substring_agenda.content.contains("bring forms"));
        assert!(missing_agenda.is_error);
        assert_eq!(
            missing_agenda.content,
            "No agenda item found matching 'dentist'."
        );
        assert!(!printed_memories.is_error);
        assert!(printed_memories.content.contains("Memories"));
        assert!(printed_memories.content.contains("runner notes"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn print_memories_lists_oldest_visible_first() {
        let unique = format!(
            "elroy-rs-app-memory-order-{}",
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
        fs::write(memory_dir.join("older.md"), "remember the earlier note\n")
            .expect("older memory should be written");
        std::thread::sleep(std::time::Duration::from_secs(1));
        fs::write(memory_dir.join("newer.md"), "remember the later note\n")
            .expect("newer memory should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let printed = registry.invoke("print_memories", "{\"n\":10}");

        assert!(!printed.is_error);
        let older_index = printed
            .content
            .find("- older | Text: remember the earlier note")
            .expect("older memory should appear");
        let newer_index = printed
            .content
            .find("- newer | Text: remember the later note")
            .expect("newer memory should appear");
        assert!(older_index < newer_index);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_list_agenda_items_for_one_date() {
        let unique = format!(
            "elroy-rs-app-list-agenda-items-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let listed = registry.invoke("list_agenda_items", "{\"item_date\":\"2026-05-15\"}");

        assert!(!listed.is_error);
        assert!(listed.content.contains("\"item_date\":\"2026-05-15\""));
        assert!(listed.content.contains("\"name\":\"doctor visit\""));
        assert!(listed.content.contains("\"text\":\"bring forms\""));
        assert!(!listed.content.contains("\"name\":\"trip\""));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_format_agenda_items_cmd() {
        let unique = format!(
            "elroy-rs-app-list-agenda-items-cmd-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
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
    fn live_tool_registry_can_show_source_content_for_memory() {
        let unique = format!(
            "elroy-rs-app-memory-source-{}",
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
            "remember the hill workout\nwith the harder second interval\n",
        )
        .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        let missing_source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"missing\"}",
        );
        let missing_source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"missing\"}",
        );
        let out_of_range = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\",\"index\":1}",
        );

        assert!(!source.is_error);
        assert_eq!(source.content, "No sources found for memory 'runner notes'");
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[]");
        assert!(missing_source_list.is_error);
        assert_eq!(
            missing_source_list.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(missing_source.is_error);
        assert_eq!(
            missing_source.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(!out_of_range.is_error);
        assert_eq!(
            out_of_range.content,
            "No sources found for memory 'runner notes'"
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_list_source_metadata_for_memory() {
        let unique = format!(
            "elroy-rs-app-memory-source-list-{}",
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
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::User,
                "Hello, I ran a marathon today!",
            )],
        )
        .expect("context messages should persist");

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_memory",
            "{\"name\":\"Running progress\",\"text\":\"I ran a marathon today\"}",
        );
        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"running progress\"}",
        );
        let out_of_range = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"running progress\",\"index\":1}",
        );

        assert!(!created.is_error);
        assert_eq!(created.content, "New memory created: Running progress");
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[[\"ContextMessageSet\",\"1\"]]");
        assert!(out_of_range.is_error);
        assert_eq!(
            out_of_range.content,
            "Index 1 out of range. Available indices: [0]"
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_memory_tool_records_context_message_source_content() {
        let unique = format!(
            "elroy-rs-app-memory-context-source-{}",
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
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::User,
                "Hello, I ran a marathon today!",
            )],
        )
        .expect("context messages should persist");

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_memory",
            "{\"name\":\"Running progress\",\"text\":\"I ran a marathon today\"}",
        );
        let source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"running progress\"}",
        );

        assert!(!created.is_error);
        assert_eq!(created.content, "New memory created: Running progress");
        assert!(!source.is_error);
        assert!(
            source
                .content
                .contains("user: Hello, I ran a marathon today!")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_memory_tool_schema_only_exposes_name_and_text() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "create_memory")
            .expect("create_memory tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("text"));
        assert!(!properties.contains_key("item_date"));
        assert!(!properties.contains_key("date"));
        assert!(!properties.contains_key("trigger_datetime"));
        assert!(!properties.contains_key("trigger_context"));
    }

    #[test]
    fn create_due_item_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "create_due_item")
            .expect("create_due_item tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 4);
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("text"));
        assert!(properties.contains_key("trigger_time"));
        assert!(properties.contains_key("trigger_context"));
        assert!(!properties.contains_key("trigger_datetime"));
        assert!(!properties.contains_key("date"));
    }

    #[test]
    fn update_due_item_text_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "update_due_item_text")
            .expect("update_due_item_text tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("new_text"));
        assert!(!properties.contains_key("text"));
    }

    #[test]
    fn rename_due_item_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "rename_due_item")
            .expect("rename_due_item tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("old_name"));
        assert!(properties.contains_key("new_name"));
        assert!(!properties.contains_key("name"));
    }

    #[test]
    fn rename_task_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "rename_task")
            .expect("rename_task tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("old_name"));
        assert!(properties.contains_key("new_name"));
        assert!(!properties.contains_key("name"));
    }

    #[test]
    fn print_memory_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "print_memory")
            .expect("print_memory tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 1);
        assert!(properties.contains_key("memory_name"));
        assert!(!properties.contains_key("name"));
    }

    #[test]
    fn print_memories_tool_schema_matches_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);
        let spec = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == "print_memories")
            .expect("print_memories tool should exist");

        let properties = match &spec.parameters {
            elroy_tools::JsonSchema::Object { properties, .. } => properties,
        };

        assert_eq!(properties.len(), 1);
        assert!(properties.contains_key("n"));
        assert!(!properties.contains_key("limit"));
    }

    #[test]
    fn due_item_print_list_tool_schemas_match_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);

        for tool_name in ["print_active_due_items", "print_inactive_due_items"] {
            let spec = registry
                .specs()
                .into_iter()
                .find(|spec| spec.name == tool_name)
                .unwrap_or_else(|| panic!("{tool_name} tool should exist"));

            let properties = match &spec.parameters {
                elroy_tools::JsonSchema::Object { properties, .. } => properties,
            };

            assert_eq!(
                properties.len(),
                1,
                "{tool_name} should expose only one field"
            );
            assert!(properties.contains_key("n"), "{tool_name} should expose n");
            assert!(
                !properties.contains_key("limit"),
                "{tool_name} should not expose limit"
            );
        }
    }

    #[test]
    fn memory_search_tool_schemas_match_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);

        for (tool_name, expected_field) in [
            ("search_memories", "query"),
            ("examine_memories", "question"),
        ] {
            let spec = registry
                .specs()
                .into_iter()
                .find(|spec| spec.name == tool_name)
                .unwrap_or_else(|| panic!("{tool_name} tool should exist"));

            let properties = match &spec.parameters {
                elroy_tools::JsonSchema::Object { properties, .. } => properties,
            };

            assert_eq!(
                properties.len(),
                1,
                "{tool_name} should expose only one field"
            );
            assert!(
                properties.contains_key(expected_field),
                "{tool_name} should expose {expected_field}"
            );
            assert!(
                !properties.contains_key("limit"),
                "{tool_name} should not expose limit"
            );
        }
    }

    #[test]
    fn task_list_tool_schemas_match_python_surface() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);

        for tool_name in [
            "list_tasks",
            "list_triggered_tasks",
            "list_due_tasks",
            "list_today_tasks",
        ] {
            let spec = registry
                .specs()
                .into_iter()
                .find(|spec| spec.name == tool_name)
                .unwrap_or_else(|| panic!("{tool_name} tool should exist"));

            let properties = match &spec.parameters {
                elroy_tools::JsonSchema::Object { properties, .. } => properties,
            };

            assert!(
                properties.is_empty(),
                "{tool_name} should not expose Rust-only limit parameters"
            );
        }
    }

    #[test]
    fn consolidated_memory_records_multiple_memory_sources() {
        let unique = format!(
            "elroy-rs-app-consolidated-memory-source-{}",
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
            memory_dir.join("running_progress.md"),
            "I ran a marathon today\n",
        )
        .expect("first memory should be written");
        fs::write(memory_dir.join("run_today.md"), "I ran 24 miles today\n")
            .expect("second memory should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_consolidated_memory",
            "{\"name\":\"Running summary\",\"text\":\"The user ran a marathon and later reported running 24 miles in total.\",\"source_names\":[\"running progress\",\"run today\"]}",
        )
        ;
        assert!(!created.is_error);

        let connection = open_sqlite_connection(&database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert_eq!(active_memories.len(), 1);
        assert_eq!(active_memories[0].name, "running summary");

        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"running summary\"}",
        );
        assert!(!source_list.is_error);
        let mut source_entries: Vec<(String, String)> =
            serde_json::from_str::<Vec<(String, String)>>(&source_list.content)
                .expect("source list should parse");
        source_entries.sort();
        assert_eq!(
            source_entries,
            vec![
                ("Memory".to_string(), "run today".to_string()),
                ("Memory".to_string(), "running progress".to_string()),
            ]
        );
        let running_progress_index =
            serde_json::from_str::<Vec<(String, String)>>(&source_list.content)
                .expect("source list should parse")
                .iter()
                .position(|entry| entry == &("Memory".to_string(), "running progress".to_string()))
                .expect("running progress source should be present");

        let source_content = registry.invoke(
            "get_source_content_for_memory",
            &format!("{{\"memory_name\":\"running summary\",\"index\":{running_progress_index}}}"),
        );
        assert!(!source_content.is_error);
        assert!(source_content.content.contains("#running progress"));
        assert!(source_content.content.contains("I ran a marathon today"));

        assert!(
            memory_dir
                .join("archive")
                .join("running_progress.md")
                .exists()
        );
        assert!(memory_dir.join("archive").join("run_today.md").exists());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_memory_tool_triggers_exact_duplicate_consolidation_at_threshold() {
        let unique = format!(
            "elroy-rs-app-duplicate-memory-consolidation-{}",
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
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.memories_between_consolidation = 2;

        let registry = build_live_tool_registry(&config);
        let first = registry.invoke(
            "create_memory",
            "{\"name\":\"Running progress\",\"text\":\"I ran a marathon today\"}",
        );
        let second = registry.invoke(
            "create_memory",
            "{\"name\":\"Run today\",\"text\":\"I ran a marathon today\"}",
        );
        assert!(!first.is_error);
        assert!(!second.is_error);

        let connection = open_sqlite_connection(&database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert_eq!(active_memories.len(), 1);
        let consolidated_name = active_memories[0].name.clone();

        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.memories_since_consolidation, 0);

        let source_list = registry.invoke(
            "get_source_list_for_memory",
            &format!("{{\"memory_name\":\"{consolidated_name}\"}}"),
        );
        assert!(!source_list.is_error);
        let mut source_entries: Vec<(String, String)> =
            serde_json::from_str::<Vec<(String, String)>>(&source_list.content)
                .expect("source list should parse");
        source_entries.sort();
        assert_eq!(
            source_entries,
            vec![
                ("Memory".to_string(), "run today".to_string()),
                ("Memory".to_string(), "running progress".to_string()),
            ]
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn exact_duplicate_consolidation_scopes_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-memory-consolidation-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("running_progress.md"),
            "I ran a marathon today\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("other_duplicate.md"),
            "I ran a marathon today\n",
        )
        .expect("other memory should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = current_home;
        config.memory_dir = current_memory_dir.clone();
        config.agenda_dir = current_agenda_dir;
        config.database_path = database_path.clone();
        config.memories_between_consolidation = 1;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("current bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("other_duplicate.md"),
            "Other Duplicate",
            "I ran a marathon today",
            9_999,
        );

        consolidate_exact_duplicate_memories(&mut connection, &BootstrapPlan::from_config(&config))
            .expect("consolidation should succeed");

        let connection = open_sqlite_connection(&database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert_eq!(active_memories.len(), 2);
        assert!(
            active_memories
                .iter()
                .any(|memory| memory.name == "running progress")
        );
        assert!(
            active_memories
                .iter()
                .any(|memory| memory.name == "Other Duplicate")
        );

        assert!(current_memory_dir.join("running_progress.md").exists());
        assert!(other_memory_dir.join("other_duplicate.md").exists());
        assert!(!current_memory_dir.join("archive").exists());
        assert!(!other_memory_dir.join("archive").exists());

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn live_tool_registry_can_add_and_drop_memory_from_current_context() {
        let unique = format!(
            "elroy-rs-app-context-memory-{}",
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
            memory_dir.join("travel_preference.md"),
            "User likes window seats on long flights.\n",
        )
        .expect("memory should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let add = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!add.is_error);
        assert_eq!(add.content, "Memory 'travel preference' added to context.");

        let context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!context.is_error);
        assert!(context.content.contains("get_fast_recall"));
        assert!(context.content.contains("travel preference"));

        let add_again = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!add_again.is_error);
        let context_again = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert_eq!(
            context_again
                .content
                .matches("\"tool_call_id\": \"context-memory:travel preference\"")
                .count(),
            1
        );

        let drop_item = registry.invoke(
            "drop_memory_from_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!drop_item.is_error);
        assert_eq!(
            drop_item.content,
            "Memory 'travel preference' dropped from context."
        );

        let stripped = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!stripped.is_error);
        assert!(
            !stripped
                .content
                .contains("context-memory:travel preference")
        );
        assert!(!stripped.content.contains("get_fast_recall"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn add_memory_to_current_context_scopes_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-context-memory-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("shared_memory_name.md"),
            "Current user memory\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("shared_memory_name.md"),
            "Other user memory\n",
        )
        .expect("other memory should be written");

        let mut current_config = AppConfig::defaults();
        current_config.home_dir = current_home;
        current_config.memory_dir = current_memory_dir.clone();
        current_config.agenda_dir = current_agenda_dir;
        current_config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&current_config))
            .expect("current bootstrap should succeed");

        let mut connection =
            open_sqlite_connection(&current_config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        connection
            .execute(
                "INSERT INTO bootstrap_documents (
                    kind,
                    path,
                    stem,
                    frontmatter_id,
                    agenda_date,
                    is_completed,
                    status,
                    body,
                    updated_at_unix,
                    trigger_datetime,
                    trigger_context,
                    closing_comment,
                    checklist_total,
                    checklist_completed
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                rusqlite::params![
                    "memory",
                    other_memory_dir
                        .join("shared_memory_name.md")
                        .display()
                        .to_string(),
                    "shared_memory_name",
                    Option::<i64>::None,
                    Option::<String>::None,
                    0_i64,
                    Option::<String>::None,
                    "Other user memory",
                    9_999_i64,
                    Option::<String>::None,
                    Option::<String>::None,
                    Option::<String>::None,
                    0_i64,
                    0_i64,
                ],
            )
            .expect("other bootstrap document should insert");
        let bootstrap_document_id: i64 = connection
            .query_row(
                "SELECT id FROM bootstrap_documents WHERE path = ?1",
                [other_memory_dir
                    .join("shared_memory_name.md")
                    .display()
                    .to_string()],
                |row| row.get(0),
            )
            .expect("other bootstrap document should load");
        connection
            .execute(
                "INSERT INTO memories (
                    bootstrap_document_id,
                    legacy_frontmatter_id,
                    name,
                    file_path,
                    body,
                    is_active,
                    updated_at_unix
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    bootstrap_document_id,
                    Option::<i64>::None,
                    "Shared Memory Name",
                    other_memory_dir
                        .join("shared_memory_name.md")
                        .display()
                        .to_string(),
                    "Other user memory",
                    1_i64,
                    9_999_i64,
                ],
            )
            .expect("other memory row should insert");

        let registry = build_live_tool_registry(&current_config);
        let add = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"Shared Memory Name\"}",
        );
        assert!(!add.is_error);
        assert_eq!(add.content, "Memory 'shared memory name' added to context.");

        let context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!context.is_error);
        assert!(context.content.contains("Current user memory"));
        assert!(!context.content.contains("Other user memory"));

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn exact_name_memory_tools_scope_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-memory-tool-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("shared_memory_name.md"),
            "Current user memory\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("shared_memory_name.md"),
            "Other user memory\n",
        )
        .expect("other memory should be written");
        fs::write(
            other_memory_dir.join("other_only_memory.md"),
            "Other user only memory\n",
        )
        .expect("other unique memory should be written");

        let mut current_config = AppConfig::defaults();
        current_config.home_dir = current_home;
        current_config.memory_dir = current_memory_dir.clone();
        current_config.agenda_dir = current_agenda_dir;
        current_config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&current_config))
            .expect("current bootstrap should succeed");

        let mut connection =
            open_sqlite_connection(&current_config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("shared_memory_name.md"),
            "Shared Memory Name",
            "Other user memory",
            9_999,
        );
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("other_only_memory.md"),
            "Other Only Memory",
            "Other user only memory",
            10_000,
        );

        let registry = build_live_tool_registry(&current_config);
        let shown = registry.invoke("show_memory", "{\"memory_name\":\"Shared Memory Name\"}");
        let printed = registry.invoke("print_memory", "{\"memory_name\":\"Shared Memory Name\"}");
        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"Shared Memory Name\"}",
        );
        let source_content = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"Shared Memory Name\"}",
        );
        let missing_other =
            registry.invoke("show_memory", "{\"memory_name\":\"Other Only Memory\"}");

        assert!(!shown.is_error);
        assert!(shown.content.contains("Current user memory"));
        assert!(!shown.content.contains("Other user memory"));
        assert!(!printed.is_error);
        assert!(printed.content.contains("Current user memory"));
        assert!(!printed.content.contains("Other user memory"));
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[]");
        assert!(!source_content.is_error);
        assert_eq!(
            source_content.content,
            "No sources found for memory 'shared memory name'"
        );
        assert!(missing_other.is_error);
        assert_eq!(
            missing_other.content,
            "Memory 'Other Only Memory' not found for the current user."
        );

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn memory_mutation_tools_scope_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-memory-mutation-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("runner_notes.md"),
            "Current user memory\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("runner_notes.md"),
            "Other user memory\n",
        )
        .expect("other memory should be written");

        let mut current_config = AppConfig::defaults();
        current_config.home_dir = current_home;
        current_config.memory_dir = current_memory_dir.clone();
        current_config.agenda_dir = current_agenda_dir;
        current_config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&current_config))
            .expect("current bootstrap should succeed");

        let mut connection =
            open_sqlite_connection(&current_config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("runner_notes.md"),
            "Runner Notes",
            "Other user memory",
            9_999,
        );

        let registry = build_live_tool_registry(&current_config);
        let updated = registry.invoke(
            "update_memory",
            "{\"memory_name\":\"Runner Notes\",\"text\":\"Updated current user memory\"}",
        );
        assert!(!updated.is_error);
        assert!(
            fs::read_to_string(current_memory_dir.join("runner_notes.md"))
                .expect("current memory should be readable")
                .contains("Updated current user memory")
        );
        assert!(
            fs::read_to_string(other_memory_dir.join("runner_notes.md"))
                .expect("other memory should be readable")
                .contains("Other user memory")
        );

        let outdated = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"Runner Notes\",\"update_text\":\"Current correction\"}",
        );
        assert!(!outdated.is_error);
        assert!(
            fs::read_to_string(current_memory_dir.join("runner_notes.md"))
                .expect("current memory should be readable after outdated update")
                .contains("Current correction")
        );
        assert!(
            fs::read_to_string(other_memory_dir.join("runner_notes.md"))
                .expect("other memory should still be readable")
                .contains("Other user memory")
        );
        assert!(
            current_memory_dir
                .join("archive")
                .join("runner_notes.md")
                .exists()
        );
        assert!(other_memory_dir.join("runner_notes.md").exists());

        let archived = registry.invoke("archive_memory", "{\"memory_name\":\"Runner Notes\"}");
        assert!(!archived.is_error);
        assert!(
            current_memory_dir
                .join("archive")
                .join("runner_notes.md")
                .exists()
        );
        assert!(other_memory_dir.join("runner_notes.md").exists());

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn memory_query_tools_scope_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-memory-query-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("tea_preference.md"),
            "Current preference is tea\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("coffee_preference.md"),
            "Current preference is coffee\n",
        )
        .expect("other memory should be written");

        let mut current_config = AppConfig::defaults();
        current_config.home_dir = current_home;
        current_config.memory_dir = current_memory_dir.clone();
        current_config.agenda_dir = current_agenda_dir;
        current_config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&current_config))
            .expect("current bootstrap should succeed");

        let mut connection =
            open_sqlite_connection(&current_config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("coffee_preference.md"),
            "Coffee Preference",
            "Current preference is coffee",
            9_999,
        );

        let registry = build_live_tool_registry(&current_config);
        let listed = registry.invoke("list_memories", "{\"limit\":10}");
        let printed = registry.invoke("print_memories", "{\"n\":10}");
        let searched = registry.invoke("search_memories", "{\"query\":\"coffee\"}");
        let examined = registry.invoke(
            "examine_memories",
            "{\"question\":\"What coffee note do I have?\"}",
        );

        assert!(!listed.is_error);
        assert!(listed.content.contains("tea preference"));
        assert!(!listed.content.contains("coffee preference"));
        assert!(!printed.is_error);
        assert!(printed.content.contains("tea preference"));
        assert!(!printed.content.contains("coffee preference"));
        assert!(!searched.is_error);
        assert_eq!(searched.content, "No relevant memories found");
        assert!(!examined.is_error);
        assert_eq!(examined.content, "No relevant memories found");

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_scopes_memory_recall_to_current_memory_dir() {
        let unique = format!(
            "elroy-rs-app-memory-recall-scope-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let current_home = root.join("current-user");
        let current_memory_dir = current_home.join("memories");
        let other_memory_dir = root.join("other-user").join("memories");
        let current_agenda_dir = current_home.join("agenda");
        let database_path = root.join("shared.db");
        fs::create_dir_all(&current_memory_dir).expect("current memory dir should be created");
        fs::create_dir_all(&other_memory_dir).expect("other memory dir should be created");
        fs::create_dir_all(&current_agenda_dir).expect("current agenda dir should be created");
        fs::write(
            current_memory_dir.join("tea_preference.md"),
            "Current preference is tea\n",
        )
        .expect("current memory should be written");
        fs::write(
            other_memory_dir.join("coffee_preference.md"),
            "Current preference is coffee\n",
        )
        .expect("other memory should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = current_home.clone();
        config.memory_dir = current_memory_dir.clone();
        config.agenda_dir = current_agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("current bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        seed_competing_memory_record(
            &connection,
            &other_memory_dir.join("coffee_preference.md"),
            "Coffee Preference",
            "Current preference is coffee",
            9_999,
        );

        let model = MemoryRecallScopeModel;
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "What preference did I mention?",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &current_home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: false,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content } if content == "You prefer tea."
        )));

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn live_tool_registry_includes_get_fast_recall_ack_tool() {
        let registry = build_live_tool_registry(&AppConfig::defaults());
        let result = registry.invoke("get_fast_recall", "{}");

        assert!(!result.is_error);
        assert_eq!(result.content, "OK");
    }

    #[test]
    fn live_tool_registry_includes_get_reflective_recall_ack_tool() {
        let registry = build_live_tool_registry(&AppConfig::defaults());
        let result = registry.invoke("get_reflective_recall", "{}");

        assert!(!result.is_error);
        assert_eq!(result.content, "OK");
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
            "{\"memory_name\":\"runner notes\",\"text\":\"new text\"}",
        );
        assert!(!update.is_error);
        assert!(
            fs::read_to_string(memory_dir.join("runner_notes.md"))
                .expect("updated memory should be readable")
                .contains("new text")
        );

        let archive = registry.invoke("archive_memory", "{\"memory_name\":\"runner notes\"}");
        assert!(!archive.is_error);
        assert!(memory_dir.join("archive").join("runner_notes.md").exists());

        let connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert!(active_memories.is_empty());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_append_outdated_memory_update() {
        let unique = format!(
            "elroy-rs-app-outdated-memory-update-{}",
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
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let update = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"runner notes\",\"update_text\":\"new correction\"}",
        );
        assert!(!update.is_error);
        assert_eq!(update.content, "Memory 'runner notes' has been updated");
        let missing_update = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"missing\",\"update_text\":\"unused\"}",
        );
        assert!(!missing_update.is_error);
        assert_eq!(missing_update.content, "Memory 'missing' not found");

        let file_text =
            fs::read_to_string(memory_dir.join("runner_notes.md")).expect("memory should read");
        assert!(file_text.contains("old text"));
        assert!(file_text.contains("Update ("));
        assert!(file_text.contains("new correction"));
        assert!(memory_dir.join("archive").join("runner_notes.md").exists());

        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[[\"Memory\",\"runner notes\"]]");

        let source_content = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\",\"index\":0}",
        );
        assert!(!source_content.is_error);
        assert!(source_content.content.contains("#runner notes"));
        assert!(source_content.content.contains("old text"));
        assert!(!source_content.content.contains("new correction"));

        let connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert_eq!(active_memories.len(), 1);
        assert!(active_memories[0].body.contains("new correction"));
        let historical_rows: Vec<(String, bool, String)> = {
            let mut statement = connection
                .prepare(
                    "SELECT file_path, is_active, body
                     FROM memories
                     WHERE LOWER(name) = LOWER(?1)
                     ORDER BY is_active DESC, updated_at_unix DESC, file_path ASC",
                )
                .expect("memory history query should prepare");
            let rows = statement
                .query_map(["runner notes"], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)? != 0,
                        row.get::<_, String>(2)?,
                    ))
                })
                .expect("memory history rows should map");
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .expect("memory history rows should collect")
        };
        assert_eq!(historical_rows.len(), 2);
        assert_eq!(
            historical_rows[0].0,
            memory_dir.join("runner_notes.md").display().to_string()
        );
        assert!(historical_rows[0].1);
        assert!(historical_rows[0].2.contains("old text"));
        assert!(historical_rows[0].2.contains("new correction"));
        assert_eq!(
            historical_rows[1].0,
            memory_dir
                .join("archive")
                .join("runner_notes.md")
                .display()
                .to_string()
        );
        assert!(!historical_rows[1].1);
        assert!(historical_rows[1].2.contains("old text"));
        assert!(!historical_rows[1].2.contains("new correction"));

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
        let added_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!added_context.is_error);
        assert!(
            added_context
                .content
                .contains("context-task:project kickoff")
        );
        assert!(added_context.content.contains("Prepare slides"));

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
        let updated_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!updated_context.is_error);
        assert!(
            updated_context
                .content
                .contains("context-task:project kickoff")
        );
        assert!(updated_context.content.contains("called ahead"));

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
        let completed_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!completed_context.is_error);
        assert!(
            !completed_context
                .content
                .contains("context-task:project kickoff")
        );

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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
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
        let deleted_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!deleted_context.is_error);
        assert!(!deleted_context.content.contains("context-task:desk notes"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn add_agenda_item_can_derive_name_and_default_date_from_text() {
        let unique = format!(
            "elroy-rs-app-agenda-derived-name-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let today = Local::now().date_naive().to_string();
        let added = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Sprint kickoff planning\\nReview owners and milestones.\"}",
        );
        assert!(!added.is_error);
        assert_eq!(
            added.content,
            format!("Agenda item added for {today}: sprint_kickoff_planning")
        );
        let stored = fs::read_to_string(agenda_dir.join("sprint_kickoff_planning.md"))
            .expect("agenda file should read");
        assert!(stored.contains(&format!("date: {today}")));
        assert!(stored.contains("Sprint kickoff planning"));
        assert!(stored.contains("Review owners and milestones."));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn add_agenda_item_accepts_item_date_alias() {
        let unique = format!(
            "elroy-rs-app-agenda-item-date-alias-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let added = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Quarterly planning\",\"item_date\":\"2026-05-22\"}",
        );
        assert!(!added.is_error);
        assert_eq!(
            added.content,
            "Agenda item added for 2026-05-22: quarterly_planning"
        );
        let stored = fs::read_to_string(agenda_dir.join("quarterly_planning.md"))
            .expect("agenda file should read");
        assert!(stored.contains("date: 2026-05-22"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_item_date_tools_reject_invalid_dates() {
        let unique = format!(
            "elroy-rs-app-agenda-invalid-date-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let added = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Quarterly planning\",\"item_date\":\"2026/05/22\"}",
        );
        let listed = registry.invoke("list_agenda_items", "{\"item_date\":\"2026/05/22\"}");
        let listed_cmd = registry.invoke("list_agenda_items_cmd", "{\"item_date\":\"2026/05/22\"}");

        assert!(added.is_error);
        assert_eq!(
            added.content,
            "Invalid date format '2026/05/22'. Use YYYY-MM-DD."
        );
        assert!(listed.is_error);
        assert_eq!(
            listed.content,
            "Invalid date format '2026/05/22'. Use YYYY-MM-DD."
        );
        assert!(listed_cmd.is_error);
        assert_eq!(
            listed_cmd.content,
            "Invalid date format '2026/05/22'. Use YYYY-MM-DD."
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn list_agenda_items_excludes_deleted_and_due_items() {
        let unique = format!(
            "elroy-rs-app-agenda-list-filtering-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let added = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Write the Q2 report.\",\"item_date\":\"2026-03-20\"}",
        );
        assert!(!added.is_error);
        let deleted = registry.invoke(
            "delete_agenda_item",
            "{\"item_name\":\"write the q2 report\"}",
        );
        assert!(!deleted.is_error);

        let readded = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Write the Q2 report.\",\"item_date\":\"2026-03-20\"}",
        );
        assert!(!readded.is_error);

        let due_item = registry.invoke(
            "create_due_item",
            "{\"name\":\"Pay rent\",\"text\":\"Pay rent before the first of the month.\",\"trigger_context\":\"when I mention rent\"}",
        );
        assert!(!due_item.is_error);

        let listed = registry.invoke("list_agenda_items", "{\"item_date\":\"2026-03-20\"}");
        assert!(!listed.is_error);
        let payload: serde_json::Value =
            serde_json::from_str(&listed.content).expect("agenda listing should be valid json");
        assert_eq!(payload["item_date"], "2026-03-20");
        let items = payload["items"]
            .as_array()
            .expect("agenda listing should contain items");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["name"], "write the q2 report");
        assert_eq!(items[0]["text"], "Write the Q2 report.");

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn show_agenda_item_ignores_completed_and_deleted_same_name_rows() {
        let unique = format!(
            "elroy-rs-app-agenda-active-lookup-{}",
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
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let first = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Write Q2 report.\\nOriginal details\",\"item_date\":\"2026-03-20\"}",
        );
        assert!(!first.is_error);
        let completed = registry.invoke(
            "complete_agenda_item",
            "{\"item_name\":\"write q2 report\"}",
        );
        assert!(!completed.is_error);

        let second = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Write Q2 report.\\nReplacement details\",\"item_date\":\"2026-03-20\"}",
        );
        assert!(!second.is_error);
        let shown_after_complete = registry.invoke("show_agenda_item", "{\"name\":\"write\"}");
        assert!(!shown_after_complete.is_error);
        assert!(shown_after_complete.content.contains("Replacement details"));
        assert!(!shown_after_complete.content.contains("Original details"));

        let deleted = registry.invoke("delete_agenda_item", "{\"item_name\":\"write q2 report\"}");
        assert!(!deleted.is_error);
        let third = registry.invoke(
            "add_agenda_item",
            "{\"text\":\"Write Q2 report.\\nThird details\",\"item_date\":\"2026-03-20\"}",
        );
        assert!(!third.is_error);
        let shown_after_delete = registry.invoke("show_agenda_item", "{\"name\":\"write\"}");
        assert!(!shown_after_delete.is_error);
        assert!(shown_after_delete.content.contains("Third details"));
        assert!(!shown_after_delete.content.contains("Replacement details"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn agenda_sidebar_delete_rejects_plain_agenda_items() {
        let unique = format!(
            "elroy-rs-app-sidebar-agenda-delete-{}",
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
            "---\ndate: 2026-05-16\ncompleted: false\nstatus: created\n---\n\ncall mom\n",
        )
        .expect("agenda file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let runtime = AppRuntime::new(config);
        let error = runtime
            .mutate_sidebar_item(
                elroy_tui::SidebarSection::Agenda,
                "call mom",
                elroy_tui::SidebarAction::Delete,
            )
            .expect_err("plain agenda item should not be deletable");

        assert!(
            error
                .to_string()
                .contains("agenda item is not deletable from the sidebar")
        );

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
        let context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!context.is_error);
        assert!(context.content.contains("get_fast_recall"));
        assert!(context.content.contains("context-due-item:call mom"));
        assert!(context.content.contains("call mom"));

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
        let timed_context = registry.invoke("show_context_messages", "{\"limit\":40}");
        assert!(!timed_context.is_error);
        assert!(timed_context.content.contains("context-due-item:pay rent"));
        assert!(timed_context.content.contains("pay rent"));

        let deleted = registry.invoke(
            "delete_due_item",
            "{\"name\":\"call mom\",\"closing_comment\":\"done\"}",
        );
        assert!(!deleted.is_error);
        let recreated = registry.invoke(
            "create_due_item",
            "{\"name\":\"call mom\",\"text\":\"Call mom again\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!recreated.is_error);
        assert_eq!(
            recreated.content,
            "Contextual due item 'call mom' has been created."
        );
        let stripped = registry.invoke("show_context_messages", "{\"limit\":40}");
        assert!(!stripped.is_error);
        assert!(stripped.content.contains("context-due-item:call mom"));
        assert!(stripped.content.contains("context-due-item:pay rent"));

        let listed = registry.invoke("list_due_items", "{\"limit\":10}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("pay rent"));
        assert!(listed.content.contains("call mom"));
    }

    #[test]
    fn live_tool_registry_can_show_and_list_inactive_due_items() {
        let unique = format!(
            "elroy-rs-app-inactive-due-items-{}",
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

        let registry = build_live_tool_registry(&config);
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
        assert_eq!(
            duplicate.content,
            "Preferred name already set to Jimmy. If this should be changed, use override_existing=True."
        );

        let preferred_override = registry.invoke(
            "set_user_preferred_name",
            "{\"preferred_name\":\"James\",\"override_existing\":true}",
        );
        assert!(!preferred_override.is_error);
        assert_eq!(
            preferred_override.content,
            "Set user preferred name to James. Was Jimmy."
        );

        let assistant = registry.invoke("set_assistant_name", "{\"assistant_name\":\"Nova\"}");
        assert!(!assistant.is_error);
        assert!(assistant.content.contains("Nova"));

        let full_name = registry.invoke("set_user_full_name", "{\"full_name\":\"James Smith\"}");
        assert!(!full_name.is_error);
        assert_eq!(
            full_name.content,
            "Full name set to James Smith. Previous value was Unknown name."
        );

        let full_name_duplicate =
            registry.invoke("set_user_full_name", "{\"full_name\":\"James T. Smith\"}");
        assert!(!full_name_duplicate.is_error);
        assert_eq!(
            full_name_duplicate.content,
            "Full name already set to James Smith. If this should be changed, set override_existing=True."
        );

        let full_name_override = registry.invoke(
            "set_user_full_name",
            "{\"full_name\":\"James T. Smith\",\"override_existing\":true}",
        );
        assert!(!full_name_override.is_error);
        assert_eq!(
            full_name_override.content,
            "Full name set to James T. Smith. Previous value was James Smith."
        );

        let get_full_name = registry.invoke("get_user_full_name", "{}");
        assert!(!get_full_name.is_error);
        assert_eq!(get_full_name.content, "James T. Smith");

        let persona = registry.invoke(
            "set_persona",
            "{\"system_persona\":\"You are $ASSISTANT_ALIAS helping $USER_ALIAS.\"}",
        );
        assert!(!persona.is_error);
        assert_eq!(persona.content, "System persona updated.");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        let context =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(context[0].role, MessageRole::System);
        assert_eq!(
            context[0].content.as_deref(),
            Some("You are Nova helping James.")
        );

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
        let context =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(context[0].role, MessageRole::System);
        assert!(
            context[0]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("I am Nova"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_list_make_and_edit_feature_requests() {
        let unique = format!(
            "elroy-rs-app-feature-requests-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;

        let registry = build_live_tool_registry(&config);

        let empty = registry.invoke("list_feature_requests", "{}");
        assert!(!empty.is_error);
        assert_eq!(empty.content, "No feature requests found.");

        let created = registry.invoke(
            "make_feature_request",
            "{\"title\":\"Add calendar sync\",\"description\":\"Sync Elroy tasks to a calendar provider.\",\"rationale\":\"Users want a unified schedule.\"}",
        );
        assert!(!created.is_error);
        assert!(
            created
                .content
                .contains("Created feature request: Add calendar sync")
        );

        let merged = registry.invoke(
            "make_feature_request",
            "{\"title\":\"Add calendar synchronization\",\"description\":\"Sync tasks to an external calendar.\",\"rationale\":\"Users want calendar parity.\"}",
        );
        assert!(!merged.is_error);
        assert!(
            merged
                .content
                .contains("Merged into existing feature request: Add calendar sync")
        );

        let listed = registry.invoke("list_feature_requests", "{}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("Feature requests (1):"));
        assert!(
            listed
                .content
                .contains("aliases: Add calendar synchronization")
        );

        let edited = registry.invoke(
            "edit_feature_request",
            "{\"identifier\":\"add calendar sync\",\"status\":\"closed\",\"description\":\"Sync Elroy tasks to a calendar provider with account selection.\"}",
        );
        assert!(!edited.is_error);
        assert!(
            edited
                .content
                .contains("Updated feature request: Add calendar sync")
        );

        let records = list_feature_requests(&home).expect("feature requests should list");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, "closed");
        assert_eq!(
            records[0].summary,
            "Sync Elroy tasks to a calendar provider with account selection."
        );
        assert!(
            records[0]
                .supporting_context
                .as_deref()
                .is_some_and(|content| content.contains("Edited by user token: local-user"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_reset_context_messages() {
        let unique = format!(
            "elroy-rs-app-reset-context-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "hello"),
                ConversationMessage::new(MessageRole::Assistant, "hi"),
            ],
        )
        .expect("messages should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let registry = build_live_tool_registry(&config);
        let reset = registry.invoke("reset_messages", "{}");
        assert!(!reset.is_error);
        assert_eq!(reset.content, "Context reset complete");

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].role, MessageRole::System);
        assert!(
            stored[0]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("I am Elroy"))
        );

        let refresh = registry.invoke("refresh_system_instructions", "{}");
        assert!(!refresh.is_error);
        assert_eq!(refresh.content, "System instruction refresh complete");
        let refreshed =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(refreshed.len(), 1);
        assert_eq!(refreshed[0].role, MessageRole::System);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn app_runtime_load_context_messages_repairs_system_message_placement() {
        let unique = format!(
            "elroy-rs-app-repair-system-placement-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "hello"),
                ConversationMessage::new(MessageRole::System, "stale system"),
                ConversationMessage::new(MessageRole::Assistant, "hi"),
            ],
        )
        .expect("messages should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let runtime = AppRuntime::new(config);
        let repaired = runtime
            .load_context_messages()
            .expect("context messages should load");

        assert_eq!(repaired.len(), 3);
        assert_eq!(repaired[0].role, MessageRole::System);
        assert!(
            repaired[0]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("I am Elroy"))
        );
        assert_eq!(repaired[1].role, MessageRole::User);
        assert_eq!(repaired[2].role, MessageRole::Assistant);

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored, repaired);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn app_runtime_load_context_messages_inserts_synthetic_first_user_for_anthropic() {
        let unique = format!(
            "elroy-rs-app-repair-first-user-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "hello first",
            )],
        )
        .expect("messages should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.chat_model = "claude-sonnet-4-20250514".to_string();

        let runtime = AppRuntime::new(config);
        let repaired = runtime
            .load_context_messages()
            .expect("context messages should load");

        assert_eq!(repaired.len(), 3);
        assert_eq!(repaired[0].role, MessageRole::System);
        assert_eq!(repaired[1].role, MessageRole::User);
        assert_eq!(
            repaired[1].content.as_deref(),
            Some(SYNTHETIC_FIRST_USER_MESSAGE)
        );
        assert_eq!(repaired[2].role, MessageRole::Assistant);
        assert_eq!(repaired[2].content.as_deref(), Some("hello first"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn load_snapshot_filters_synthetic_first_user_line() {
        let unique = format!(
            "elroy-rs-app-snapshot-filter-synthetic-user-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::System, "system"),
                ConversationMessage::new(MessageRole::User, SYNTHETIC_FIRST_USER_MESSAGE),
                ConversationMessage::new(MessageRole::Assistant, "hello"),
            ],
        )
        .expect("messages should persist");
        drop(connection);

        let snapshot = AppRuntime::new(config)
            .load_snapshot()
            .expect("snapshot should load");
        assert!(
            !snapshot
                .conversation_lines
                .iter()
                .any(|line| line.contains(SYNTHETIC_FIRST_USER_MESSAGE))
        );
        assert!(
            snapshot
                .conversation_lines
                .iter()
                .any(|line| line == "assistant: hello")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn load_snapshot_exposes_plain_agenda_input_completions_only() {
        let unique = format!(
            "elroy-rs-app-snapshot-input-completions-{}",
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
            agenda_dir.join("desk reset.md"),
            "---\ndate: 2025-05-16\ncompleted: false\nstatus: created\n---\n\nDesk reset\n",
        )
        .expect("plain agenda item should persist");
        fs::write(
            agenda_dir.join("call-mom.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after dinner\n---\n\nCall Mom\n",
        )
        .expect("triggered task should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let runtime = AppRuntime::new(config);
        let snapshot = runtime.load_snapshot().expect("snapshot should load");

        assert!(
            snapshot
                .input_completions
                .contains(&"desk reset".to_string())
        );
        assert!(snapshot.input_completions.contains(&"/help".to_string()));
        assert!(
            snapshot
                .input_completions
                .contains(&"/reset_messages".to_string())
        );
        assert!(!snapshot.input_completions.contains(&"call mom".to_string()));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn handle_slash_command_executes_and_launches_forms() {
        let unique = format!(
            "elroy-rs-app-slash-command-exec-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;

        let runtime = AppRuntime::new(config);
        let TuiSlashCommandAction::Execute(help_execution) = runtime
            .handle_slash_command("/help")
            .expect("slash command should execute")
        else {
            panic!("help should execute immediately");
        };
        assert_eq!(
            help_execution,
            TuiCommandExecution {
                command_name: "get_help".to_string(),
                display_name: "help".to_string(),
                values: vec![],
            }
        );
        let help_snapshot = runtime
            .execute_command(
                &help_execution.command_name,
                &help_execution.display_name,
                &help_execution.values,
            )
            .expect("help command should execute");
        assert_eq!(
            help_snapshot.status.as_deref(),
            Some("slash command executed: /help")
        );
        assert!(
            help_snapshot
                .conversation_lines
                .last()
                .is_some_and(|line| line.starts_with("tool result: "))
        );
        assert!(
            help_snapshot
                .conversation_lines
                .last()
                .is_some_and(|line| line.contains("get_help"))
        );

        let create_memory = build_live_tool_registry(&runtime.config).invoke(
            "create_memory",
            "{\"name\":\"runner\",\"text\":\"Remember the training plan.\"}",
        );
        assert!(!create_memory.is_error);

        let TuiSlashCommandAction::Execute(shown_execution) = runtime
            .handle_slash_command("/show_memory runner")
            .expect("parameterized slash command should execute")
        else {
            panic!("show_memory should execute when all values are provided");
        };
        assert_eq!(
            shown_execution,
            TuiCommandExecution {
                command_name: "show_memory".to_string(),
                display_name: "show_memory".to_string(),
                values: vec![("memory_name".to_string(), "runner".to_string())],
            }
        );
        let shown_snapshot = runtime
            .execute_command(
                &shown_execution.command_name,
                &shown_execution.display_name,
                &shown_execution.values,
            )
            .expect("show_memory command should execute");
        assert_eq!(
            shown_snapshot.status.as_deref(),
            Some("slash command executed: /show_memory")
        );
        assert!(
            shown_snapshot
                .conversation_lines
                .last()
                .is_some_and(|line| line.contains("Remember the training plan."))
        );

        let TuiSlashCommandAction::OpenForm(missing_form) = runtime
            .handle_slash_command("/show_memory")
            .expect("underspecified slash command should stay local")
        else {
            panic!("underspecified known command should open a form");
        };
        assert_eq!(missing_form.command_name, "show_memory");
        assert_eq!(missing_form.parameters.len(), 1);
        assert_eq!(missing_form.parameters[0].name, "memory_name");
        assert!(missing_form.initial_values.is_empty());

        let TuiSlashCommandAction::OpenForm(prefilled_form) = runtime
            .handle_slash_command("/create_memory trip")
            .expect("partially specified slash command should open a form")
        else {
            panic!("create_memory with one value should open a form");
        };
        assert_eq!(prefilled_form.command_name, "create_memory");
        assert_eq!(
            prefilled_form.initial_values,
            vec![("name".to_string(), "trip".to_string())]
        );
        assert!(
            prefilled_form
                .parameters
                .iter()
                .map(|parameter| parameter.name.as_str())
                .eq(["name", "text"])
        );
        assert!(
            runtime
                .handle_slash_command("/missing_command")
                .is_err_and(|error| error.to_string() == "Invalid command: missing_command")
        );

        let submitted_snapshot = runtime
            .execute_command(
                "create_memory",
                "create_memory",
                &[
                    ("name".to_string(), "trip".to_string()),
                    ("text".to_string(), "Aisle seats.".to_string()),
                ],
            )
            .expect("command form submit should execute");
        assert_eq!(
            submitted_snapshot.status.as_deref(),
            Some("New memory created: trip")
        );
        assert!(
            submitted_snapshot
                .conversation_lines
                .iter()
                .all(|line| !line.contains("New memory created: trip"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn command_palette_entries_and_launch_path_cover_help_and_forms() {
        let unique = format!(
            "elroy-rs-app-command-palette-{}",
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
        create_agenda_file(
            &agenda_dir,
            "Trip note",
            "Remember the aisle seat preference.",
            Some("2026-05-16"),
            None,
            None,
        )
        .expect("agenda item should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;

        let runtime = AppRuntime::new(config);
        let entries = runtime
            .load_command_palette_entries()
            .expect("command palette entries should load");
        assert!(entries.iter().any(|entry| {
            entry.title == "/help"
                && entry.action == TuiCommandPaletteAction::ToolCommand("get_help".to_string())
        }));

        let TuiSlashCommandAction::OpenForm(form) = runtime
            .launch_named_command("create_memory")
            .expect("parameterized command should launch a form")
        else {
            panic!("create_memory should launch a form from the palette path");
        };
        assert_eq!(form.command_name, "create_memory");
        assert!(
            form.parameters
                .iter()
                .map(|parameter| parameter.name.as_str())
                .eq(["name", "text"])
        );
        assert_eq!(
            form.parameters
                .first()
                .expect("first parameter should exist")
                .suggestions,
            vec!["trip note".to_string()]
        );

        let TuiSlashCommandAction::OpenForm(show_task_form) = runtime
            .launch_named_command("show_task")
            .expect("show_task should launch a form")
        else {
            panic!("show_task should launch a form from the palette path");
        };
        assert_eq!(
            show_task_form
                .parameters
                .first()
                .expect("show_task parameter should exist")
                .suggestions,
            vec!["trip note".to_string()]
        );

        let TuiSlashCommandAction::Execute(execution) = runtime
            .launch_named_command("get_help")
            .expect("zero-arg command should execute from the palette path")
        else {
            panic!("get_help should execute immediately from the palette path");
        };
        assert_eq!(
            execution,
            TuiCommandExecution {
                command_name: "get_help".to_string(),
                display_name: "help".to_string(),
                values: vec![],
            }
        );
        let snapshot = runtime
            .execute_command(
                &execution.command_name,
                &execution.display_name,
                &execution.values,
            )
            .expect("get_help command should execute from the palette path");
        assert_eq!(
            snapshot.status.as_deref(),
            Some("slash command executed: /help")
        );
        assert!(
            snapshot
                .conversation_lines
                .last()
                .is_some_and(|line| line.starts_with("tool result: "))
        );

        let toast_snapshot = runtime
            .execute_command(
                "create_memory",
                "create_memory",
                &[
                    ("name".to_string(), "trip".to_string()),
                    ("text".to_string(), "Aisle seats.".to_string()),
                ],
            )
            .expect("create_memory should execute from the palette path");
        assert_eq!(
            toast_snapshot.status.as_deref(),
            Some("New memory created: trip")
        );
        assert!(
            toast_snapshot
                .conversation_lines
                .iter()
                .all(|line| !line.contains("New memory created: trip"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_disable_base_tools_from_config() {
        let unique = format!(
            "elroy-rs-app-no-base-tools-{}",
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
        config.include_base_tools = false;

        let registry = build_live_tool_registry(&config);
        let result = registry.invoke(
            "create_memory",
            "{\"name\":\"Runner Notes\",\"text\":\"Remember the hill workout\"}",
        );

        assert!(result.is_error);
        assert!(result.content.contains("unknown tool"));
        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_exclude_specific_tools_from_config() {
        let mut config = AppConfig::defaults();
        config.exclude_tools = vec![
            "get_user_preferred_name".to_string(),
            "get_help".to_string(),
        ];

        let registry = build_live_tool_registry(&config);

        assert!(
            !registry
                .specs()
                .iter()
                .any(|tool| tool.name == "get_user_preferred_name")
        );
        assert!(!registry.specs().iter().any(|tool| tool.name == "get_help"));

        let result = registry.invoke("get_user_preferred_name", "{}");
        assert!(result.is_error);
        assert!(result.content.contains("unknown tool"));
    }

    #[test]
    fn live_tool_registry_can_use_filesystem_and_time_base_tools() {
        let unique = format!(
            "elroy-rs-app-base-tools-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        let notes_dir = home.join("notes");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::create_dir_all(&notes_dir).expect("notes dir should be created");
        fs::write(
            notes_dir.join("todo.txt"),
            "alpha\nbeta\ngamma\ndelta\nepsilon\n",
        )
        .expect("fixture file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;

        let previous_dir = std::env::current_dir().expect("cwd should resolve");
        std::env::set_current_dir(&home).expect("cwd should switch");

        let registry = build_live_tool_registry(&config);
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
    fn live_tool_registry_can_schedule_restart_when_supported() {
        crate::disable_session_restart_support();
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);

        let unavailable = registry.invoke("restart_session", "{}");
        assert!(unavailable.is_error);
        assert!(
            unavailable
                .content
                .contains("Session restart is not available in this Elroy runtime.")
        );

        let runtime = AppRuntime::new(config);
        runtime.enable_restart_support();
        let scheduled = registry.invoke(
            "restart_session",
            "{\"resume_message\":\"Restarted successfully. Ready to continue.\"}",
        );
        assert!(!scheduled.is_error);
        assert_eq!(
            scheduled.content,
            "Restart scheduled. Elroy will restart after this response completes."
        );
        assert_eq!(
            runtime.consume_restart_request().as_deref(),
            Some("Restarted successfully. Ready to continue.")
        );
        runtime.disable_restart_support();
    }

    #[test]
    fn live_tool_registry_can_print_config_and_tail_logs() {
        let unique = format!(
            "elroy-rs-app-developer-tools-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let logs_dir = home.join("logs");
        let database_path = home.join("elroy.db");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::create_dir_all(&logs_dir).expect("logs dir should be created");
        fs::write(
            logs_dir.join("elroy.log"),
            "line one\nline two\nline three\n",
        )
        .expect("log file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        config.config_path = home.join("elroy.conf.yaml");
        config.openai_api_key = Some("openai-secret".to_string());
        config.anthropic_api_key = Some("anthropic-secret".to_string());

        let registry = build_live_tool_registry(&config);
        let printed = registry.invoke("print_config", "{}");
        let tailed = registry.invoke("tail_elroy_logs", "{\"lines\":2}");

        assert!(!printed.is_error);
        assert!(printed.content.contains("Elroy Configuration"));
        assert!(printed.content.contains("Section"));
        assert!(printed.content.contains("Setting"));
        assert!(printed.content.contains("Value"));
        assert!(printed.content.contains("Chat Model"));
        assert!(printed.content.contains("Config Path"));
        assert!(printed.content.contains("Chat API Key"));
        assert!(printed.content.contains("********"));
        assert!(printed.content.contains("Anthropic API Key"));
        assert!(printed.content.contains("Exclude Tools"));
        assert!(printed.content.contains("(none)"));
        assert!(printed.content.contains("Reflect"));
        assert!(printed.content.contains("Memories Between Consolidation"));
        assert!(
            printed
                .content
                .contains("L2 Memory Relevance Distance Threshold")
        );
        assert!(printed.content.contains("Memory Cluster Similarity"));
        assert!(printed.content.contains("Max Memory Cluster Size"));
        assert!(printed.content.contains("Min Memory Cluster Size"));
        assert!(!tailed.is_error);
        assert_eq!(tailed.content, "line two\nline three\n");

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_can_print_help() {
        let config = AppConfig::defaults();
        let registry = build_live_tool_registry(&config);

        let help = registry.invoke("get_help", "{}");

        assert!(!help.is_error);
        assert!(help.content.contains("Available Slash Commands"));
        assert!(help.content.contains("Command"));
        assert!(help.content.contains("Description"));
        assert!(help.content.contains("get_help"));
        assert!(
            help.content
                .contains("Print the available system commands.")
        );
        assert!(help.content.contains("print_config"));
        assert!(
            help.content
                .contains("Print the current Elroy configuration in a formatted report.")
        );
        assert!(help.content.contains("tail_elroy_logs"));
        assert!(
            help.content
                .contains("Return the last lines of the Elroy log file.")
        );
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
            "{\"name\":\"Job Search\",\"text\":\"Reach out to three contacts\",\"item_date\":\"2026-05-20\",\"trigger_context\":\"after breakfast\"}",
        );
        assert!(!created.is_error);
        assert_eq!(created.content, "Task 'job search' has been created.");
        assert!(agenda_dir.join("job_search.md").exists());
        let created_text =
            fs::read_to_string(agenda_dir.join("job_search.md")).expect("task file should read");
        assert!(created_text.contains("date: 2026-05-20"));
        assert!(created_text.contains("trigger_context: after breakfast"));
        let duplicate_created = registry.invoke(
            "create_task",
            "{\"name\":\"Job Search\",\"text\":\"Reach out to one contact\"}",
        );
        assert!(duplicate_created.is_error);
        assert_eq!(
            duplicate_created.content,
            "Task 'Job Search' already exists"
        );
        let blank_name = registry.invoke(
            "create_task",
            "{\"name\":\"   \",\"text\":\"This should fail\"}",
        );
        assert!(blank_name.is_error);
        assert_eq!(blank_name.content, "Task name cannot be empty");
        let invalid_date = registry.invoke(
            "create_task",
            "{\"name\":\"Bad Date\",\"text\":\"This should fail\",\"item_date\":\"2026/05/20\"}",
        );
        assert!(invalid_date.is_error);
        assert_eq!(
            invalid_date.content,
            "Invalid date format '2026/05/20'. Use YYYY-MM-DD."
        );
        let past_trigger = registry.invoke(
            "create_task",
            "{\"name\":\"Old Reminder\",\"text\":\"This should fail\",\"trigger_datetime\":\"2000-01-01 09:00\"}",
        );
        assert!(past_trigger.is_error);
        assert!(
            past_trigger
                .content
                .contains("Attempted to create a due item for")
        );
        assert!(past_trigger.content.contains("which is in the past"));

        let triggered = registry.invoke("list_triggered_tasks", "{\"limit\":10}");
        assert!(!triggered.is_error);
        assert!(triggered.content.contains("job search"));
        assert!(triggered.content.contains("after breakfast"));
        let triggered_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!triggered_context.is_error);
        assert!(
            !triggered_context
                .content
                .contains("context-task:job search")
        );

        let today = registry.invoke("list_today_tasks", "{\"limit\":10}");
        assert!(!today.is_error);
        assert!(!today.content.contains("job search"));

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
        let task_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!task_context.is_error);
        assert!(task_context.content.contains("context-task:inbox zero"));
        let plain_updated = registry.invoke(
            "update_task_text",
            "{\"name\":\"inbox zero\",\"text\":\"Clear email backlog tonight\"}",
        );
        assert!(!plain_updated.is_error);
        let updated_task_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!updated_task_context.is_error);
        assert!(
            updated_task_context
                .content
                .contains("context-task:inbox zero")
        );
        assert!(
            updated_task_context
                .content
                .contains("Clear email backlog tonight")
        );
        let plain_renamed = registry.invoke(
            "rename_task",
            "{\"old_name\":\"inbox zero\",\"new_name\":\"Inbox Clean\"}",
        );
        assert!(!plain_renamed.is_error);
        let renamed_task_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!renamed_task_context.is_error);
        assert!(
            !renamed_task_context
                .content
                .contains("context-task:inbox zero")
        );
        assert!(
            renamed_task_context
                .content
                .contains("context-task:inbox clean")
        );
        let plain_completed = registry.invoke(
            "complete_task",
            "{\"name\":\"inbox clean\",\"closing_comment\":\"done\"}",
        );
        assert!(!plain_completed.is_error);
        let completed_task_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!completed_task_context.is_error);
        assert!(
            !completed_task_context
                .content
                .contains("context-task:inbox clean")
        );

        let deleted_created = registry.invoke(
            "create_task",
            "{\"name\":\"Desk Reset\",\"text\":\"Tidy the desk\"}",
        );
        assert!(!deleted_created.is_error);
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let mut transcript = elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN)
            .expect("context should load");
        transcript.insert(
            0,
            ConversationMessage::new(MessageRole::User, "keep context"),
        );
        elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
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
        let stripped_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!stripped_context.is_error);
        assert!(!stripped_context.content.contains("context-task:desk reset"));
        assert!(stripped_context.content.contains("keep context"));
        let recreated = registry.invoke(
            "create_task",
            "{\"name\":\"Desk Reset\",\"text\":\"Tidy the desk again\"}",
        );
        assert!(!recreated.is_error);
        assert_eq!(recreated.content, "Task 'desk reset' has been created.");
        let recreated_shown = registry.invoke("show_task", "{\"name\":\"desk reset\"}");
        assert!(!recreated_shown.is_error);
        assert!(recreated_shown.content.contains("Tidy the desk again"));
        let recreated_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!recreated_context.is_error);
        assert!(
            recreated_context
                .content
                .contains("context-task:desk reset")
        );
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
            "elroy-rs-app-due-task-filtering-{}",
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

        let registry = build_live_tool_registry(&config);
        let due = registry.invoke("list_due_tasks", "{}");
        assert!(!due.is_error);
        assert!(due.content.contains("past due"));
        assert!(!due.content.contains("future due"));
        assert!(!due.content.contains("context-only"));

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
        clear_background_status(&codex_background_status_key("thread-123"));
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
            thread::sleep(Duration::from_millis(200));
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
        assert_eq!(
            get_background_status().as_deref(),
            Some("codex session thread-123 running...")
        );
        wait_for_codex_status(&database_path, "thread-123", "completed");
        wait_for_background_status_message("processing codex session thread-123 completion...");
        thread::sleep(Duration::from_millis(250));
        assert!(get_background_status().is_none());

        let resumed = registry.invoke(
            "resume_codex_session",
            "{\"session_id\":\"thread-123\",\"prompt\":\"follow up\"}",
        );
        assert!(!resumed.is_error);
        assert!(resumed.content.contains("\"status\":\"running\""));
        assert_eq!(
            get_background_status().as_deref(),
            Some("codex session thread-123 running...")
        );
        wait_for_codex_status(&database_path, "thread-123", "completed");
        wait_for_background_status_message("processing codex session thread-123 completion...");
        thread::sleep(Duration::from_millis(250));
        assert!(get_background_status().is_none());

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

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path;

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_due_item",
            "{\"name\":\"Call Mom\",\"text\":\"Call mom tonight\",\"trigger_context\":\"after dinner\"}",
        );
        assert!(!created.is_error);
        let seeded_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!seeded_context.is_error);
        assert!(seeded_context.content.contains("context-due-item:call mom"));
        let updated = registry.invoke(
            "update_due_item_text",
            "{\"name\":\"call mom\",\"new_text\":\"Call mom after dinner\"}",
        );
        assert!(!updated.is_error);
        assert_eq!(
            updated.content,
            "Due item 'call mom' text has been updated."
        );
        let updated_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!updated_context.is_error);
        assert!(
            updated_context
                .content
                .contains("context-due-item:call mom")
        );
        assert!(updated_context.content.contains("Call mom after dinner"));
        let missing_updated = registry.invoke(
            "update_due_item_text",
            "{\"name\":\"missing\",\"new_text\":\"No-op\"}",
        );
        assert!(missing_updated.is_error);
        assert_eq!(
            missing_updated.content,
            "Due item 'missing' not found. Valid items: call mom"
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
        let renamed_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!renamed_context.is_error);
        assert!(
            !renamed_context
                .content
                .contains("context-due-item:call mom")
        );
        assert!(
            renamed_context
                .content
                .contains("context-due-item:call parents")
        );
        let missing_renamed = registry.invoke(
            "rename_due_item",
            "{\"old_name\":\"missing\",\"new_name\":\"Call Family\"}",
        );
        assert!(missing_renamed.is_error);
        assert_eq!(
            missing_renamed.content,
            "Active due item 'missing' not found. Active items: call parents"
        );
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
        let completed_context = registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!completed_context.is_error);
        assert!(
            !completed_context
                .content
                .contains("context-due-item:call parents")
        );
        let missing_completed = registry.invoke("complete_due_item", "{\"name\":\"missing\"}");
        assert!(missing_completed.is_error);
        assert_eq!(
            missing_completed.content,
            "Active due item 'missing' not found. Active due items: "
        );

        fs::write(
            agenda_dir.join("pay_bill.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: tomorrow\n---\n\nPay bill\n",
        )
        .expect("second due item file should be written");
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        let deleted = registry.invoke(
            "delete_due_item",
            "{\"name\":\"pay bill\",\"closing_comment\":\"paid online\"}",
        );
        assert!(!deleted.is_error);
        assert_eq!(
            deleted.content,
            "Due item 'pay bill' has been deleted. Comment: paid online"
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
        let completed_recreated_context =
            registry.invoke("show_context_messages", "{\"limit\":20}");
        assert!(!completed_recreated_context.is_error);
        assert!(
            completed_recreated_context
                .content
                .contains("context-due-item:call parents")
        );
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
            trigger_datetime: Some("2000-01-01T09:00:00".to_string()),
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
            Some("get_due_items")
        );
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert!(messages[1].content.as_deref().is_some_and(|content| {
            content.contains("Call mom tonight")
                && content.contains("delete_due_item")
                && content.contains("⏰ DUE ITEM")
                && content.contains("2000-01-01 09:00:00")
        }));
    }

    #[test]
    fn due_item_context_messages_skip_context_only_items() {
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

        assert!(messages.is_empty());
    }

    #[test]
    fn due_item_context_messages_include_multiple_due_items() {
        let messages = due_item_context_messages(&[
            AgendaItemRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "call mom".to_string(),
                file_path: "/tmp/call_mom.md".to_string(),
                agenda_date: Some("unscheduled".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                trigger_datetime: Some("2000-01-01T09:00:00".to_string()),
                trigger_context: None,
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Call mom tonight".to_string(),
                is_active: true,
                updated_at_unix: 1,
            },
            AgendaItemRecord {
                id: 2,
                legacy_frontmatter_id: None,
                name: "pay rent".to_string(),
                file_path: "/tmp/pay_rent.md".to_string(),
                agenda_date: Some("unscheduled".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                trigger_datetime: Some("2000-01-02T09:00:00".to_string()),
                trigger_context: None,
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Pay rent".to_string(),
                is_active: true,
                updated_at_unix: 2,
            },
        ]);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[1].role, MessageRole::Tool);
        let content = messages[1].content.as_deref().unwrap_or_default();
        assert!(content.contains("Call mom tonight"));
        assert!(content.contains("Pay rent"));
        assert!(content.contains("2000-01-01 09:00:00"));
        assert!(content.contains("2000-01-02 09:00:00"));
    }

    #[test]
    fn select_recalled_due_items_prefers_trigger_context_overlap() {
        let due_items = vec![
            AgendaItemRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "Payroll Follow-up".to_string(),
                file_path: "/tmp/payroll.md".to_string(),
                agenda_date: Some("unscheduled".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                trigger_datetime: None,
                trigger_context: Some("after payroll email".to_string()),
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Reply to payroll".to_string(),
                is_active: true,
                updated_at_unix: 20,
            },
            AgendaItemRecord {
                id: 2,
                legacy_frontmatter_id: None,
                name: "Dinner".to_string(),
                file_path: "/tmp/dinner.md".to_string(),
                agenda_date: Some("unscheduled".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                trigger_datetime: None,
                trigger_context: Some("after dinner".to_string()),
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Call family".to_string(),
                is_active: true,
                updated_at_unix: 10,
            },
        ];

        let recalled = select_recalled_due_items(
            "I just got the payroll email",
            &due_items,
            "2026-05-15T12:00:00",
            2,
        );

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].name, "Payroll Follow-up");
    }

    #[test]
    fn select_due_items_by_overlap_can_include_not_yet_due_items_for_search() {
        let due_items = vec![AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "Payroll Follow-up".to_string(),
            file_path: "/tmp/payroll.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: Some("2099-01-01T09:00:00".to_string()),
            trigger_context: Some("after payroll email".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Reply to payroll".to_string(),
            is_active: true,
            updated_at_unix: 20,
        }];

        let recalled =
            select_due_items_by_overlap("I just got the payroll email", &due_items, 2, None);

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].name, "Payroll Follow-up");
    }

    #[test]
    fn select_recalled_due_items_skips_time_due_items() {
        let due_items = vec![AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "Payroll Follow-up".to_string(),
            file_path: "/tmp/payroll.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: Some("2000-01-01T09:00:00".to_string()),
            trigger_context: Some("after payroll email".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Reply to payroll".to_string(),
            is_active: true,
            updated_at_unix: 20,
        }];

        let recalled = select_recalled_due_items(
            "I just got the payroll email",
            &due_items,
            "2026-05-15T12:00:00",
            2,
        );

        assert!(recalled.is_empty());
    }

    #[test]
    fn recall_due_item_context_messages_creates_contextual_tool_message() {
        let transcript = vec![ConversationMessage::new(
            MessageRole::Assistant,
            "Tell me when payroll follows up.",
        )];
        let due_items = vec![AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "Payroll Follow-up".to_string(),
            file_path: "/tmp/payroll.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: None,
            trigger_context: Some("after payroll email".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Reply to payroll".to_string(),
            is_active: true,
            updated_at_unix: 20,
        }];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0]
                .tool_calls
                .as_ref()
                .map(|calls| calls[0].id.as_str()),
            Some("context-due-item:payroll follow-up")
        );
        assert!(messages[1].content.as_deref().is_some_and(|content| {
            content.contains("DUE ITEM") && content.contains("Reply to payroll")
        }));
    }

    #[test]
    fn recall_due_item_context_messages_skip_already_pinned_due_items() {
        let transcript = context_due_item_tool_messages(&AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "payroll follow up".to_string(),
            file_path: "/tmp/payroll.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: None,
            trigger_context: Some("after payroll email".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Reply to payroll".to_string(),
            is_active: true,
            updated_at_unix: 20,
        });
        let due_items = vec![AgendaItemRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "payroll follow up".to_string(),
            file_path: "/tmp/payroll.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: None,
            trigger_context: Some("after payroll email".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Reply to payroll".to_string(),
            is_active: true,
            updated_at_unix: 20,
        }];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
        );

        assert!(messages.is_empty());
    }

    #[test]
    fn live_tool_registry_search_memories_can_return_due_items() {
        let unique = format!(
            "elroy-rs-app-search-memories-due-items-{}",
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
            agenda_dir.join("payroll_follow_up.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after payroll email\n---\n\nReply to payroll\n",
        )
        .expect("due item should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let search = registry.invoke(
            "search_memories",
            "{\"query\":\"payroll email\",\"limit\":5}",
        );

        assert!(!search.is_error);
        assert!(search.content.contains("Search Results"));
        assert!(
            search
                .content
                .contains("DueItem | payroll follow up | Reply to payroll")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_search_memories_can_return_plain_agenda_items() {
        let unique = format!(
            "elroy-rs-app-search-memories-agenda-items-{}",
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
            agenda_dir.join("draft_launch_recap.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\n---\n\nDraft the launch recap for the product update.\n",
        )
        .expect("agenda item should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let search = registry.invoke(
            "search_memories",
            "{\"query\":\"product launch recap\",\"limit\":5}",
        );

        assert!(!search.is_error);
        assert!(search.content.contains("Search Results"));
        assert!(search.content.contains(
            "AgendaItem | draft launch recap | Draft the launch recap for the product update."
        ));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_search_memories_caps_results_to_two_per_type() {
        let unique = format!(
            "elroy-rs-app-search-memories-limit-{}",
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

        for idx in 1..=3 {
            fs::write(
                memory_dir.join(format!("project_phoenix_note_{idx}.md")),
                format!("Project Phoenix launch update note {idx} with launch planning details.\n"),
            )
            .expect("memory should be written");
            fs::write(
                agenda_dir.join(format!("project_phoenix_due_{idx}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after the Project Phoenix launch update\n---\n\nProject Phoenix due item {idx}\n"
                ),
            )
            .expect("due item should be written");
            fs::write(
                agenda_dir.join(format!("project_phoenix_agenda_{idx}.md")),
                format!(
                    "---\ndate: 2026-05-2{idx}\ncompleted: false\nstatus: created\n---\n\nProject Phoenix agenda item {idx}\n"
                ),
            )
            .expect("agenda item should be written");
        }

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let search = registry.invoke(
            "search_memories",
            "{\"query\":\"Project Phoenix launch update\",\"limit\":5}",
        );

        assert!(!search.is_error);
        assert_eq!(
            search
                .content
                .lines()
                .filter(|line| line.starts_with("- Memory | "))
                .count(),
            2
        );
        assert_eq!(
            search
                .content
                .lines()
                .filter(|line| line.starts_with("- DueItem | "))
                .count(),
            2
        );
        assert_eq!(
            search
                .content
                .lines()
                .filter(|line| line.starts_with("- AgendaItem | "))
                .count(),
            2
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_examine_memories_can_return_memory_and_due_item_sections() {
        let unique = format!(
            "elroy-rs-app-examine-memories-{}",
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
            memory_dir.join("running_notes.md"),
            "User is training for a marathon in October.\n",
        )
        .expect("memory should be written");
        fs::write(
            agenda_dir.join("running_follow_up.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after the marathon training check-in\n---\n\nAsk about long run recovery.\n",
        )
        .expect("due item should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let result = registry.invoke(
            "examine_memories",
            "{\"question\":\"What do I know about the marathon training check-in?\",\"limit\":5}",
        );

        assert!(!result.is_error);
        assert!(result.content.contains("# Memory: running notes"));
        assert!(
            result
                .content
                .contains("get_source_content_for_memory(running notes, idx)")
        );
        assert!(result.content.contains("marathon in October"));
        assert!(result.content.contains("# Due Item: running follow up"));
        assert!(result.content.contains("long run recovery"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_examine_memories_can_return_agenda_item_sections() {
        let unique = format!(
            "elroy-rs-app-examine-memories-agenda-items-{}",
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
            agenda_dir.join("plan_team_offsite.md"),
            "---\ndate: 2026-05-21\ncompleted: false\nstatus: created\n---\n\nPlan the team offsite agenda and venue shortlist.\n",
        )
        .expect("agenda item should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let result = registry.invoke(
            "examine_memories",
            "{\"question\":\"What do I know about the offsite venue shortlist?\",\"limit\":5}",
        );

        assert!(!result.is_error);
        assert!(result.content.contains("# Agenda Item: plan team offsite"));
        assert!(
            result
                .content
                .contains("team offsite agenda and venue shortlist")
        );
        assert!(result.content.contains("Agenda date: 2026-05-21"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn live_tool_registry_examine_memories_caps_results_to_two_per_type() {
        let unique = format!(
            "elroy-rs-app-examine-memories-limit-{}",
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

        for idx in 1..=3 {
            fs::write(
                memory_dir.join(format!("project_phoenix_memory_{idx}.md")),
                format!("Project Phoenix planning memory {idx} with launch checklist notes.\n"),
            )
            .expect("memory should be written");
            fs::write(
                agenda_dir.join(format!("project_phoenix_follow_up_{idx}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after the Project Phoenix launch planning check-in\n---\n\nProject Phoenix due follow-up {idx}\n"
                ),
            )
            .expect("due item should be written");
            fs::write(
                agenda_dir.join(format!("project_phoenix_plan_{idx}.md")),
                format!(
                    "---\ndate: 2026-06-0{idx}\ncompleted: false\nstatus: created\n---\n\nProject Phoenix agenda planning item {idx}\n"
                ),
            )
            .expect("agenda item should be written");
        }

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = build_live_tool_registry(&config);
        let result = registry.invoke(
            "examine_memories",
            "{\"question\":\"What do I know about the Project Phoenix launch planning check-in?\",\"limit\":5}",
        );

        assert!(!result.is_error);
        assert_eq!(result.content.matches("# Memory: ").count(), 2);
        assert_eq!(result.content.matches("# Due Item: ").count(), 2);
        assert_eq!(result.content.matches("# Agenda Item: ").count(), 2);

        fs::remove_dir_all(home).expect("home should be removed");
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
        let config = AppConfig::defaults();
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "A background Codex session completed.",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: false,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
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
    fn run_prompt_with_model_and_registry_surfaces_and_cleans_up_due_items() {
        let unique = format!(
            "elroy-rs-app-due-item-prompt-integration-{}",
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
            agenda_dir.join("medicine_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\n---\n\nTake your daily medicine\n",
        )
        .expect("due item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir.clone();
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let registry = build_live_tool_registry(&config);
        let model = DueItemSurfacingModel::new();
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "Hi, how are you doing today?",
            &model,
            registry,
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallRequested(call)
                if call.name == "delete_due_item"
                    && call.arguments_json == "{\"name\":\"medicine reminder\"}"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantToolResult { content, is_error }
                if !is_error
                    && content.contains("Due item 'medicine reminder' has been deleted.")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content.contains("take your daily medicine")
        )));

        let active_due_items =
            list_active_due_items(&connection, 10).expect("due items should list");
        assert!(
            !active_due_items
                .iter()
                .any(|item| item.name == "medicine reminder")
        );
        assert!(!agenda_dir.join("medicine_reminder.md").exists());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_skips_future_due_item_context() {
        let unique = format!(
            "elroy-rs-app-future-due-item-prompt-{}",
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
            agenda_dir.join("future_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2999-01-01T09:00:00\n---\n\nThis is for tomorrow\n",
        )
        .expect("future due item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "How's the weather today?",
            &NoDueItemContextModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallRequested(call) if call.name == "delete_due_item"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content } if content == "Weather looks calm today."
        )));

        let active_due_items =
            list_active_due_items(&connection, 10).expect("due items should list");
        assert!(
            active_due_items
                .iter()
                .any(|item| item.name == "future reminder")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_stream_can_defer_self_reflection() {
        let unique = format!(
            "elroy-rs-app-deferred-self-reflection-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "Draft a reply to this message."),
                ConversationMessage::new(MessageRole::Assistant, "Here is a draft."),
            ],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "I will revise it.".to_string(),
        }]]);
        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        config.messages_between_self_reflection = 2;

        let mut stream = run_prompt_with_model_and_registry_stream(
            connection,
            home.clone(),
            "That's wrong. You forgot the main deadline.",
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: true,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
            Box::new(model),
            ExecutableToolRegistry::new(vec![]),
        )
        .expect("prompt should stream");

        while stream.next().is_some() {}
        let _snapshot = stream.into_snapshot().expect("snapshot should finalize");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert!(records.is_empty());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_stream_can_defer_auto_memory() {
        let unique = format!(
            "elroy-rs-app-deferred-auto-memory-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "We agreed to save this summary."),
                ConversationMessage::new(MessageRole::Assistant, "I'll remember it."),
            ],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Adding the missing detail now.".to_string(),
        }]]);
        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.messages_between_memory = 2;

        let mut stream = run_prompt_with_model_and_registry_stream(
            connection,
            home.clone(),
            "The main point was the Friday launch deadline.",
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: true,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
            Box::new(model),
            ExecutableToolRegistry::new(vec![]),
        )
        .expect("prompt should stream");

        while stream.next().is_some() {}
        let completion = stream
            .into_completion()
            .expect("completion should finalize");
        let deferred = completion
            .deferred_auto_memory
            .expect("auto memory should be deferred");

        let before_memories = crate::list_active_memories_in_scope(
            &open_sqlite_connection(&database_path).expect("database should reopen"),
            &memory_dir,
            10,
        )
        .expect("active memories should load");
        assert!(before_memories.is_empty());

        let runtime = AppRuntime::new(config);
        runtime
            .run_auto_memory_for_transcript(
                deferred.existing_transcript_len,
                deferred.transcript.clone(),
            )
            .expect("deferred auto memory should succeed");

        let mut reopened = open_sqlite_connection(&database_path).expect("database should reopen");
        let active_memories = crate::list_active_memories_in_scope(&reopened, &memory_dir, 10)
            .expect("active memories should load");
        assert_eq!(active_memories.len(), 1);
        let tracker =
            elroy_db::get_or_create_memory_operation_tracker(&mut reopened, LOCAL_USER_TOKEN)
                .expect("tracker should load");
        assert_eq!(tracker.messages_since_memory, 0);
        assert_eq!(tracker.memories_since_consolidation, 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_surfaces_multiple_due_items() {
        let unique = format!(
            "elroy-rs-app-multiple-due-items-prompt-{}",
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
            agenda_dir.join("reminder1.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\n---\n\nFirst due reminder\n",
        )
        .expect("first due item file should be written");
        fs::write(
            agenda_dir.join("reminder2.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T10:00:00\n---\n\nSecond due reminder\n",
        )
        .expect("second due item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "What's on my schedule today?",
            &MultipleDueItemsModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content.contains("First due reminder")
                    && content.contains("Second due reminder")
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_surfaces_hybrid_due_item_when_time_due() {
        let unique = format!(
            "elroy-rs-app-hybrid-due-item-prompt-{}",
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
            agenda_dir.join("hybrid_test.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\ntrigger_context: when user mentions work\n---\n\nHybrid reminder text\n",
        )
        .expect("hybrid due item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "What's happening?",
            &HybridDueItemModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content.contains("Hybrid reminder text")
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_surfaces_contextual_due_item() {
        let unique = format!(
            "elroy-rs-app-contextual-due-item-prompt-{}",
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
            agenda_dir.join("payroll_follow_up.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after payroll email\n---\n\nReply to payroll\n",
        )
        .expect("contextual due item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "I just got the payroll email.",
            &ContextualDueItemModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content.contains("Reply to payroll")
        )));
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        let due_item_tool_call_id = context_due_item_tool_call_id("payroll follow up");
        assert_eq!(
            stored
                .iter()
                .filter(|message| message_matches_tool_call_id(message, &due_item_tool_call_id))
                .count(),
            2
        );

        let second_events = run_prompt_with_model_and_registry(
            &mut connection,
            "I'm following up after that payroll email now.",
            &ContextualDueItemModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("second prompt should succeed");

        assert!(second_events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content.contains("Reply to payroll")
        )));
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(
            stored
                .iter()
                .filter(|message| message_matches_tool_call_id(message, &due_item_tool_call_id))
                .count(),
            2
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_persist_non_user_roles() {
        let unique = format!(
            "elroy-rs-app-system-role-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Acknowledged system bootstrap.".to_string(),
        }]]);
        let config = AppConfig::defaults();
        run_prompt_with_model_and_registry(
            &mut connection,
            "System bootstrap message",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::System,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("system-role prompt should succeed");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");

        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, MessageRole::System);
        assert_eq!(
            stored[0].content.as_deref(),
            Some("System bootstrap message")
        );
        assert_eq!(stored[1].role, MessageRole::Assistant);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_repairs_missing_system_message_before_turn() {
        let unique = format!(
            "elroy-rs-app-prompt-repair-system-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(MessageRole::User, "hello")],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "repaired reply".to_string(),
        }]]);
        let config = AppConfig::defaults();
        run_prompt_with_model_and_registry(
            &mut connection,
            "what next",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored[0].role, MessageRole::System);
        assert_eq!(stored[1].role, MessageRole::User);
        assert_eq!(stored[1].content.as_deref(), Some("hello"));
        assert_eq!(stored[2].role, MessageRole::User);
        assert_eq!(stored[2].content.as_deref(), Some("what next"));
        assert_eq!(stored[3].role, MessageRole::Assistant);
        assert_eq!(stored[3].content.as_deref(), Some("repaired reply"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_inserts_synthetic_first_user_for_anthropic() {
        let unique = format!(
            "elroy-rs-app-prompt-repair-first-user-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "assistant opened first",
            )],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "anthropic reply".to_string(),
        }]]);
        let mut config = AppConfig::defaults();
        config.chat_model = "claude-sonnet-4-20250514".to_string();
        run_prompt_with_model_and_registry(
            &mut connection,
            "continue",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: true,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored[0].role, MessageRole::System);
        assert_eq!(stored[1].role, MessageRole::User);
        assert_eq!(
            stored[1].content.as_deref(),
            Some(SYNTHETIC_FIRST_USER_MESSAGE)
        );
        assert_eq!(stored[2].role, MessageRole::Assistant);
        assert_eq!(stored[2].content.as_deref(), Some("assistant opened first"));
        assert_eq!(stored[3].role, MessageRole::User);
        assert_eq!(stored[3].content.as_deref(), Some("continue"));
        assert_eq!(stored[4].role, MessageRole::Assistant);
        assert_eq!(stored[4].content.as_deref(), Some("anthropic reply"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_triggers_self_reflection_feature_request() {
        let unique = format!(
            "elroy-rs-app-self-reflection-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "Draft a reply to this message."),
                ConversationMessage::new(MessageRole::Assistant, "Here is a draft."),
            ],
        )
        .expect("messages should persist");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "I will revise it.".to_string(),
        }]]);
        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        config.messages_between_self_reflection = 2;

        run_prompt_with_model_and_registry(
            &mut connection,
            "That's wrong. You forgot the main deadline.",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        let records = list_feature_requests(&home).expect("feature requests should load");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source, "self_reflection");
        assert!(
            records[0]
                .supporting_context
                .as_deref()
                .is_some_and(|value| value.contains("You forgot the main deadline."))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_auto_create_memory_on_message_threshold() {
        let unique = format!(
            "elroy-rs-app-auto-memory-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: "Test response 1".to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: "Test response 2".to_string(),
            }],
        ]);
        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.messages_between_memory = 3;

        run_prompt_with_model_and_registry(
            &mut connection,
            "Test message 1",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");
        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 2);

        run_prompt_with_model_and_registry(
            &mut connection,
            "Test message 2",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("second prompt should succeed");

        let memories =
            elroy_db::list_active_memories(&connection, 10).expect("memories should list");
        assert_eq!(memories.len(), 1);
        assert!(memories[0].body.contains("Test message 1"));
        assert!(memories[0].body.contains("Test response 2"));

        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 0);
        assert_eq!(tracker.memories_since_consolidation, 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_memory_tool_resets_auto_memory_tracker() {
        let unique = format!(
            "elroy-rs-app-create-memory-reset-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: "First response".to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: "Second response".to_string(),
            }],
        ]);
        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.messages_between_memory = 3;

        run_prompt_with_model_and_registry(
            &mut connection,
            "First message",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("first prompt should succeed");

        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 2);

        let registry = build_live_tool_registry(&config);
        let created = registry.invoke(
            "create_memory",
            "{\"name\":\"Manual memory\",\"text\":\"A manual memory\"}",
        );
        assert!(!created.is_error);
        assert_eq!(created.content, "New memory created: Manual memory");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 0);
        assert_eq!(tracker.memories_since_consolidation, 1);

        run_prompt_with_model_and_registry(
            &mut connection,
            "Second message",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("second prompt should succeed");

        let memories =
            elroy_db::list_active_memories(&connection, 10).expect("memories should list");
        assert_eq!(memories.len(), 1);
        assert!(memories[0].body.contains("A manual memory"));

        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 2);
        assert_eq!(tracker.memories_since_consolidation, 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn refresh_context_if_needed_compresses_transcript_and_creates_memory() {
        let unique = format!(
            "elroy-rs-app-context-refresh-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let mut transcript = vec![ConversationMessage::new(MessageRole::System, "system")];
        for index in 0..8 {
            transcript.push(ConversationMessage::new(
                MessageRole::User,
                format!("user {index} words repeated repeated repeated repeated"),
            ));
            transcript.push(ConversationMessage::new(
                MessageRole::Assistant,
                format!("assistant {index} words repeated repeated repeated repeated"),
            ));
        }
        let original_len = transcript.len();
        elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
            .expect("messages should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.max_tokens = 40;

        let refreshed = refresh_context_if_needed(
            &mut connection,
            &config,
            &BootstrapPlan::from_config(&config),
        )
        .expect("context refresh should succeed");

        assert!(refreshed);

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored[0].role, MessageRole::System);
        assert!(stored.len() < original_len);
        assert_eq!(
            stored[stored.len() - 2]
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.first())
                .map(|call| call.name.as_str()),
            Some("context_summary")
        );
        assert_eq!(stored[stored.len() - 1].role, MessageRole::Tool);
        let summary_content = stored[stored.len() - 1]
            .content
            .as_deref()
            .expect("summary tool content should exist");
        assert!(summary_content.starts_with("Recent conversation summary:"));
        assert!(summary_content.contains("user 7 words repeated"));
        assert!(summary_content.contains("assistant 7 words repeated"));

        let memories = elroy_db::list_active_memories(&connection, 10).expect("memories load");
        assert_eq!(memories.len(), 1);
        assert!(memories[0].body.contains("user"));

        let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
            .expect("tracker should load")
            .expect("tracker should exist");
        assert_eq!(tracker.messages_since_memory, 0);
        assert_eq!(tracker.memories_since_consolidation, 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn format_context_summary_message_creates_bounded_summary_text() {
        let now = Utc::now().timestamp();
        let summary = format_context_summary_message(&[
            ConversationMessage::new(MessageRole::System, "ignore"),
            ConversationMessage {
                id: None,
                role: MessageRole::User,
                content: Some("A very long user message that should appear".to_string()),
                created_at_unix: now - 60,
                tool_calls: None,
                tool_call_id: None,
                chat_model: None,
            },
            ConversationMessage {
                id: None,
                role: MessageRole::Assistant,
                content: Some("A very long assistant message that should also appear".to_string()),
                created_at_unix: now - 30,
                tool_calls: None,
                tool_call_id: None,
                chat_model: None,
            },
        ]);

        assert!(summary.starts_with("Recent conversation summary:"));
        assert!(summary.contains("User ("));
        assert!(summary.contains("): A very long user message"));
        assert!(summary.contains("Assistant ("));
        assert!(summary.contains("): A very long assistant message"));
        assert!(summary.contains("Messages from "));
        assert!(!summary.contains("ignore"));
    }

    #[test]
    fn format_context_summary_message_includes_tool_interactions() {
        let summary = format_context_summary_message(&[
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "search_memories".to_string(),
                    arguments_json: "{\"query\":\"project update\"}".to_string(),
                }],
            ),
            ConversationMessage::tool_result("call-1", "Found the project update memory."),
        ]);

        assert!(summary.starts_with("Recent conversation summary:"));
        assert!(summary.contains("Assistant Tool Call ("));
        assert!(summary.contains("): search_memories"));
        assert!(summary.contains("project update"));
        assert!(summary.contains("Tool Result ("));
        assert!(summary.contains("): Found the project update memory."));
    }

    #[test]
    fn format_context_messages_for_summary_uses_named_roles_and_tools() {
        let formatted = format_context_messages_for_summary(
            &[
                ConversationMessage::new(MessageRole::System, "system"),
                ConversationMessage::new(MessageRole::User, "I need to finish payroll"),
                ConversationMessage::assistant_with_tool_calls(
                    "I should look that up.",
                    vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "search_memories".to_string(),
                        arguments_json: "{\"query\":\"payroll\"}".to_string(),
                    }],
                ),
                ConversationMessage::tool_result("call-1", "Found payroll reminder."),
            ],
            "User",
            "Elroy",
        );

        assert!(formatted.contains("User: I need to finish payroll"));
        assert!(formatted.contains("Elroy: I should look that up."));
        assert!(formatted.contains("Elroy Tool Call: search_memories"));
        assert!(formatted.contains("Tool Result: Found payroll reminder."));
        assert!(!formatted.contains("system"));
    }

    #[test]
    fn summarize_context_messages_with_model_returns_prefixed_summary() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "I reminded the user about payroll and the tone stayed focused.".to_string(),
        }]]);

        let summary = summarize_context_messages_with_model(
            &model,
            "Elroy",
            "User",
            &[
                ConversationMessage::new(MessageRole::User, "I need to finish payroll"),
                ConversationMessage::new(
                    MessageRole::Assistant,
                    "You mentioned a payroll deadline.",
                ),
            ],
        )
        .expect("summary should be generated");

        assert_eq!(
            summary,
            "Recent conversation summary: I reminded the user about payroll and the tone stayed focused."
        );
    }

    #[test]
    fn refresh_context_if_needed_skips_when_transcript_is_under_threshold() {
        let unique = format!(
            "elroy-rs-app-context-refresh-skip-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::System, "system"),
                ConversationMessage::new(MessageRole::User, "hello"),
                ConversationMessage::new(MessageRole::Assistant, "hi"),
            ],
        )
        .expect("messages should persist");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        config.max_tokens = 1_000;

        let refreshed = refresh_context_if_needed(
            &mut connection,
            &config,
            &BootstrapPlan::from_config(&config),
        )
        .expect("context refresh should succeed");

        assert!(!refreshed);
        assert!(
            elroy_db::list_active_memories(&connection, 10)
                .expect("memories load")
                .is_empty()
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_rejects_unknown_force_tool() {
        let unique = format!(
            "elroy-rs-app-force-tool-{}",
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

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let model = FakeModel::new(vec![]);
        let config = AppConfig::defaults();
        let error = run_prompt_with_model_and_registry(
            &mut connection,
            "Hello",
            &model,
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: Some("missing_tool"),
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect_err("missing force tool should fail");

        assert!(
            error
                .to_string()
                .contains("Requested tool missing_tool not available")
        );
        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn prompt_event_stream_finalizes_snapshot_after_drain() {
        let unique = format!(
            "elroy-rs-app-stream-finalize-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let database_path = home.join("elroy.db");
        fs::create_dir_all(home.join("memories")).expect("memory dir should be created");
        fs::create_dir_all(home.join("agenda")).expect("agenda dir should be created");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "streamed hello".to_string(),
        }]]);
        let config = AppConfig::defaults();
        let mut stream = run_prompt_with_model_and_registry_stream(
            connection,
            home.clone(),
            "hello",
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
            Box::new(model),
            ExecutableToolRegistry::new(vec![]),
        )
        .expect("stream should start");

        assert!(matches!(
            stream.next(),
            Some(StreamEvent::StatusUpdate { content }) if content == "loading context..."
        ));
        assert!(matches!(
            stream.next(),
            Some(StreamEvent::StatusUpdate { content }) if content == "thinking..."
        ));
        assert!(matches!(
            stream.next(),
            Some(StreamEvent::AssistantResponse { content }) if content == "streamed hello"
        ));
        assert!(stream.snapshot().is_none());
        assert!(stream.next().is_none());
        assert_eq!(
            stream
                .snapshot()
                .and_then(|snapshot| snapshot.status.as_deref()),
            Some("loaded persisted transcript and sidebar data")
        );

        let snapshot = stream.into_snapshot().expect("snapshot should finalize");
        assert_eq!(
            snapshot.status.as_deref(),
            Some("loaded persisted transcript and sidebar data")
        );
        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn should_offer_greeting_requires_recent_user_message_to_be_old_enough() {
        let now = Utc::now().timestamp();
        let recent = vec![ConversationMessage {
            role: MessageRole::User,
            content: Some("hello".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: now - 60,
            tool_calls: None,
            tool_call_id: None,
        }];
        let stale = vec![ConversationMessage {
            role: MessageRole::User,
            content: Some("hello".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: now - 600,
            tool_calls: None,
            tool_call_id: None,
        }];

        assert!(!should_offer_greeting(&[], 5.0));
        assert!(!should_offer_greeting(&recent, 5.0));
        assert!(should_offer_greeting(&stale, 5.0));
    }

    #[test]
    fn drop_old_context_messages_preserves_first_non_system_message() {
        let unique = format!(
            "elroy-rs-app-prune-context-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(&home).expect("home dir should be created");
        let database_path = home.join("elroy.db");
        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let stale_first_user = ConversationMessage {
            role: MessageRole::User,
            content: Some("old opener".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: Utc::now().timestamp() - 100_000,
            tool_calls: None,
            tool_call_id: None,
        };
        let stale_assistant = ConversationMessage {
            role: MessageRole::Assistant,
            content: Some("old reply".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: Utc::now().timestamp() - 100_000,
            tool_calls: None,
            tool_call_id: None,
        };
        let recent_user = ConversationMessage::new(MessageRole::User, "recent");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[stale_first_user, stale_assistant, recent_user.clone()],
        )
        .expect("messages should persist");

        drop_old_context_messages(&mut connection, 60.0).expect("prune should succeed");

        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].content.as_deref(), Some("old opener"));
        assert_eq!(stored[1].content.as_deref(), recent_user.content.as_deref());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn startup_prompt_stream_can_run_restart_prompt_without_persisting_input() {
        let unique = format!(
            "elroy-rs-app-startup-restart-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(home.join("memories")).expect("memory dir should be created");
        fs::create_dir_all(home.join("agenda")).expect("agenda dir should be created");

        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "event: response.output_text.delta\n",
                "data: {\"delta\":\"Restarted successfully. Ready to continue.\"}\n\n",
                "data: [DONE]\n\n"
            ))
            .create();

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.config_path = home.join("elroy.conf.yaml");
        config.memory_dir = home.join("memories");
        config.agenda_dir = home.join("agenda");
        config.database_path = home.join("elroy.db");
        config.openai_api_key = Some("test-key".to_string());
        config.openai_base_url = format!("{}/responses", server.url());
        config.messages_between_memory = 1;

        let runtime = AppRuntime::new(config.clone());
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
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
        drop(connection);

        let mut stream = runtime
            .startup_prompt_stream(Some("Restarted successfully. Ready to continue."))
            .expect("startup prompt should start")
            .expect("restart stream should be present");
        while stream.next().is_some() {}
        let completion = stream
            .into_completion()
            .expect("completion should finalize");
        let deferred = completion
            .deferred_auto_memory
            .expect("restart stream should defer auto memory");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert!(!stored.iter().any(|message| {
            message.role == MessageRole::User
                && message.content.as_deref() == Some("Restarted successfully. Ready to continue.")
        }));
        assert!(stored.iter().any(|message| {
            message.role == MessageRole::Assistant
                && message.content.as_deref() == Some("Restarted successfully. Ready to continue.")
        }));
        let before_memories =
            crate::list_active_memories_in_scope(&connection, &config.memory_dir, 10)
                .expect("active memories should load");
        assert!(before_memories.is_empty());
        drop(connection);

        runtime
            .run_auto_memory_for_transcript(
                deferred.existing_transcript_len,
                deferred.transcript.clone(),
            )
            .expect("deferred auto memory should succeed");

        let reopened =
            open_sqlite_connection(&config.database_path).expect("database should reopen again");
        let active_memories =
            crate::list_active_memories_in_scope(&reopened, &config.memory_dir, 10)
                .expect("active memories should load");
        assert_eq!(active_memories.len(), 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn startup_prompt_stream_can_offer_greeting_when_user_message_is_old() {
        let unique = format!(
            "elroy-rs-app-startup-greeting-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(home.join("memories")).expect("memory dir should be created");
        fs::create_dir_all(home.join("agenda")).expect("agenda dir should be created");

        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "event: response.output_text.delta\n",
                "data: {\"delta\":\"Good to see you again.\"}\n\n",
                "data: [DONE]\n\n"
            ))
            .create();

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.config_path = home.join("elroy.conf.yaml");
        config.memory_dir = home.join("memories");
        config.agenda_dir = home.join("agenda");
        config.database_path = home.join("elroy.db");
        config.openai_api_key = Some("test-key".to_string());
        config.openai_base_url = format!("{}/responses", server.url());
        config.enable_assistant_greeting = true;
        config.min_convo_age_for_greeting_minutes = 5.0;
        config.messages_between_memory = 1;

        let runtime = AppRuntime::new(config.clone());
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        let stale_user = ConversationMessage {
            role: MessageRole::User,
            content: Some("hello".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: Utc::now().timestamp() - 600,
            tool_calls: None,
            tool_call_id: None,
        };
        elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &[stale_user])
            .expect("messages should persist");
        drop(connection);

        let mut stream = runtime
            .startup_prompt_stream(None)
            .expect("startup prompt should evaluate")
            .expect("greeting stream should be present");
        while stream.next().is_some() {}
        let completion = stream
            .into_completion()
            .expect("completion should finalize");
        let deferred = completion
            .deferred_auto_memory
            .expect("greeting stream should defer auto memory");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert!(stored.iter().any(|message| {
            message.role == MessageRole::Assistant
                && message.content.as_deref() == Some("Good to see you again.")
        }));
        let before_memories =
            crate::list_active_memories_in_scope(&connection, &config.memory_dir, 10)
                .expect("active memories should load");
        assert!(before_memories.is_empty());
        drop(connection);

        runtime
            .run_auto_memory_for_transcript(
                deferred.existing_transcript_len,
                deferred.transcript.clone(),
            )
            .expect("deferred auto memory should succeed");

        let reopened =
            open_sqlite_connection(&config.database_path).expect("database should reopen again");
        let active_memories =
            crate::list_active_memories_in_scope(&reopened, &config.memory_dir, 10)
                .expect("active memories should load");
        assert_eq!(active_memories.len(), 1);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn load_snapshot_drops_old_context_messages_on_runtime_open() {
        let unique = format!(
            "elroy-rs-app-drop-old-context-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(home.join("memories")).expect("memory dir should be created");
        fs::create_dir_all(home.join("agenda")).expect("agenda dir should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.config_path = home.join("elroy.conf.yaml");
        config.memory_dir = home.join("memories");
        config.agenda_dir = home.join("agenda");
        config.database_path = home.join("elroy.db");
        config.max_context_age_minutes = 60.0;

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        let stale_user = ConversationMessage {
            role: MessageRole::User,
            content: Some("stale user".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: Utc::now().timestamp() - 100_000,
            tool_calls: None,
            tool_call_id: None,
        };
        let recent_assistant = ConversationMessage::new(MessageRole::Assistant, "recent answer");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[stale_user, recent_assistant.clone()],
        )
        .expect("messages should persist");
        drop(connection);

        let runtime = AppRuntime::new(config.clone());
        let snapshot = runtime.load_snapshot().expect("snapshot should load");
        assert!(
            snapshot
                .conversation_lines
                .iter()
                .any(|line| line == "user: stale user")
        );
        assert!(
            snapshot
                .conversation_lines
                .iter()
                .any(|line| line == "assistant: recent answer")
        );

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].content.as_deref(), Some("stale user"));
        assert_eq!(
            stored[1].content.as_deref(),
            recent_assistant.content.as_deref()
        );

        let stale_follow_up = ConversationMessage {
            role: MessageRole::Assistant,
            content: Some("stale follow-up".to_string()),
            chat_model: None,
            id: None,
            created_at_unix: Utc::now().timestamp() - 100_000,
            tool_calls: None,
            tool_call_id: None,
        };
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[stored[0].clone(), stored[1].clone(), stale_follow_up],
        )
        .expect("messages should persist");
        drop(connection);

        let _ = runtime.load_snapshot().expect("snapshot should reload");
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].content.as_deref(), Some("stale user"));
        assert_eq!(stored[1].content.as_deref(), Some("recent answer"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn trivial_messages_skip_memory_recall() {
        assert!(should_skip_memory_recall("hi"));
        assert!(should_skip_memory_recall("thanks"));
        assert!(!should_skip_memory_recall("I am going running tomorrow"));
    }

    #[test]
    fn trivial_messages_skip_memory_recall_case_insensitively() {
        assert!(should_skip_memory_recall("hello"));
        assert!(should_skip_memory_recall("HELLO"));
        assert!(should_skip_memory_recall("HeLLo"));
        assert!(should_skip_memory_recall("OK"));
        assert!(should_skip_memory_recall("ThAnKs"));
    }

    #[test]
    fn trivial_messages_skip_memory_recall_with_punctuation() {
        assert!(should_skip_memory_recall("hello!"));
        assert!(should_skip_memory_recall("Thanks!!!"));
        assert!(should_skip_memory_recall("ok?"));
        assert!(should_skip_memory_recall("good morning,"));
        assert!(!should_skip_memory_recall("what about bob?"));
    }

    #[test]
    fn clarification_only_messages_skip_memory_recall() {
        assert!(should_skip_memory_recall("what?"));
        assert!(should_skip_memory_recall("huh"));
        assert!(should_skip_memory_recall("pardon?"));
        assert!(should_skip_memory_recall("sorry!"));
        assert!(should_skip_memory_recall("excuse me."));
        assert!(!should_skip_memory_recall(
            "what about the payroll follow-up?"
        ));
    }

    #[test]
    fn parse_memory_recall_decision_accepts_json_and_fenced_json() {
        assert_eq!(
            parse_memory_recall_decision(r#"{"needs_recall":true,"reasoning":"topic mentioned"}"#),
            Some((true, "topic mentioned".to_string()))
        );
        assert_eq!(
            parse_memory_recall_decision(
                "```json\n{\"needs_recall\":false,\"reasoning\":\"pure greeting\"}\n```"
            ),
            Some((false, "pure greeting".to_string()))
        );
    }

    #[test]
    fn parse_relevance_filter_response_accepts_json_and_fenced_json() {
        assert_eq!(
            parse_relevance_filter_response(r#"{"answers":[true,false],"reasoning":"only first"}"#),
            Some(vec![true, false])
        );
        assert_eq!(
            parse_relevance_filter_response(
                "```json\n{\"answers\":[false,true],\"reasoning\":\"only second\"}\n```"
            ),
            Some(vec![false, true])
        );
    }

    #[test]
    fn parse_reflective_recall_model_response_accepts_json_and_fenced_json() {
        assert_eq!(
            parse_reflective_recall_model_response(
                r#"{"is_relevant":true,"content":"I should remind the user about payroll."}"#
            ),
            Some((
                true,
                Some("I should remind the user about payroll.".to_string())
            ))
        );
        assert_eq!(
            parse_reflective_recall_model_response(
                "```json\n{\"is_relevant\":false,\"content\":null}\n```"
            ),
            Some((false, None))
        );
    }

    #[test]
    fn classify_memory_recall_with_model_parses_structured_response() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: r#"{"needs_recall":true,"reasoning":"The message references prior context."}"#
                .to_string(),
        }]]);
        let decision = classify_memory_recall_with_model(
            &model,
            "What was that library you mentioned?",
            &[
                ConversationMessage::new(MessageRole::User, "I'm working on a Python project"),
                ConversationMessage::new(
                    MessageRole::Assistant,
                    "You should look at the requests library.",
                ),
            ],
            3,
        )
        .expect("classifier should parse JSON");

        assert!(decision.needs_recall);
        assert!(decision.used_llm);
        assert_eq!(decision.reasoning, "The message references prior context.");
    }

    #[test]
    fn determine_memory_recall_decision_falls_back_to_conservative_recall() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "not json".to_string(),
        }]]);
        let decision = determine_memory_recall_decision(
            true,
            3,
            "What was that library you mentioned?",
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "You should look at the requests library.",
            )],
            Some(&model),
        );

        assert!(decision.needs_recall);
        assert!(!decision.used_llm);
        assert!(decision.reasoning.contains("classifier unavailable"));
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_skip_recall_via_classifier_model() {
        struct NoRecallPromptModel;

        impl ModelClient for NoRecallPromptModel {
            fn next_events(
                &self,
                request: ConversationRequest<'_>,
            ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
                assert_eq!(request.user_message, "What was that library you mentioned?");
                assert!(!request.transcript.iter().any(|message| {
                    message.role == MessageRole::Tool
                        && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                }));
                Ok(vec![StreamEvent::AssistantResponse {
                    content: "No recall was injected.".to_string(),
                }])
            }
        }

        let unique = format!(
            "elroy-rs-app-recall-classifier-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        fs::write(
            memory_dir.join("python_library.md"),
            "# Python Library\n\nYou mentioned the requests library for Python projects.\n",
        )
        .expect("memory file should be written");
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");
        elroy_db::replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::Assistant,
                "You should look at the requests library.",
            )],
        )
        .expect("messages should persist");

        let classifier = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content:
                r#"{"needs_recall":false,"reasoning":"This is only a lightweight follow-up."}"#
                    .to_string(),
        }]]);
        let events = run_prompt_with_model_and_registry_internal(
            &mut connection,
            "What was that library you mentioned?",
            &NoRecallPromptModel,
            Some(&classifier),
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: true,
                memory_recall_classifier_window: 3,
                reflect: false,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::StatusUpdate { content } if content == "classifying recall..."
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::StatusUpdate { content } if content == "fetching memories..."
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_filter_overlap_matches_via_relevance_model() {
        struct NoRecallContextPromptModel;

        impl ModelClient for NoRecallContextPromptModel {
            fn next_events(
                &self,
                request: ConversationRequest<'_>,
            ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
                assert_eq!(request.user_message, "What workout gear should I bring?");
                assert!(!request.transcript.iter().any(|message| {
                    message.role == MessageRole::Tool
                        && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                }));
                Ok(vec![StreamEvent::AssistantResponse {
                    content: "No recall was injected.".to_string(),
                }])
            }
        }

        let unique = format!(
            "elroy-rs-app-recall-relevance-filter-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        fs::write(
            memory_dir.join("gym_note.md"),
            "# Gym Note\n\nBring dumbbells to the gym workout.\n",
        )
        .expect("memory file should be written");
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let classifier_and_filter = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"needs_recall":true,"reasoning":"The prompt references workout gear."}"#
                        .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[false],"reasoning":"This memory is not about gear the user should bring."}"#.to_string(),
            }],
        ]);
        let events = run_prompt_with_model_and_registry_internal(
            &mut connection,
            "What workout gear should I bring?",
            &NoRecallContextPromptModel,
            Some(&classifier_and_filter),
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: true,
                memory_recall_classifier_window: 3,
                reflect: false,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::StatusUpdate { content } if content == "classifying recall..."
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::StatusUpdate { content } if content == "fetching memories..."
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_can_broaden_recall_beyond_overlap_via_relevance_model() {
        struct RecallContextPromptModel;

        impl ModelClient for RecallContextPromptModel {
            fn next_events(
                &self,
                request: ConversationRequest<'_>,
            ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
                assert_eq!(
                    request.user_message,
                    "What gear should I bring to practice?"
                );
                let recall_payload = request
                    .transcript
                    .iter()
                    .find(|message| {
                        message.role == MessageRole::Tool
                            && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                    })
                    .and_then(|message| message.content.as_deref())
                    .expect("fast recall payload should be injected");
                assert!(recall_payload.contains("resistance bands"));
                Ok(vec![StreamEvent::AssistantResponse {
                    content: "Bring the resistance bands.".to_string(),
                }])
            }
        }

        let unique = format!(
            "elroy-rs-app-recall-relevance-expansion-{}",
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
        config.home_dir = home.clone();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        fs::write(
            memory_dir.join("practice_gear.md"),
            "# Practice Gear\n\nPack resistance bands before training.\n",
        )
        .expect("memory file should be written");
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let classifier_and_filter = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"needs_recall":true,"reasoning":"The user is asking what equipment to bring."}"#
                        .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"This candidate is relevant even though the wording differs."}"#.to_string(),
            }],
        ]);
        let events = run_prompt_with_model_and_registry_internal(
            &mut connection,
            "What gear should I bring to practice?",
            &RecallContextPromptModel,
            Some(&classifier_and_filter),
            ExecutableToolRegistry::new(vec![]),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: true,
                memory_recall_classifier_window: 3,
                reflect: false,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::StatusUpdate { content } if content == "fetching memories..."
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn run_prompt_with_model_and_registry_fast_recall_can_include_due_and_agenda_items() {
        struct MixedFastRecallPromptModel;

        impl ModelClient for MixedFastRecallPromptModel {
            fn next_events(
                &self,
                request: ConversationRequest<'_>,
            ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
                assert_eq!(
                    request.user_message,
                    "What should I remember for basketball practice?"
                );
                let recall_payload = request
                    .transcript
                    .iter()
                    .find(|message| {
                        message.role == MessageRole::Tool
                            && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                    })
                    .and_then(|message| message.content.as_deref())
                    .expect("fast recall payload should be injected");
                assert!(recall_payload.contains("basketball form"));
                assert!(recall_payload.contains("practice reminder"));
                assert!(recall_payload.contains("drill plan"));
                assert!(recall_payload.contains("\"memory_type\": \"Memory\""));
                assert!(recall_payload.contains("\"memory_type\": \"AgendaItem\""));
                Ok(vec![StreamEvent::AssistantResponse {
                    content: "Injected mixed fast recall.".to_string(),
                }])
            }
        }

        let unique = format!(
            "elroy-rs-app-fast-recall-mixed-items-{}",
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
            memory_dir.join("basketball_form.md"),
            "# Basketball Form\n\nRemember to follow through on your shot.\n",
        )
        .expect("memory file should be written");
        fs::write(
            agenda_dir.join("practice_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nBring the resistance bands\n",
        )
        .expect("due item file should be written");
        fs::write(
            agenda_dir.join("drill_plan.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\n---\n\nFocus on basketball practice footwork and follow-through\n",
        )
        .expect("agenda item file should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        config.memory_recall_classifier_enabled = false;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");

        let events = run_prompt_with_model_and_registry(
            &mut connection,
            "What should I remember for basketball practice?",
            &MixedFastRecallPromptModel,
            build_live_tool_registry(&config),
            PromptExecutionOptions {
                role: MessageRole::User,
                persist_input_message: true,
                force_tool: None,
                assistant_name: &config.assistant_name,
                ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
                home_dir: &home,
                bootstrap_plan: BootstrapPlan::from_config(&config),
                messages_between_memory: config.messages_between_memory,
                memories_between_consolidation: config.memories_between_consolidation,
                messages_between_self_reflection: config.messages_between_self_reflection,
                defer_auto_memory: false,
                defer_self_reflection: false,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
                reflect: config.reflect,
            },
        )
        .expect("prompt should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content } if content == "Injected mixed fast recall."
        )));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn context_refresh_is_not_needed_without_user_messages() {
        let context_messages = vec![
            ConversationMessage::new(MessageRole::System, "system"),
            ConversationMessage::new(MessageRole::Assistant, "assistant only"),
        ];

        assert!(!is_context_refresh_needed(&context_messages, 1));
    }

    #[test]
    fn context_refresh_is_needed_when_token_budget_is_exceeded() {
        let context_messages = vec![
            ConversationMessage::new(MessageRole::System, "system"),
            ConversationMessage::new(MessageRole::User, "one two three four five six"),
            ConversationMessage::new(MessageRole::Assistant, "seven eight nine ten eleven"),
        ];

        assert!(count_context_tokens(&context_messages) > 5);
        assert!(is_context_refresh_needed(&context_messages, 5));
    }

    #[test]
    fn compress_context_messages_preserves_system_and_relative_order() {
        let mut context_messages = vec![ConversationMessage::new(MessageRole::System, "system")];
        for index in 0..12 {
            context_messages.push(ConversationMessage::new(
                MessageRole::User,
                format!("{index} user words repeated repeated repeated"),
            ));
            context_messages.push(ConversationMessage::new(
                MessageRole::Assistant,
                format!("{index} assistant words repeated repeated repeated"),
            ));
        }

        let compressed = compress_context_messages(&context_messages, 30, 10_000.0);

        assert_eq!(compressed[0].role, MessageRole::System);
        assert_eq!(compressed[0].content.as_deref(), Some("system"));
        assert!(compressed.len() < context_messages.len());
        for pair in compressed[1..].windows(2) {
            let left = pair[0]
                .content
                .as_deref()
                .and_then(|content| content.split_whitespace().next())
                .and_then(|token| token.parse::<usize>().ok());
            let right = pair[1]
                .content
                .as_deref()
                .and_then(|content| content.split_whitespace().next())
                .and_then(|token| token.parse::<usize>().ok());
            if let (Some(left), Some(right)) = (left, right) {
                assert!(left <= right);
            }
        }
    }

    #[test]
    fn compress_context_messages_keeps_assistant_tool_result_pair_together() {
        let tool_call = ToolCall {
            id: "call-1".to_string(),
            name: "get_weather".to_string(),
            arguments_json: "{\"location\":\"Paris\"}".to_string(),
        };
        let context_messages = vec![
            ConversationMessage::new(MessageRole::System, "system"),
            ConversationMessage::new(MessageRole::User, "older context words words words"),
            ConversationMessage::assistant_with_tool_calls("", vec![tool_call.clone()]),
            ConversationMessage::tool_result(&tool_call.id, "{\"temp\":25}"),
        ];

        let compressed = compress_context_messages(&context_messages, 7, 10_000.0);

        assert_eq!(compressed.len(), 3);
        assert_eq!(compressed[0].role, MessageRole::System);
        assert_eq!(compressed[1].role, MessageRole::Assistant);
        assert_eq!(compressed[2].role, MessageRole::Tool);
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
        let config = AppConfig::defaults();
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "practice reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Bring the resistance bands".to_string(),
            trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
            trigger_context: Some("before basketball practice".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];
        let agenda_items = vec![AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "drill plan".to_string(),
            file_path: "/tmp/drill_plan.md".to_string(),
            agenda_date: Some("2026-05-20".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Focus on basketball practice footwork and follow-through".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        }];
        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            RecallContext {
                transcript: &[],
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &due_items,
                agenda_items: &agenda_items,
            },
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(
            messages[0]
                .tool_calls
                .as_ref()
                .map(|calls| calls[0].name.as_str()),
            Some("get_fast_recall")
        );
        assert_eq!(messages[1].role, MessageRole::Tool);
        let payload = messages[1]
            .content
            .as_deref()
            .expect("tool payload should exist");
        assert!(payload.contains("basketball form"));
        assert!(payload.contains("practice reminder"));
        assert!(payload.contains("drill plan"));
        assert!(payload.contains("\"recall_metadata\""));
        assert!(payload.contains("\"memory_type\": \"Memory\""));
        assert!(payload.contains("\"memory_type\": \"AgendaItem\""));
    }

    #[test]
    fn recall_memory_context_messages_use_reflective_recall_when_enabled() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        config.reflect = true;
        let transcript = vec![
            ConversationMessage::new(MessageRole::User, "I am getting ready for practice"),
            ConversationMessage::new(MessageRole::Assistant, "What part do you want to focus on?"),
        ];
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "practice reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Bring the resistance bands".to_string(),
            trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
            trigger_context: Some("before basketball practice".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];
        let agenda_items = vec![AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "drill plan".to_string(),
            file_path: "/tmp/drill_plan.md".to_string(),
            agenda_date: Some("2026-05-20".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Focus on basketball practice footwork and follow-through".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        }];
        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            RecallContext {
                transcript: &transcript,
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &due_items,
                agenda_items: &agenda_items,
            },
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0]
                .tool_calls
                .as_ref()
                .map(|calls| calls[0].name.as_str()),
            Some("get_reflective_recall")
        );
        let payload = messages[1]
            .content
            .as_deref()
            .expect("reflective recall payload should exist");
        assert!(payload.contains("I remember these memory details may be relevant"));
        assert!(payload.contains("\"recall_metadata\""));
        assert!(payload.contains("basketball form"));
        assert!(payload.contains("I also recall these due items may matter"));
        assert!(payload.contains("practice reminder"));
        assert!(payload.contains("I also recall these agenda items may matter"));
        assert!(payload.contains("drill plan"));
        assert!(payload.contains("Recent conversation context"));
        assert!(payload.contains("user: I am getting ready for practice"));
        assert!(payload.contains("assistant: What part do you want to focus on?"));
        assert!(
            payload.contains("The latest user message is: I am heading to basketball practice")
        );
    }

    #[test]
    fn reflective_recall_uses_model_authored_content_when_available() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        config.reflect = true;
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: r#"{"is_relevant":true,"content":"I remember that the user should bring the resistance bands to practice."}"#.to_string(),
        }]]);
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "practice reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Bring the resistance bands".to_string(),
            trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
            trigger_context: Some("before basketball practice".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];

        let messages = recall_memory_context_messages_with_decision(
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            true,
            Some(&model),
            RecallContext {
                transcript: &[],
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &due_items,
                agenda_items: &[],
            },
        );

        let payload = messages[1]
            .content
            .as_deref()
            .expect("reflective recall payload should exist");
        assert!(
            payload.contains(
                "I remember that the user should bring the resistance bands to practice."
            )
        );
        assert!(!payload.contains("I remember these memory details may be relevant"));
    }

    #[test]
    fn reflective_recall_can_be_suppressed_when_model_marks_it_irrelevant() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        config.reflect = true;
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: r#"{"is_relevant":false,"content":null}"#.to_string(),
        }]]);

        let messages = recall_memory_context_messages_with_decision(
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            true,
            Some(&model),
            RecallContext {
                transcript: &[],
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &[],
                agenda_items: &[],
            },
        );

        assert!(messages.is_empty());
    }

    #[test]
    fn reflective_recall_skips_already_recalled_due_items_and_agenda_items() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        config.reflect = true;
        let transcript = vec![ConversationMessage::tool_result(
            "bootstrap-memory-recall",
            r#"{"content":"Earlier reflective recall.","recall_metadata":[{"memory_type":"AgendaItem","memory_id":2,"name":"Practice Reminder"},{"memory_type":"AgendaItem","memory_id":3,"name":"Drill Plan"}]}"#,
        )];
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "practice reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Bring the resistance bands".to_string(),
            trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
            trigger_context: Some("before basketball practice".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];
        let agenda_items = vec![AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "drill plan".to_string(),
            file_path: "/tmp/drill_plan.md".to_string(),
            agenda_date: Some("2026-05-20".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Focus on basketball practice footwork and follow-through".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        }];

        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            RecallContext {
                transcript: &transcript,
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &due_items,
                agenda_items: &agenda_items,
            },
        );

        let payload = messages[1]
            .content
            .as_deref()
            .expect("reflective recall payload should exist");
        assert!(payload.contains("basketball form"));
        assert!(!payload.contains("practice reminder"));
        assert!(!payload.contains("drill plan"));
    }

    #[test]
    fn recall_memory_context_messages_limits_to_two_memories() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "basketball shooting drills",
            RecallContext {
                transcript: &[],
                memories: &[
                    MemoryRecord {
                        id: 1,
                        legacy_frontmatter_id: None,
                        name: "basketball form".to_string(),
                        file_path: "/tmp/basketball.md".to_string(),
                        body: "Focus on shooting form during basketball drills".to_string(),
                        is_active: true,
                        updated_at_unix: 30,
                    },
                    MemoryRecord {
                        id: 2,
                        legacy_frontmatter_id: None,
                        name: "basketball warmup".to_string(),
                        file_path: "/tmp/warmup.md".to_string(),
                        body: "Warm up shoulders before basketball shooting".to_string(),
                        is_active: true,
                        updated_at_unix: 20,
                    },
                    MemoryRecord {
                        id: 3,
                        legacy_frontmatter_id: None,
                        name: "basketball recovery".to_string(),
                        file_path: "/tmp/recovery.md".to_string(),
                        body: "Stretch after basketball practice and shooting".to_string(),
                        is_active: true,
                        updated_at_unix: 10,
                    },
                ],
                due_items: &[],
                agenda_items: &[],
            },
        );

        assert_eq!(messages.len(), 2);
        let tool_payload = messages[1]
            .content
            .as_deref()
            .expect("tool payload should exist");
        assert!(tool_payload.contains("basketball form"));
        assert!(tool_payload.contains("basketball warmup"));
        assert!(!tool_payload.contains("basketball recovery"));
    }

    #[test]
    fn recall_memory_context_messages_can_bypass_classifier_when_disabled() {
        let mut config = AppConfig::defaults();
        config.memory_recall_classifier_enabled = false;
        let transcript = vec![ConversationMessage::new(
            MessageRole::User,
            "I am training for basketball",
        )];

        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "hi",
            RecallContext {
                transcript: &transcript,
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "practice plan".to_string(),
                    file_path: "/tmp/practice.md".to_string(),
                    body: "Warm up before basketball drills".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &[],
                agenda_items: &[],
            },
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[1].role, MessageRole::Tool);
    }

    #[test]
    fn recall_memory_context_messages_respect_configured_window() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::User, "I am training for basketball"),
            ConversationMessage::new(MessageRole::Assistant, "How is practice going?"),
            ConversationMessage::new(MessageRole::User, "My sleep schedule is rough"),
        ];
        let memories = vec![MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "practice plan".to_string(),
            file_path: "/tmp/practice.md".to_string(),
            body: "Warm up before basketball drills".to_string(),
            is_active: true,
            updated_at_unix: 10,
        }];

        let mut narrow = AppConfig::defaults();
        narrow.memory_recall_classifier_window = 1;
        let narrow_messages = recall_memory_context_messages(
            narrow.memory_recall_classifier_enabled,
            narrow.memory_recall_classifier_window,
            narrow.reflect,
            "What should I focus on?",
            RecallContext {
                transcript: &transcript,
                memories: &memories,
                due_items: &[],
                agenda_items: &[],
            },
        );

        let mut wide = AppConfig::defaults();
        wide.memory_recall_classifier_window = 3;
        let wide_messages = recall_memory_context_messages(
            wide.memory_recall_classifier_enabled,
            wide.memory_recall_classifier_window,
            wide.reflect,
            "What should I focus on?",
            RecallContext {
                transcript: &transcript,
                memories: &memories,
                due_items: &[],
                agenda_items: &[],
            },
        );

        assert!(narrow_messages.is_empty());
        assert_eq!(wide_messages.len(), 2);
    }

    #[test]
    fn memory_recall_status_updates_classify_before_fetch_when_enabled() {
        let events = memory_recall_status_updates(true, "What should I focus on?", true);

        assert_eq!(
            events,
            vec![
                StreamEvent::StatusUpdate {
                    content: "classifying recall...".to_string(),
                },
                StreamEvent::StatusUpdate {
                    content: "fetching memories...".to_string(),
                },
            ]
        );
    }

    #[test]
    fn memory_recall_status_updates_skip_classify_for_trivial_prompt() {
        let events = memory_recall_status_updates(true, "hi", false);

        assert!(events.is_empty());
    }

    #[test]
    fn memory_recall_status_updates_skip_classify_when_classifier_disabled() {
        let events = memory_recall_status_updates(false, "What should I focus on?", true);

        assert_eq!(
            events,
            vec![StreamEvent::StatusUpdate {
                content: "fetching memories...".to_string(),
            }]
        );
    }

    #[test]
    fn prompt_prelude_status_updates_wrap_recall_and_due_item_statuses() {
        let events = prompt_prelude_status_updates(true, "What should I focus on?", true, true);

        assert_eq!(
            events,
            vec![
                StreamEvent::StatusUpdate {
                    content: "loading context...".to_string(),
                },
                StreamEvent::StatusUpdate {
                    content: "classifying recall...".to_string(),
                },
                StreamEvent::StatusUpdate {
                    content: "fetching memories...".to_string(),
                },
                StreamEvent::StatusUpdate {
                    content: "surfacing due items...".to_string(),
                },
                StreamEvent::StatusUpdate {
                    content: "thinking...".to_string(),
                },
            ]
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
        assert_eq!(context[0], "user: I am training for basketball");
        assert_eq!(context[1], "assistant: How is practice going?");
        assert_eq!(context[2], "user: My jump shot is inconsistent");
    }

    #[test]
    fn build_recall_query_includes_recent_context_and_prompt() {
        let transcript = vec![
            ConversationMessage::new(MessageRole::User, "I am training for basketball"),
            ConversationMessage::new(MessageRole::Assistant, "How is practice going?"),
        ];

        let query = build_recall_query("What should I focus on?", &transcript, 4);

        assert!(query.contains("user: I am training for basketball"));
        assert!(query.contains("assistant: How is practice going?"));
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
    fn parse_reflective_recall_metadata_names_from_transcript() {
        let transcript = vec![ConversationMessage::tool_result(
            "bootstrap-memory-recall",
            r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"Memory","memory_id":2,"name":"Sleep Routine"}]}"#,
        )];

        let parsed = parse_recalled_memory_names(
            r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"Memory","memory_id":2,"name":"Sleep Routine"}]}"#,
        );
        let names = recalled_memory_names(&transcript);

        assert_eq!(parsed.len(), 2);
        assert!(names.contains("basketball form"));
        assert!(names.contains("sleep routine"));
        assert!(!names.contains("payroll followup"));
    }

    #[test]
    fn parse_reflective_recall_agenda_item_names_from_transcript() {
        let transcript = vec![ConversationMessage::tool_result(
            "bootstrap-memory-recall",
            r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"AgendaItem","memory_id":10,"name":"Drill Plan"}]}"#,
        )];

        let parsed = parse_recalled_item_names(
            r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"AgendaItem","memory_id":10,"name":"Drill Plan"}]}"#,
            "AgendaItem",
        );
        let names = recalled_item_names_by_type(&transcript, "AgendaItem");

        assert_eq!(parsed.len(), 2);
        assert!(names.contains("payroll followup"));
        assert!(names.contains("drill plan"));
        assert!(!names.contains("basketball form"));
    }

    #[test]
    fn current_context_fast_recall_messages_expose_recall_metadata() {
        let memory_messages = context_memory_tool_messages(&MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "basketball form".to_string(),
            file_path: "/tmp/basketball.md".to_string(),
            body: "Remember to follow through on your shot".to_string(),
            is_active: true,
            updated_at_unix: 10,
        });
        let due_item_messages = context_due_item_tool_messages(&AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "practice reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Bring the resistance bands".to_string(),
            trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
            trigger_context: Some("before basketball practice".to_string()),
            is_active: true,
            updated_at_unix: 11,
        });
        let task_messages = context_task_tool_messages(&AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "drill plan".to_string(),
            file_path: "/tmp/drill_plan.md".to_string(),
            agenda_date: Some("2026-05-20".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Focus on basketball practice footwork and follow-through".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        });

        let memory_payload = memory_messages[1]
            .content
            .as_deref()
            .expect("memory payload should exist");
        let due_item_payload = due_item_messages[1]
            .content
            .as_deref()
            .expect("due-item payload should exist");
        let task_payload = task_messages[1]
            .content
            .as_deref()
            .expect("task payload should exist");

        assert!(memory_payload.contains("\"recall_metadata\""));
        assert!(memory_payload.contains("\"memory_type\": \"Memory\""));
        assert!(due_item_payload.contains("\"recall_metadata\""));
        assert!(due_item_payload.contains("\"memory_type\": \"AgendaItem\""));
        assert!(task_payload.contains("\"recall_metadata\""));
        assert!(task_payload.contains("\"memory_type\": \"AgendaItem\""));
    }

    #[test]
    fn fast_recall_skips_items_already_pinned_in_current_context() {
        let config = AppConfig::defaults();
        let transcript = [
            context_memory_tool_messages(&MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "basketball form".to_string(),
                file_path: "/tmp/basketball.md".to_string(),
                body: "Remember to follow through on your shot".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }),
            context_due_item_tool_messages(&AgendaItemRecord {
                id: 2,
                legacy_frontmatter_id: None,
                name: "practice reminder".to_string(),
                file_path: "/tmp/practice_reminder.md".to_string(),
                agenda_date: Some("unscheduled".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Bring the resistance bands".to_string(),
                trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
                trigger_context: Some("before basketball practice".to_string()),
                is_active: true,
                updated_at_unix: 11,
            }),
            context_task_tool_messages(&AgendaItemRecord {
                id: 3,
                legacy_frontmatter_id: None,
                name: "drill plan".to_string(),
                file_path: "/tmp/drill_plan.md".to_string(),
                agenda_date: Some("2026-05-20".to_string()),
                is_completed: false,
                status: Some("created".to_string()),
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
                body: "Focus on basketball practice footwork and follow-through".to_string(),
                trigger_datetime: None,
                trigger_context: None,
                is_active: true,
                updated_at_unix: 12,
            }),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
            config.reflect,
            "I am heading to basketball practice",
            RecallContext {
                transcript: &transcript,
                memories: &[MemoryRecord {
                    id: 1,
                    legacy_frontmatter_id: None,
                    name: "basketball form".to_string(),
                    file_path: "/tmp/basketball.md".to_string(),
                    body: "Remember to follow through on your shot".to_string(),
                    is_active: true,
                    updated_at_unix: 10,
                }],
                due_items: &[AgendaItemRecord {
                    id: 2,
                    legacy_frontmatter_id: None,
                    name: "practice reminder".to_string(),
                    file_path: "/tmp/practice_reminder.md".to_string(),
                    agenda_date: Some("unscheduled".to_string()),
                    is_completed: false,
                    status: Some("created".to_string()),
                    closing_comment: None,
                    checklist_total: 0,
                    checklist_completed: 0,
                    body: "Bring the resistance bands".to_string(),
                    trigger_datetime: Some("2026-05-20T09:00:00".to_string()),
                    trigger_context: Some("before basketball practice".to_string()),
                    is_active: true,
                    updated_at_unix: 11,
                }],
                agenda_items: &[AgendaItemRecord {
                    id: 3,
                    legacy_frontmatter_id: None,
                    name: "drill plan".to_string(),
                    file_path: "/tmp/drill_plan.md".to_string(),
                    agenda_date: Some("2026-05-20".to_string()),
                    is_completed: false,
                    status: Some("created".to_string()),
                    closing_comment: None,
                    checklist_total: 0,
                    checklist_completed: 0,
                    body: "Focus on basketball practice footwork and follow-through".to_string(),
                    trigger_datetime: None,
                    trigger_context: None,
                    is_active: true,
                    updated_at_unix: 12,
                }],
            },
        );

        assert!(messages.is_empty());
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

    #[test]
    fn select_relevant_recall_helpers_can_filter_tool_surface_candidates() {
        let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[false],"reasoning":"Not actually relevant."}"#.to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[false],"reasoning":"Not actually relevant."}"#.to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[false],"reasoning":"Not actually relevant."}"#.to_string(),
            }],
        ]);
        let memories = vec![MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "gym note".to_string(),
            file_path: "/tmp/gym_note.md".to_string(),
            body: "Bring dumbbells to the gym workout.".to_string(),
            is_active: true,
            updated_at_unix: 10,
        }];
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "gym follow up".to_string(),
            file_path: "/tmp/gym_follow_up.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Ask about weight progression.".to_string(),
            trigger_datetime: None,
            trigger_context: Some("after the gym workout".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];
        let agenda_items = vec![AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "gym planning".to_string(),
            file_path: "/tmp/gym_planning.md".to_string(),
            agenda_date: Some("2026-05-21".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Review the workout block and equipment list.".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        }];

        let relevant_memories = select_relevant_recall_memories(
            "What workout gear should I bring?",
            &memories,
            &HashSet::new(),
            2,
            Some(&model),
        );
        let relevant_due_items = select_relevant_recall_due_items(
            "What workout gear should I bring?",
            &due_items,
            2,
            Some(&model),
        );
        let relevant_agenda_items = select_relevant_recall_agenda_items(
            "What workout gear should I bring?",
            &agenda_items,
            2,
            Some(&model),
        );

        assert!(relevant_memories.is_empty());
        assert!(relevant_due_items.is_empty());
        assert!(relevant_agenda_items.is_empty());
    }

    #[test]
    fn select_relevant_recall_helpers_can_expand_beyond_overlap_candidates() {
        let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"This memory matches semantically."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"This due item matches semantically."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"answers":[true],"reasoning":"This agenda item matches semantically."}"#
                        .to_string(),
            }],
        ]);
        let memories = vec![MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "practice gear".to_string(),
            file_path: "/tmp/practice_gear.md".to_string(),
            body: "Pack resistance bands before training.".to_string(),
            is_active: true,
            updated_at_unix: 10,
        }];
        let due_items = vec![AgendaItemRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "bring backup cleats".to_string(),
            file_path: "/tmp/backup_cleats.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Carry the spare cleats in the trunk.".to_string(),
            trigger_datetime: None,
            trigger_context: Some("before scrimmage".to_string()),
            is_active: true,
            updated_at_unix: 11,
        }];
        let agenda_items = vec![AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "practice packing list".to_string(),
            file_path: "/tmp/practice_packing_list.md".to_string(),
            agenda_date: Some("2026-05-21".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Review the equipment checklist before leaving.".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        }];

        let relevant_memories = select_relevant_recall_memories(
            "What gear should I bring to practice?",
            &memories,
            &HashSet::new(),
            2,
            Some(&model),
        );
        let relevant_due_items = select_relevant_recall_due_items(
            "What gear should I bring to practice?",
            &due_items,
            2,
            Some(&model),
        );
        let relevant_agenda_items = select_relevant_recall_agenda_items(
            "What gear should I bring to practice?",
            &agenda_items,
            2,
            Some(&model),
        );

        assert_eq!(relevant_memories.len(), 1);
        assert_eq!(relevant_memories[0].name, "practice gear");
        assert_eq!(relevant_due_items.len(), 1);
        assert_eq!(relevant_due_items[0].name, "bring backup cleats");
        assert_eq!(relevant_agenda_items.len(), 1);
        assert_eq!(relevant_agenda_items[0].name, "practice packing list");
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

    fn wait_for_background_status_message(expected_status: &str) {
        let started = Instant::now();
        loop {
            if get_background_status().as_deref() == Some(expected_status) {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "timed out waiting for background status {expected_status}, last status {:?}",
                get_background_status()
            );
            thread::sleep(Duration::from_millis(25));
        }
    }
}
