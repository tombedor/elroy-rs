use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{Local, NaiveDateTime, Utc};
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
    StreamingModelClient, TurnEventStream, clear_background_status, get_background_status,
    set_background_status, validated_transcript,
};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, UserPreferenceRecord, find_active_agenda_item_by_name,
    find_active_memory_by_name, get_or_create_memory_operation_tracker, list_active_due_items,
    list_active_memories, list_active_plain_agenda_items, list_inactive_due_items,
    load_context_messages, load_user_preferences, open_sqlite_connection, replace_context_messages,
    run_migrations, save_memory_operation_tracker, save_user_preferences, search_active_memories,
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
use elroy_memory::{archive_memory_file, create_memory_file, update_memory_body};
use elroy_self_reflection::{SelfReflectionConfig, SelfReflectionOrchestrator};
use elroy_tasks::{
    complete_task_file, create_task_file_with_schedule, delete_task_file, find_task_by_name,
    list_active_tasks, list_due_tasks, list_today_tasks, list_triggered_tasks, rename_task_file,
    update_task_text_file,
};
use elroy_tools::{
    ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry, ToolSpec,
};
use elroy_tui::{SidebarAction, SidebarSection, TuiSidebarDetail, TuiSnapshot};
use elroy_user::{effective_persona, effective_user_full_name, effective_user_preferred_name};
use serde_json::{Value, json};

const LOCAL_USER_TOKEN: &str = "local-user";
const SYNTHETIC_FIRST_USER_MESSAGE: &str = "The user has begun the conversation";
const DEFAULT_MAX_LIST_ENTRIES: usize = 50;
const DEFAULT_MAX_LIST_DEPTH: usize = 2;
const DEFAULT_READ_LINE_LIMIT: usize = 200;

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

pub struct PromptEventStream {
    state: Option<PromptEventStreamState>,
    finalized_snapshot: Option<Result<TuiSnapshot, AppError>>,
}

impl PromptEventStream {
    pub fn snapshot(&self) -> Option<&TuiSnapshot> {
        self.finalized_snapshot
            .as_ref()
            .and_then(|result| result.as_ref().ok())
    }

    pub fn into_snapshot(mut self) -> Result<TuiSnapshot, AppError> {
        while self.next().is_some() {}
        self.finalized_snapshot
            .take()
            .unwrap_or_else(|| Err(AppError::Runtime("stream did not finalize".to_string())))
    }

    pub fn cancel(mut self) -> Result<TuiSnapshot, AppError> {
        if let Some(result) = self.finalized_snapshot.take() {
            return result;
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
                self.finalized_snapshot = Some(Err(AppError::from(error)));
                self.state = None;
                None
            }
            None => {
                let state = self.state.take().expect("stream state should exist");
                self.finalized_snapshot = Some(finalize_prompt_event_stream(state));
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
    messages_between_self_reflection: usize,
    prelude_events: VecDeque<StreamEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageProcessOptions {
    pub role: MessageRole,
    pub enable_tools: bool,
    pub persist_input_message: bool,
    pub force_tool: Option<String>,
}

impl Default for MessageProcessOptions {
    fn default() -> Self {
        Self {
            role: MessageRole::User,
            enable_tools: true,
            persist_input_message: true,
            force_tool: None,
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
    messages_between_self_reflection: usize,
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
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
        let mut snapshot = load_snapshot_from_connection(&mut connection, &self.config.home_dir)?;
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

    pub fn refresh_context_if_needed(&self) -> Result<bool, AppError> {
        let mut connection = self.open_connection()?;
        refresh_context_if_needed(
            &mut connection,
            &self.config,
            &BootstrapPlan::from_config(&self.config),
        )
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
        run_prompt_with_model_and_registry_stream(
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
                messages_between_self_reflection: self.config.messages_between_self_reflection,
                memory_recall_classifier_enabled: self.config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: self.config.memory_recall_classifier_window,
            },
            Box::new(model),
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
        let events = run_prompt_with_model_and_registry(
            &mut connection,
            prompt,
            &model,
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
                messages_between_self_reflection: self.config.messages_between_self_reflection,
                memory_recall_classifier_enabled: self.config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: self.config.memory_recall_classifier_window,
            },
        )?;

        Ok(PromptRunResult {
            events,
            snapshot: load_snapshot_from_connection(&mut connection, &self.config.home_dir)?,
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
                ..MessageProcessOptions::default()
            },
        )
        .map(Some)
    }

    pub fn open_sidebar_item(
        &self,
        section: SidebarSection,
        title: &str,
    ) -> Result<TuiSidebarDetail, AppError> {
        let connection = self.open_read_connection()?;
        match section {
            SidebarSection::Memories => {
                let Some(memory) = find_active_memory_by_name(&connection, title)? else {
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
}

fn load_snapshot_from_connection(
    connection: &mut rusqlite::Connection,
    home_dir: &Path,
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
    let memory_titles = list_active_memories(connection, 15)?
        .into_iter()
        .map(|memory| memory.name)
        .collect::<Vec<_>>();
    let now = Utc::now().naive_utc();
    let agenda_titles = list_active_tasks(connection, 15)?
        .into_iter()
        .map(|item| format_agenda_sidebar_title(&item, now))
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
) -> Result<TuiSnapshot, AppError> {
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
    run_auto_memory_if_needed(
        &mut state.connection,
        &state.bootstrap_plan,
        state.existing_transcript_len,
        persisted_transcript.as_slice(),
        state.messages_between_memory,
    )?;
    run_self_reflection_if_needed(
        &state.home_dir,
        persisted_transcript.as_slice(),
        state.messages_between_self_reflection,
    )?;
    load_snapshot_from_connection(&mut state.connection, &state.home_dir)
}

fn cancel_prompt_event_stream(mut state: PromptEventStreamState) -> Result<TuiSnapshot, AppError> {
    load_snapshot_from_connection(&mut state.connection, &state.home_dir)
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
    let recall_context = recall_memory_context_messages(
        options.memory_recall_classifier_enabled,
        options.memory_recall_classifier_window,
        prompt,
        &existing_transcript,
        &list_active_memories(connection, 50)?,
    );
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let all_due_items = list_active_due_items(connection, 20)?;
    let timed_due_item_context =
        due_item_context_messages(&list_due_tasks(connection, 20, &now_iso)?);
    let contextual_due_item_context =
        recall_due_item_context_messages(prompt, &existing_transcript, &all_due_items, &now_iso);
    let mut model_transcript = existing_transcript.clone();
    model_transcript.extend(recall_context.iter().cloned());
    model_transcript.extend(timed_due_item_context.iter().cloned());
    model_transcript.extend(contextual_due_item_context.iter().cloned());

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
            existing_transcript.len(),
            recall_context.len() + timed_due_item_context.len() + contextual_due_item_context.len(),
        ),
        existing_transcript.len(),
        options.persist_input_message,
    );
    replace_context_messages(connection, LOCAL_USER_TOKEN, &persisted_transcript)?;
    run_auto_memory_if_needed(
        connection,
        &options.bootstrap_plan,
        existing_transcript.len(),
        persisted_transcript.as_slice(),
        options.messages_between_memory,
    )?;
    run_self_reflection_if_needed(
        options.home_dir,
        persisted_transcript.as_slice(),
        options.messages_between_self_reflection,
    )?;

    let mut events = prompt_prelude_status_updates(
        options.memory_recall_classifier_enabled,
        prompt,
        !recall_context.is_empty(),
        !(timed_due_item_context.is_empty() && contextual_due_item_context.is_empty()),
    );
    events.extend(turn_run.events);
    Ok(events)
}

fn run_prompt_with_model_and_registry_stream(
    mut connection: rusqlite::Connection,
    home_dir: PathBuf,
    prompt: &str,
    options: PromptExecutionOptions<'_>,
    model: Box<dyn StreamingModelClient>,
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
    let recall_context = recall_memory_context_messages(
        options.memory_recall_classifier_enabled,
        options.memory_recall_classifier_window,
        prompt,
        &existing_transcript,
        &list_active_memories(&connection, 50)?,
    );
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let all_due_items = list_active_due_items(&connection, 20)?;
    let timed_due_item_context =
        due_item_context_messages(&list_due_tasks(&connection, 20, &now_iso)?);
    let contextual_due_item_context =
        recall_due_item_context_messages(prompt, &existing_transcript, &all_due_items, &now_iso);
    let mut model_transcript = existing_transcript.clone();
    model_transcript.extend(recall_context.iter().cloned());
    model_transcript.extend(timed_due_item_context.iter().cloned());
    model_transcript.extend(contextual_due_item_context.iter().cloned());

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

    let prelude_events = VecDeque::from(prompt_prelude_status_updates(
        options.memory_recall_classifier_enabled,
        prompt,
        !recall_context.is_empty(),
        !(timed_due_item_context.is_empty() && contextual_due_item_context.is_empty()),
    ));

    Ok(PromptEventStream {
        state: Some(PromptEventStreamState {
            home_dir,
            bootstrap_plan: options.bootstrap_plan,
            connection,
            turn_stream,
            existing_transcript_len: existing_transcript.len(),
            transient_context_count: recall_context.len()
                + timed_due_item_context.len()
                + contextual_due_item_context.len(),
            persist_input_message: options.persist_input_message,
            messages_between_memory: options.messages_between_memory,
            messages_between_self_reflection: options.messages_between_self_reflection,
            prelude_events,
        }),
        finalized_snapshot: None,
    })
}

fn run_self_reflection_if_needed(
    home_dir: &Path,
    transcript: &[ConversationMessage],
    messages_between_self_reflection: usize,
) -> Result<(), AppError> {
    SelfReflectionOrchestrator::new(SelfReflectionConfig {
        messages_between_self_reflection,
    })
    .run(home_dir, transcript)?;
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
        let removed_prefix_len = transcript.len().saturating_sub(compressed.len());
        let dropped_messages = transcript
            .iter()
            .skip(1)
            .take(removed_prefix_len)
            .cloned()
            .collect::<Vec<_>>();

        if transcript
            .iter()
            .any(|message| message.role == MessageRole::User)
        {
            let (name, text) = formulate_memory_from_transcript(&transcript);
            create_memory_file(&bootstrap_plan.memory_dir, &name, &text)?;
            elroy_db::bootstrap_database(bootstrap_plan)
                .map_err(|error| AppError::Runtime(error.to_string()))?;
            *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;

            let mut tracker = get_or_create_memory_operation_tracker(connection, LOCAL_USER_TOKEN)?;
            tracker.messages_since_memory = 0;
            tracker.memories_since_consolidation += 1;
            tracker.updated_at_unix = Utc::now().timestamp();
            save_memory_operation_tracker(connection, &tracker)?;
        }

        let mut refreshed_transcript = compressed;
        if !dropped_messages.is_empty() {
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
                format_context_summary_message(&dropped_messages),
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
    create_memory_file(&bootstrap_plan.memory_dir, &name, &text)?;
    elroy_db::bootstrap_database(bootstrap_plan)
        .map_err(|error| AppError::Runtime(error.to_string()))?;
    *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;

    tracker.messages_since_memory = 0;
    tracker.memories_since_consolidation += 1;
    tracker.updated_at_unix = Utc::now().timestamp();
    save_memory_operation_tracker(connection, &tracker)?;
    Ok(())
}

fn reset_memory_tracker_after_creation(config: &AppConfig) -> Result<(), AppError> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let mut tracker = get_or_create_memory_operation_tracker(&mut connection, LOCAL_USER_TOKEN)?;
    tracker.messages_since_memory = 0;
    tracker.memories_since_consolidation += 1;
    tracker.updated_at_unix = Utc::now().timestamp();
    save_memory_operation_tracker(&mut connection, &tracker)?;
    Ok(())
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
    run_prompt_with_model_and_registry(
        &mut connection,
        &prompt,
        &model,
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
            messages_between_self_reflection: config.messages_between_self_reflection,
            memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
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
            let start_line = arguments
                .get("start_line")
                .and_then(Value::as_i64)
                .unwrap_or(1);
            let end_line = arguments.get("end_line").and_then(Value::as_i64);
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
                    reset_memory_tracker_after_creation(&config_for_memory_write)
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

    let get_fast_recall = ExecutableTool::new(
        ToolSpec::new(
            "get_fast_recall",
            "No-op tool used to acknowledge synthetic recall context.",
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

    let config_for_outdated_memory_update = config.clone();
    let update_outdated_or_incorrect_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_outdated_or_incorrect_memory",
            "Append corrective information to one active memory by exact name.",
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
            mutate_memory_file_from_config(
                &config_for_outdated_memory_update,
                memory_name,
                |path| {
                    let existing = std::fs::read_to_string(path)?;
                    let mut updated = existing.trim_end_matches('\n').to_string();
                    if !updated.is_empty() {
                        updated.push_str("\n\n");
                    }
                    updated.push_str(&format!(
                        "Update ({}):\n{}",
                        Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                        update_text.trim()
                    ));
                    update_memory_body(path, &updated)
                },
            );
            ToolExecutionResult::success(format!("Memory '{memory_name}' has been updated"))
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
            let memory = match find_active_memory_by_name(&connection, memory_name) {
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
            let Some(memory) = (match find_active_memory_by_name(&connection, memory_name) {
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

            let completion_hook = {
                let upstream_hook = codex_completion_hook_for_dispatch.clone();
                Arc::new(move |result: CodexSessionResult| {
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
    let print_memories = ExecutableTool::new(
        ToolSpec::new(
            "print_memories",
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
                let due_items = list_active_due_items(connection, limit * 3)?;
                let relevant_due_items =
                    select_due_items_by_overlap(query, &due_items, limit, None);
                let mut payload = memories
                    .into_iter()
                    .map(|memory| {
                        json!({
                            "type": "memory",
                            "name": memory.name,
                            "file_path": memory.file_path,
                            "excerpt": excerpt(&memory.body, 180),
                            "updated_at_unix": memory.updated_at_unix,
                        })
                    })
                    .collect::<Vec<_>>();
                payload.extend(relevant_due_items.into_iter().map(|item| {
                    json!({
                        "type": "due_item",
                        "name": item.name,
                        "file_path": item.file_path,
                        "trigger_datetime": item.trigger_datetime,
                        "trigger_context": item.trigger_context,
                        "excerpt": excerpt(&item.body, 180),
                        "updated_at_unix": item.updated_at_unix,
                    })
                }));
                Ok(ToolExecutionResult::success(
                    serde_json::to_string_pretty(&payload)
                        .expect("memory search payload should serialize"),
                ))
            })
        },
    );

    let database_path = config.database_path.clone();
    let examine_memories = ExecutableTool::new(
        ToolSpec::new(
            "examine_memories",
            "Search memories and due items for the answer to a question.",
            JsonSchema::object(
                [
                    ("question", json!({"type": "string"})),
                    ("limit", json!({"type": "integer"})),
                ],
                ["question"],
            ),
        ),
        move |arguments| {
            let Some(question) = arguments.get("question").and_then(Value::as_str) else {
                return ToolExecutionResult::error("examine_memories requires a string question");
            };
            let limit = argument_limit(&arguments, 10);
            with_tool_connection(&database_path, |connection| {
                let memories = list_active_memories(connection, limit * 3)?;
                let relevant_memories =
                    select_recalled_memories(question, &memories, &HashSet::new(), limit);
                let due_items = list_active_due_items(connection, limit * 3)?;
                let relevant_due_items =
                    select_due_items_by_overlap(question, &due_items, limit, None);

                let mut sections = relevant_memories
                    .into_iter()
                    .map(|memory| format!("# Memory: {}\n\n{}", memory.name, memory.body.trim()))
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
            let target_date = arguments
                .get("item_date")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| Local::now().date_naive().format("%Y-%m-%d").to_string());
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
    let print_memory = ExecutableTool::new(
        ToolSpec::new(
            "print_memory",
            "Show one active memory by exact name.",
            JsonSchema::object([("name", json!({"type": "string"}))], ["name"]),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("print_memory requires a string name");
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
                    "index {index} out of range. Available indices: [0]"
                ));
            }
            with_tool_connection(&database_path, |connection| {
                let Some(memory) = find_active_memory_by_name(connection, memory_name)? else {
                    return Ok(ToolExecutionResult::error(format!(
                        "memory not found: {memory_name}"
                    )));
                };
                if index > 0 {
                    return Ok(ToolExecutionResult::error(format!(
                        "index {index} out of range. Available indices: [0]"
                    )));
                }
                let path = Path::new(&memory.file_path);
                let Ok(source_content) = std::fs::read_to_string(path) else {
                    return Ok(ToolExecutionResult::success(format!(
                        "No sources found for memory '{memory_name}'"
                    )));
                };
                Ok(ToolExecutionResult::success(format!(
                    "# Source content for memory: {} (0 / 0)\n\n{}",
                    memory.name,
                    source_content.trim()
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
        get_current_date,
        pwd,
        ls,
        read_file,
        create_memory,
        get_fast_recall,
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
        list_due_items,
        print_active_due_items,
        list_inactive_due_items_tool,
        print_inactive_due_items,
        show_task,
        show_due_item,
        print_due_item,
        show_memory,
        print_memory,
        get_source_content_for_memory,
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

    let lines = items
        .iter()
        .filter_map(|item| {
            let trigger_datetime = item.trigger_datetime.as_deref()?;
            Some(format!(
                "DUE ITEM: '{}' - {}\n\nThis item was scheduled for {} and is now due. Please inform the user about it and then use the delete_due_item tool to remove it from active due items.",
                item.name,
                item.body,
                trigger_datetime,
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

fn context_memory_tool_messages(memory: &elroy_db::MemoryRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&json!({
        "content": format!("MEMORY: '{}' - {}", memory.name, memory.body),
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

fn transcript_contains_context_memory(
    transcript: &[ConversationMessage],
    memory_name: &str,
) -> bool {
    let tool_call_id = context_memory_tool_call_id(memory_name);
    transcript
        .iter()
        .any(|message| message_matches_context_memory(message, &tool_call_id))
}

fn message_matches_context_memory(message: &ConversationMessage, tool_call_id: &str) -> bool {
    message.tool_call_id.as_deref() == Some(tool_call_id)
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| tool_calls.iter().any(|call| call.id == tool_call_id))
}

fn format_context_summary_message(messages: &[ConversationMessage]) -> String {
    let lines = messages
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::Tool => "Tool",
                MessageRole::System => return None,
            };
            let content = message.content.as_deref()?.trim();
            if content.is_empty() {
                return None;
            }
            Some(format!("{role}: {}", excerpt(content, 160)))
        })
        .collect::<Vec<_>>();

    if lines.is_empty() {
        "Recent conversation summary: (No earlier conversation summary available.)".to_string()
    } else {
        format!("Recent conversation summary: {}", lines.join("\n"))
    }
}

fn recall_memory_context_messages(
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
    prompt: &str,
    transcript: &[ConversationMessage],
    memories: &[elroy_db::MemoryRecord],
) -> Vec<ConversationMessage> {
    if memory_recall_classifier_enabled && should_skip_memory_recall(prompt) {
        return Vec::new();
    }

    let recall_query = build_recall_query(prompt, transcript, memory_recall_classifier_window);
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

    synthetic_tool_context_messages(
        "bootstrap-memory-recall",
        "search_memories",
        json!({
            "query": recall_query,
            "limit": recalled.len(),
        })
        .to_string(),
        content,
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

    let lines = recalled
        .iter()
        .map(|item| {
            let trigger_context = item.trigger_context.as_deref().unwrap_or("relevant context");
            format!(
                "RELEVANT DUE ITEM: '{}' - {}\n\nThis item should surface when the conversation matches this context: {}.",
                item.name,
                item.body,
                trigger_context,
            )
        })
        .collect::<Vec<_>>();

    synthetic_tool_context_messages(
        "bootstrap-contextual-due-items",
        "get_due_items",
        "{}",
        lines.join("\n\n"),
    )
}

fn memory_recall_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let skipped = should_skip_memory_recall(prompt);
    if memory_recall_classifier_enabled && !skipped {
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

fn prompt_prelude_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
    surfaced_due_items: bool,
) -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::StatusUpdate {
        content: "loading context...".to_string(),
    }];
    events.extend(memory_recall_status_updates(
        memory_recall_classifier_enabled,
        prompt,
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

fn select_recalled_due_items<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    select_due_items_by_overlap(prompt, due_items, limit, Some(now_iso))
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
    use chrono::Utc;
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
        build_recall_query, codex_background_status_key, compress_context_messages,
        count_context_tokens, drop_old_context_messages, due_item_context_messages,
        format_context_summary_message, is_context_refresh_needed, memory_recall_status_updates,
        parse_recalled_memory_names, prompt_prelude_status_updates,
        provider_config_from_app_config, recall_due_item_context_messages,
        recall_memory_context_messages, recalled_memory_names, recent_recall_context,
        refresh_context_if_needed, run_prompt_with_model_and_registry,
        run_prompt_with_model_and_registry_stream, select_due_items_by_overlap,
        select_recalled_due_items, select_recalled_memories, should_offer_greeting,
        should_skip_memory_recall, significant_tokens, strip_input_message_for_persistence,
        strip_transient_context_messages,
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
        AgendaItemRecord, BootstrapPlan, MemoryRecord, load_memory_operation_tracker,
        load_user_preferences, open_sqlite_connection, run_migrations,
    };
    use elroy_feature_requests::{list_feature_requests, write_new_feature_request};
    use elroy_llm::ToolCall;
    use elroy_llm::{ConversationMessage, MessageRole, Provider, StreamEvent};
    use elroy_memory::{create_memory_file, sanitize_filename};
    use elroy_tools::ExecutableToolRegistry;

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
        let memory = registry.invoke("show_memory", "{\"name\":\"runner notes\"}");
        let printed_memory = registry.invoke("print_memory", "{\"name\":\"runner notes\"}");
        let agenda = registry.invoke("show_agenda_item", "{\"name\":\"doctor visit\"}");
        let printed_memories = registry.invoke("print_memories", "{\"limit\":10}");

        assert!(!memory.is_error);
        assert!(memory.content.contains("remember the hill workout"));
        assert!(!printed_memory.is_error);
        assert!(printed_memory.content.contains("remember the hill workout"));
        assert!(!agenda.is_error);
        assert!(agenda.content.contains("bring forms"));
        assert!(!printed_memories.is_error);
        assert!(printed_memories.content.contains("runner notes"));

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
        let out_of_range = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\",\"index\":1}",
        );

        assert!(!source.is_error);
        assert!(
            source
                .content
                .contains("# Source content for memory: runner notes (0 / 0)")
        );
        assert!(source.content.contains("with the harder second interval"));
        assert!(out_of_range.is_error);
        assert!(out_of_range.content.contains("Available indices: [0]"));

        fs::remove_dir_all(home).expect("home should be removed");
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
    fn live_tool_registry_includes_get_fast_recall_ack_tool() {
        let registry = build_live_tool_registry(&AppConfig::defaults());
        let result = registry.invoke("get_fast_recall", "{}");

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

        let file_text =
            fs::read_to_string(memory_dir.join("runner_notes.md")).expect("memory should read");
        assert!(file_text.contains("old text"));
        assert!(file_text.contains("Update ("));
        assert!(file_text.contains("new correction"));

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
        assert!(printed.content.contains("Pay bill"));
        assert!(printed.content.contains("2026-05-15T09:00:00"));

        let inactive = registry.invoke("list_inactive_due_items", "{\"limit\":10}");
        assert!(!inactive.is_error);
        assert!(inactive.content.contains("call mom"));
        assert!(inactive.content.contains("done"));

        let printed_active = registry.invoke("print_active_due_items", "{\"limit\":10}");
        assert!(!printed_active.is_error);
        assert!(printed_active.content.contains("pay bill"));

        let printed_inactive = registry.invoke("print_inactive_due_items", "{\"limit\":10}");
        assert!(!printed_inactive.is_error);
        assert!(printed_inactive.content.contains("call mom"));
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

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        let context =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(context[0].role, MessageRole::System);
        assert_eq!(
            context[0].content.as_deref(),
            Some("You are Nova helping Jimmy.")
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
        assert!(bad_range.is_error);
        assert!(
            bad_range
                .content
                .contains("end_line must be greater than or equal to start_line")
        );

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
                && content.contains("2000-01-01T09:00:00")
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
            Some("bootstrap-contextual-due-items")
        );
        assert!(messages[1].content.as_deref().is_some_and(|content| {
            content.contains("RELEVANT DUE ITEM")
                && content.contains("after payroll email")
                && content.contains("Reply to payroll")
        }));
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
        assert!(search.content.contains("\"type\": \"due_item\""));
        assert!(search.content.contains("payroll_follow_up.md"));
        assert!(search.content.contains("after payroll email"));

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
        assert!(result.content.contains("marathon in October"));
        assert!(result.content.contains("# Due Item: running follow up"));
        assert!(result.content.contains("long run recovery"));

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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
        assert!(
            stored[stored.len() - 1]
                .content
                .as_deref()
                .is_some_and(|content| content.starts_with("Recent conversation summary:"))
        );

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
        let summary = format_context_summary_message(&[
            ConversationMessage::new(MessageRole::System, "ignore"),
            ConversationMessage::new(
                MessageRole::User,
                "A very long user message that should appear",
            ),
            ConversationMessage::new(
                MessageRole::Assistant,
                "A very long assistant message that should also appear",
            ),
        ]);

        assert!(summary.starts_with("Recent conversation summary:"));
        assert!(summary.contains("User: A very long user message"));
        assert!(summary.contains("Assistant: A very long assistant message"));
        assert!(!summary.contains("ignore"));
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
                messages_between_self_reflection: config.messages_between_self_reflection,
                memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: config.memory_recall_classifier_window,
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
        let _ = stream.into_snapshot().expect("snapshot should finalize");

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
        let _ = stream.into_snapshot().expect("snapshot should finalize");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let stored =
            elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert!(stored.iter().any(|message| {
            message.role == MessageRole::Assistant
                && message.content.as_deref() == Some("Good to see you again.")
        }));

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
        let messages = recall_memory_context_messages(
            config.memory_recall_classifier_enabled,
            config.memory_recall_classifier_window,
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
            "hi",
            &transcript,
            &[MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "practice plan".to_string(),
                file_path: "/tmp/practice.md".to_string(),
                body: "Warm up before basketball drills".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }],
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
            "What should I focus on?",
            &transcript,
            &memories,
        );

        let mut wide = AppConfig::defaults();
        wide.memory_recall_classifier_window = 3;
        let wide_messages = recall_memory_context_messages(
            wide.memory_recall_classifier_enabled,
            wide.memory_recall_classifier_window,
            "What should I focus on?",
            &transcript,
            &memories,
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
