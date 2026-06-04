use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use elroy_codex::{CodexSessionResult, get_codex_session_by_thread_id, list_recent_codex_sessions};
use elroy_config::{AppConfig, LlmProvider, embedding_provider_config_from_app_config};
use elroy_core::{
    ConversationOrchestrator, LocalToolExecutor, ModelClient, StreamingModelClient,
    TurnEventStream, clear_background_status, get_background_status, set_background_status,
};
use elroy_db::{
    BootstrapPlan, LOCAL_USER_TOKEN, list_active_due_items, list_active_plain_agenda_items,
    load_context_messages, load_user_preferences, open_sqlite_connection, replace_context_messages,
    run_migrations,
};
use elroy_feature_requests::{is_active_feature_request, update_feature_request};
use elroy_llm::{ConversationMessage, MessageRole, StreamEvent};
use elroy_tasks::{list_active_tasks, list_due_tasks};
use elroy_tools::{ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry};

use elroy_tui::{
    SidebarAction, SidebarSection, TuiCommandExecution, TuiCommandForm, TuiCommandPaletteAction,
    TuiCommandPaletteEntry, TuiCommandSource, TuiSidebarDetail, TuiSlashCommandAction, TuiSnapshot,
};
use serde_json::{Value, json};

mod context;
use context::*;

mod recall;
use recall::*;

mod reminders;
use reminders::*;

mod tool_registry;
use tool_registry::*;

mod ui_helpers;
use ui_helpers::*;

mod runtime_helpers;
use runtime_helpers::*;

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

impl From<anyhow::Error> for AppError {
    fn from(value: anyhow::Error) -> Self {
        Self::Runtime(value.to_string())
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
    show_internal_thought: bool,
    connection: rusqlite::Connection,
    turn_stream: TurnEventStream,
    existing_transcript_len: usize,
    transient_context_count: usize,
    persist_input_message: bool,
    messages_between_memory: usize,
    memories_between_consolidation: usize,
    memory_consolidation_settings: Option<MemoryConsolidationSettings>,
    messages_between_self_reflection: usize,
    defer_auto_memory: bool,
    defer_self_reflection: bool,
    prelude_events: VecDeque<StreamEvent>,
}

struct PromptStreamUiOptions {
    home_dir: PathBuf,
    show_internal_thought: bool,
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
    memory_consolidation_settings: Option<MemoryConsolidationSettings>,
    messages_between_self_reflection: usize,
    defer_auto_memory: bool,
    defer_self_reflection: bool,
    memory_recall_classifier_enabled: bool,
    memory_recall_classifier_window: usize,
    reflect: bool,
}

#[derive(Clone)]
pub struct AppRuntime {
    config: AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook_override: Option<Arc<dyn Fn(CodexSessionResult) + Send + Sync>>,
}

pub fn is_bootstrap_session_context_message(message: &ConversationMessage) -> bool {
    ui_helpers::is_bootstrap_session_context_message(message)
}

impl AppRuntime {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            codex_bin_override: None,
            codex_completion_hook_override: None,
        }
    }

    pub fn with_codex_bin_override(mut self, codex_bin_override: PathBuf) -> Self {
        self.codex_bin_override = Some(codex_bin_override);
        self
    }

    pub fn with_codex_completion_hook(
        mut self,
        codex_completion_hook_override: Arc<dyn Fn(CodexSessionResult) + Send + Sync>,
    ) -> Self {
        self.codex_completion_hook_override = Some(codex_completion_hook_override);
        self
    }

    fn tool_registry(&self) -> ExecutableToolRegistry {
        build_live_tool_registry_with_codex_bin_and_hook(
            &self.config,
            self.codex_bin_override.clone(),
            self.codex_completion_hook_override.clone(),
        )
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
            self.config.show_internal_thought,
        )?;
        let mut slash_command_names = self
            .tool_registry()
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
        .map_err(AppError::from)
    }

    pub fn append_startup_session_context(&self) -> Result<(), AppError> {
        let mut connection = self.open_connection()?;
        let mut transcript = load_validated_runtime_transcript(
            &mut connection,
            &self.config.assistant_name,
            self.config.llm_provider() == LlmProvider::Anthropic,
        )?;
        let tool_call_id = format!(
            "bootstrap-session-context:{}",
            Utc::now()
                .timestamp_nanos_opt()
                .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000)
        );
        transcript.extend(synthetic_tool_context_messages(
            tool_call_id,
            "get_session_context",
            "{}",
            build_session_context_message(&mut connection)?,
        ));
        replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)?;
        Ok(())
    }

    pub fn load_command_palette_entries(&self) -> Result<Vec<TuiCommandPaletteEntry>, AppError> {
        let mut entries = self
            .tool_registry()
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
        let registry = self.tool_registry();
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
                source: TuiCommandSource::Palette,
            }));
        }
        Ok(TuiSlashCommandAction::OpenForm(TuiCommandForm {
            command_name: name.to_string(),
            description: spec.description,
            parameters,
            initial_values: vec![],
            source: TuiCommandSource::Palette,
        }))
    }

    pub fn handle_slash_command(&self, prompt: &str) -> Result<TuiSlashCommandAction, AppError> {
        let trimmed = prompt.trim();
        let Some(command_text) = trimmed.strip_prefix('/') else {
            return Ok(TuiSlashCommandAction::NotHandled);
        };
        let parts = command_text.split_whitespace().collect::<Vec<_>>();
        if parts.is_empty() {
            return Ok(TuiSlashCommandAction::NotHandled);
        }

        let slash_name = parts[0];
        let raw_values = &parts[1..];
        let command_name = match slash_name {
            "help" => "get_help",
            _ => slash_name,
        };
        let registry = self.tool_registry();
        let Some(spec) = registry
            .specs()
            .into_iter()
            .find(|spec| spec.name == command_name)
        else {
            return Ok(TuiSlashCommandAction::NotHandled);
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
                source: TuiCommandSource::Slash,
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
            source: TuiCommandSource::Slash,
        }))
    }

    pub fn execute_command(
        &self,
        command_name: &str,
        display_name: &str,
        values: &[(String, String)],
        source: &TuiCommandSource,
    ) -> Result<TuiSnapshot, AppError> {
        self.execute_command_with_values(command_name, display_name, values, source)
    }

    fn execute_command_with_values(
        &self,
        command_name: &str,
        _display_name: &str,
        values: &[(String, String)],
        source: &TuiCommandSource,
    ) -> Result<TuiSnapshot, AppError> {
        let registry = self.tool_registry();
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
            && *source == TuiCommandSource::Palette
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
            format!("command failed: {trimmed_content}")
        } else if use_toast_result {
            trimmed_content.to_string()
        } else {
            String::new()
        });
        if snapshot
            .status
            .as_ref()
            .is_some_and(|status| status.is_empty())
        {
            snapshot.status = None;
        }
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
                Some(&memory_consolidation_settings_from_app_config(&self.config)),
                existing_transcript_len,
                transcript.as_slice(),
                self.config.messages_between_memory,
            )
        })();
        clear_background_status("auto-memory");
        result.map_err(AppError::from)
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
        let classifier_model = live_fast_provider_model(&self.config, preferences.as_ref())?;
        let embedding_provider_config =
            embedding_provider_config_from_app_config(&self.config).ok();
        let embedding_client = best_effort_embedding_client(embedding_provider_config.as_ref());
        let executable_tools = if options.enable_tools {
            self.tool_registry()
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
            PromptStreamUiOptions {
                home_dir: self.config.home_dir.clone(),
                show_internal_thought: self.config.show_internal_thought,
            },
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
                memory_consolidation_settings: Some(memory_consolidation_settings_from_app_config(
                    &self.config,
                )),
                messages_between_self_reflection: self.config.messages_between_self_reflection,
                defer_auto_memory: options.defer_auto_memory,
                defer_self_reflection: options.defer_self_reflection,
                memory_recall_classifier_enabled: self.config.memory_recall_classifier_enabled,
                memory_recall_classifier_window: self.config.memory_recall_classifier_window,
                reflect: self.config.reflect,
            },
            Box::new(model),
            RecallModelClients {
                classifier_model: Some(&classifier_model),
                embedding_client: embedding_client.as_ref(),
                embedding_distance_threshold: Some(
                    self.config.l2_memory_relevance_distance_threshold as f32,
                ),
                recency_weight: self.config.recency_weight as f32,
                reflection_max_words: self.config.memory_reflection_max_words,
            },
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
        let classifier_model = live_fast_provider_model(&self.config, preferences.as_ref())?;
        let embedding_provider_config =
            embedding_provider_config_from_app_config(&self.config).ok();
        let embedding_client = best_effort_embedding_client(embedding_provider_config.as_ref());
        let executable_tools = if options.enable_tools {
            self.tool_registry()
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
            RecallModelClients {
                classifier_model: Some(&classifier_model),
                embedding_client: embedding_client.as_ref(),
                embedding_distance_threshold: Some(
                    self.config.l2_memory_relevance_distance_threshold as f32,
                ),
                recency_weight: self.config.recency_weight as f32,
                reflection_max_words: self.config.memory_reflection_max_words,
            },
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
                memory_consolidation_settings: Some(memory_consolidation_settings_from_app_config(
                    &self.config,
                )),
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
                self.config.show_internal_thought,
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
                    format!("Status: {}", session.status),
                    format!(
                        "Updated: {}",
                        Utc.timestamp_opt(session.updated_at_unix, 0)
                            .single()
                            .unwrap_or_else(Utc::now)
                            .to_rfc3339()
                    ),
                    format!("Repo: {}", session.repo_path),
                ];
                if let Some(worktree_path) = session.worktree_path.as_ref() {
                    lines.push(format!("Worktree: {worktree_path}"));
                }
                if let Some(session_branch) = session.session_branch.as_ref() {
                    lines.push(format!("Session Branch: {session_branch}"));
                }
                if let Some(target_branch) = session.target_branch.as_ref() {
                    lines.push(format!("Target Branch: {target_branch}"));
                }
                if let Some(session_file_path) = session.session_file_path.as_ref() {
                    lines.push(format!("Session File: {session_file_path}"));
                }
                lines.extend([
                    String::new(),
                    "Summary:".to_string(),
                    session
                        .latest_summary
                        .clone()
                        .unwrap_or_else(|| "(No summary recorded.)".to_string()),
                ]);
                if let Some(agent_message) = session.latest_agent_message.as_ref() {
                    lines.extend([
                        String::new(),
                        "Latest Agent Message:".to_string(),
                        agent_message.clone(),
                    ]);
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
        let registry = self.tool_registry();
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

fn run_prompt_with_model_and_registry_internal(
    connection: &mut rusqlite::Connection,
    prompt: &str,
    model: &dyn ModelClient,
    recall_models: RecallModelClients<'_>,
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
        recall_models.classifier_model,
    );
    let recall_source_limit = semantic_recall_source_fetch_limit(
        20,
        recall_models.classifier_model,
        recall_models.embedding_client,
    );
    let all_due_items = list_active_due_items(connection, recall_source_limit)?;
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let timed_due_items = list_due_tasks(connection, recall_source_limit, &now_iso)?;
    let timed_due_item_ids = timed_due_items
        .iter()
        .map(|item| item.id)
        .collect::<HashSet<_>>();
    let recall_due_items = all_due_items
        .iter()
        .filter(|item| !timed_due_item_ids.contains(&item.id))
        .cloned()
        .collect::<Vec<_>>();
    let recall_context = recall_memory_context_messages_with_decision(
        options.memory_recall_classifier_window,
        options.reflect,
        prompt,
        memory_recall_decision.needs_recall,
        recall_models.reflection_max_words,
        RecallSelectionClients {
            limit: 2,
            relevance_model: recall_models.classifier_model,
            embedding_client: recall_models.embedding_client,
            embedding_distance_threshold: recall_models.embedding_distance_threshold,
            recency_weight: recall_models.recency_weight,
            connection: Some(connection),
            query_embedding: None,
            now_iso: Some(&now_iso),
        },
        RecallContext {
            transcript: &existing_transcript,
            memories: &list_active_memories_in_scope(
                connection,
                &options.bootstrap_plan.memory_dir,
                semantic_recall_source_fetch_limit(
                    50,
                    recall_models.classifier_model,
                    recall_models.embedding_client,
                ),
            )?,
            due_items: &recall_due_items,
            agenda_items: &list_active_plain_agenda_items(connection, recall_source_limit)?,
        },
    );
    let due_item_context = build_due_item_surfacing_context(
        prompt,
        &existing_transcript,
        &recall_context,
        &timed_due_items,
        &all_due_items,
        &now_iso,
        RecallSelectionClients {
            limit: 2,
            relevance_model: recall_models.classifier_model,
            embedding_client: recall_models.embedding_client,
            embedding_distance_threshold: recall_models.embedding_distance_threshold,
            recency_weight: recall_models.recency_weight,
            connection: Some(connection),
            query_embedding: None,
            now_iso: Some(&now_iso),
        },
    );
    let timed_due_item_context = due_item_context.timed_due_item_context;
    let contextual_due_item_context = due_item_context.contextual_due_item_context;
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
            timed_due_item_context.len(),
        ),
        persisted_transcript_start_len,
        options.persist_input_message,
    );
    replace_context_messages(connection, LOCAL_USER_TOKEN, &persisted_transcript)?;
    run_auto_memory_if_needed(
        connection,
        &options.bootstrap_plan,
        options.memories_between_consolidation,
        options.memory_consolidation_settings.as_ref(),
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
        RecallModelClients::default(),
        executable_tools,
        options,
    )
}

fn run_prompt_with_model_and_registry_stream_internal(
    mut connection: rusqlite::Connection,
    ui_options: PromptStreamUiOptions,
    prompt: &str,
    options: PromptExecutionOptions<'_>,
    model: Box<dyn StreamingModelClient>,
    recall_models: RecallModelClients<'_>,
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
        recall_models.classifier_model,
    );
    let recall_source_limit = semantic_recall_source_fetch_limit(
        20,
        recall_models.classifier_model,
        recall_models.embedding_client,
    );
    let all_due_items = list_active_due_items(&connection, recall_source_limit)?;
    let now_iso = Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string();
    let timed_due_items = list_due_tasks(&connection, recall_source_limit, &now_iso)?;
    let timed_due_item_ids = timed_due_items
        .iter()
        .map(|item| item.id)
        .collect::<HashSet<_>>();
    let recall_due_items = all_due_items
        .iter()
        .filter(|item| !timed_due_item_ids.contains(&item.id))
        .cloned()
        .collect::<Vec<_>>();
    let recall_context = recall_memory_context_messages_with_decision(
        options.memory_recall_classifier_window,
        options.reflect,
        prompt,
        memory_recall_decision.needs_recall,
        recall_models.reflection_max_words,
        RecallSelectionClients {
            limit: 2,
            relevance_model: recall_models.classifier_model,
            embedding_client: recall_models.embedding_client,
            embedding_distance_threshold: recall_models.embedding_distance_threshold,
            recency_weight: recall_models.recency_weight,
            connection: Some(&connection),
            query_embedding: None,
            now_iso: Some(&now_iso),
        },
        RecallContext {
            transcript: &existing_transcript,
            memories: &list_active_memories_in_scope(
                &connection,
                &options.bootstrap_plan.memory_dir,
                semantic_recall_source_fetch_limit(
                    50,
                    recall_models.classifier_model,
                    recall_models.embedding_client,
                ),
            )?,
            due_items: &recall_due_items,
            agenda_items: &list_active_plain_agenda_items(&connection, recall_source_limit)?,
        },
    );
    let due_item_context = build_due_item_surfacing_context(
        prompt,
        &existing_transcript,
        &recall_context,
        &timed_due_items,
        &all_due_items,
        &now_iso,
        RecallSelectionClients {
            limit: 2,
            relevance_model: recall_models.classifier_model,
            embedding_client: recall_models.embedding_client,
            embedding_distance_threshold: recall_models.embedding_distance_threshold,
            recency_weight: recall_models.recency_weight,
            connection: Some(&connection),
            query_embedding: None,
            now_iso: Some(&now_iso),
        },
    );
    let timed_due_item_context = due_item_context.timed_due_item_context;
    let contextual_due_item_context = due_item_context.contextual_due_item_context;
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
    );

    let prelude_events = VecDeque::from(prompt_prelude_status_updates_with_decision(
        memory_recall_decision.used_llm,
        !recall_context.is_empty(),
        !(timed_due_item_context.is_empty() && contextual_due_item_context.is_empty()),
    ));

    Ok(PromptEventStream {
        state: Some(PromptEventStreamState {
            home_dir: ui_options.home_dir,
            bootstrap_plan: options.bootstrap_plan,
            show_internal_thought: ui_options.show_internal_thought,
            connection,
            turn_stream,
            existing_transcript_len: persisted_transcript_start_len,
            transient_context_count: timed_due_item_context.len(),
            persist_input_message: options.persist_input_message,
            messages_between_memory: options.messages_between_memory,
            memories_between_consolidation: options.memories_between_consolidation,
            memory_consolidation_settings: options.memory_consolidation_settings,
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
        PromptStreamUiOptions {
            home_dir,
            show_internal_thought: false,
        },
        prompt,
        options,
        model,
        RecallModelClients::default(),
        executable_tools,
    )
}

#[cfg(test)]
mod tests;
