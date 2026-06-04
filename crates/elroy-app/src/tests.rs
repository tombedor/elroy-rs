use crate::{RecallContext, RecalledItemRef};
use chrono::{Local, Utc};
use std::{
    cell::RefCell,
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use super::{
    AppRuntime, LOCAL_USER_TOKEN, MessageProcessOptions, PromptExecutionOptions,
    RecallModelClients, RecallSelectionClients, best_effort_embedding_client,
    build_live_tool_registry, build_live_tool_registry_with_codex_bin_and_hook, build_recall_query,
    classify_memory_recall_with_model, compress_context_messages,
    consolidate_exact_duplicate_memories, context_due_item_tool_call_id,
    context_due_item_tool_messages, context_task_tool_messages, count_context_tokens,
    determine_memory_recall_decision, drop_old_context_messages,
    embedding_provider_config_from_app_config, format_context_messages_for_summary,
    format_context_summary_message, is_bootstrap_session_context_message,
    is_context_refresh_needed, memory_recall_status_updates, parse_memory_recall_decision,
    parse_recalled_item_refs, parse_reflective_recall_model_response,
    parse_relevance_filter_response, prompt_prelude_status_updates, recall_memory_context_messages,
    recall_memory_context_messages_with_decision, recall_model_clients, recalled_item_refs_by_type,
    recalled_memory_names, recent_recall_context, refresh_context_if_needed,
    run_prompt_with_model_and_registry, run_prompt_with_model_and_registry_internal,
    run_prompt_with_model_and_registry_stream, select_recalled_memories,
    select_relevant_recall_agenda_items, select_relevant_recall_due_items,
    select_relevant_recall_memories, should_offer_greeting, should_skip_memory_recall,
    significant_tokens, strip_input_message_for_persistence, strip_transient_context_messages,
    summarize_context_messages_with_model, synthetic_tool_context_messages,
};
use elroy_agenda::create_agenda_file;
use elroy_codex::tools::codex_background_status_key;
use elroy_codex::{CodexSessionResult, CodexSessionUpdate, upsert_codex_session};
use elroy_config::{
    AppConfig, LlmProvider, fast_provider_config_from_app_config, provider_config_from_app_config,
};
use elroy_core::{
    ConversationRequest, ModelClient, StreamingModelClient, clear_background_status,
    get_background_status_for_key,
};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, MemoryRecord, SYNTHETIC_FIRST_USER_MESSAGE,
    list_active_due_items, load_memory_operation_tracker, open_sqlite_connection, run_migrations,
    upsert_memory_embedding,
};
use elroy_feature_requests::{list_feature_requests, write_new_feature_request};
use elroy_llm::ToolCall;
use elroy_llm::{ConversationMessage, EmbeddingProviderConfig, MessageRole, Provider, StreamEvent};
use elroy_memory::{create_memory_file, sanitize_filename};
use elroy_recall::{context_memory_tool_messages, message_matches_tool_call_id};
use elroy_tasks::create_task_file_with_schedule;
use elroy_tools::ExecutableToolRegistry;
use elroy_tools::argument_limit;
use elroy_tui::{
    TuiCommandExecution, TuiCommandPaletteAction, TuiCommandSource, TuiSlashCommandAction,
};

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
                    let normalized = content.to_ascii_lowercase();
                    normalized.contains("payroll")
                        && normalized.contains("reply to payroll")
                        && normalized.contains("after payroll email")
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
fn fast_provider_config_falls_back_to_chat_model_when_fast_model_is_unset() {
    let mut config = AppConfig::defaults();
    config.chat_model = "gpt-5.4".to_string();
    config.openai_api_key = Some("openai-key".to_string());

    let provider = fast_provider_config_from_app_config(&config).expect("config should build");

    assert_eq!(provider.provider, Provider::OpenAi);
    assert_eq!(provider.model, "gpt-5.4");
    assert_eq!(provider.api_key, "openai-key");
    assert_eq!(provider.base_url, config.openai_base_url);
}

#[test]
fn fast_provider_config_uses_fast_model_and_overrides_when_configured() {
    let mut config = AppConfig::defaults();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-key".to_string());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-key".to_string());
    config.fast_model_api_base = Some("http://localhost:4321/fast".to_string());

    let provider = fast_provider_config_from_app_config(&config).expect("config should build");

    assert_eq!(provider.provider, Provider::OpenAi);
    assert_eq!(provider.model, "gpt-5.4-mini");
    assert_eq!(provider.api_key, "fast-key");
    assert_eq!(provider.base_url, "http://localhost:4321/fast");
}

#[test]
fn embedding_provider_config_falls_back_to_openai_defaults() {
    let mut config = AppConfig::defaults();
    config.openai_api_key = Some("openai-key".to_string());

    let provider = embedding_provider_config_from_app_config(&config).expect("config should build");

    assert_eq!(provider.model, "text-embedding-3-small");
    assert_eq!(provider.api_key, "openai-key");
    assert_eq!(provider.base_url, "https://api.openai.com/v1/embeddings");
}

#[test]
fn embedding_provider_config_uses_embedding_overrides_when_configured() {
    let mut config = AppConfig::defaults();
    config.openai_api_key = Some("openai-key".to_string());
    config.embedding_model = "text-embedding-3-large".to_string();
    config.embedding_model_api_key = Some("embed-key".to_string());
    config.embedding_model_api_base = Some("http://localhost:4321/embeddings".to_string());

    let provider = embedding_provider_config_from_app_config(&config).expect("config should build");

    assert_eq!(provider.model, "text-embedding-3-large");
    assert_eq!(provider.api_key, "embed-key");
    assert_eq!(provider.base_url, "http://localhost:4321/embeddings");
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
    assert!(codex_detail.content.contains("Status: completed"));
    assert!(codex_detail.content.contains("Repo: /tmp/sample"));
    assert!(codex_detail.content.contains("Summary:"));
    assert!(
        codex_detail
            .content
            .contains("Codex inspected the parser state.")
    );
    assert!(codex_detail.content.contains("Latest Agent Message:"));
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
fn create_memory_tool_triggers_semantic_consolidation_at_threshold() {
    let unique = format!(
        "elroy-rs-app-semantic-memory-consolidation-{}",
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
    let _first_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "I went to the store today, January 1".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _second_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "I went shopping at the store on New Year's Day".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.99, 0.01]}]
            })
            .to_string(),
        )
        .create();
    let _third_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Today, New Year's Day, I bought some items at the store".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.98, 0.02]}]
            })
            .to_string(),
        )
        .create();
    let _consolidation_mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"memories":[{"name":"Shopping trip on 2024-01-01","text":"User went to the store on New Year's Day and bought some items."}]}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memories_between_consolidation = 3;
    config.min_memory_cluster_size = 3;
    config.max_memory_cluster_size = 5;
    config.memory_cluster_similarity_threshold = 0.1;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    config.embedding_model = "text-embedding-3-small".to_string();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

    let registry = build_live_tool_registry(&config);
    let first = registry.invoke(
        "create_memory",
        "{\"name\":\"Shopping trip note\",\"text\":\"I went to the store today, January 1\"}",
    );
    let second = registry.invoke(
            "create_memory",
            "{\"name\":\"New Year's shopping\",\"text\":\"I went shopping at the store on New Year's Day\"}",
        );
    let third = registry.invoke(
            "create_memory",
            "{\"name\":\"Store purchases\",\"text\":\"Today, New Year's Day, I bought some items at the store\"}",
        );
    assert!(!first.is_error);
    assert!(!second.is_error);
    assert!(!third.is_error);

    let connection = open_sqlite_connection(&database_path).expect("database should reopen");
    let active_memories =
        elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
    assert_eq!(active_memories.len(), 1);
    assert_eq!(active_memories[0].name, "shopping trip on 2024 01 01");
    assert_eq!(
        active_memories[0].body,
        "User went to the store on New Year's Day and bought some items."
    );

    let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
        .expect("tracker should load")
        .expect("tracker should exist");
    assert_eq!(tracker.memories_since_consolidation, 0);

    let source_list = registry.invoke(
        "get_source_list_for_memory",
        "{\"memory_name\":\"shopping trip on 2024 01 01\"}",
    );
    assert!(!source_list.is_error);
    let mut source_entries: Vec<(String, String)> =
        serde_json::from_str::<Vec<(String, String)>>(&source_list.content)
            .expect("source list should parse");
    source_entries.sort();
    assert_eq!(
        source_entries,
        vec![
            ("Memory".to_string(), "new year s shopping".to_string()),
            ("Memory".to_string(), "shopping trip note".to_string()),
            ("Memory".to_string(), "store purchases".to_string()),
        ]
    );

    assert!(
        memory_dir
            .join("archive")
            .join("shopping_trip_note.md")
            .exists()
    );
    assert!(
        memory_dir
            .join("archive")
            .join("new_year_s_shopping.md")
            .exists()
    );
    assert!(
        memory_dir
            .join("archive")
            .join("store_purchases.md")
            .exists()
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn create_memory_tool_triggers_semantic_consolidation_with_python_style_markdown_outputs() {
    let unique = format!(
        "elroy-rs-app-markdown-memory-consolidation-{}",
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
    let _first_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "I went to the store today, January 1".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _second_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "I bought milk and bread while shopping".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.98, 0.02]}]
            })
            .to_string(),
        )
        .create();
    let _third_embedding = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "I still need to compare grocery prices later".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.97, 0.03]}]
            })
            .to_string(),
        )
        .create();
    let _consolidation_mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "# Memory Consolidation Reasoning\nI split the shopping memories into a factual trip summary and a follow-up pricing reminder.\n\n## Shopping trip on 2024-01-01\nUser went to the store on New Year's Day and bought milk and bread.\n\n## User's grocery price comparison follow-up\nUser still wants to compare grocery prices after the New Year's Day shopping trip."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memories_between_consolidation = 3;
    config.min_memory_cluster_size = 3;
    config.max_memory_cluster_size = 5;
    config.memory_cluster_similarity_threshold = 0.1;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    config.embedding_model = "text-embedding-3-small".to_string();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

    let registry = build_live_tool_registry(&config);
    let first = registry.invoke(
        "create_memory",
        "{\"name\":\"Shopping trip note\",\"text\":\"I went to the store today, January 1\"}",
    );
    let second = registry.invoke(
        "create_memory",
        "{\"name\":\"Grocery purchases\",\"text\":\"I bought milk and bread while shopping\"}",
    );
    let third = registry.invoke(
            "create_memory",
            "{\"name\":\"Price comparison follow-up\",\"text\":\"I still need to compare grocery prices later\"}",
        );
    assert!(!first.is_error);
    assert!(!second.is_error);
    assert!(!third.is_error);

    let connection = open_sqlite_connection(&database_path).expect("database should reopen");
    let active_memories =
        elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
    assert_eq!(active_memories.len(), 2);
    assert!(
        active_memories
            .iter()
            .any(|memory| memory.name == "shopping trip on 2024 01 01")
    );
    assert!(
        active_memories
            .iter()
            .any(|memory| memory.name == "user s grocery price comparison follow up")
    );

    let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
        .expect("tracker should load")
        .expect("tracker should exist");
    assert_eq!(tracker.memories_since_consolidation, 0);

    let shopping_source_list = registry.invoke(
        "get_source_list_for_memory",
        "{\"memory_name\":\"shopping trip on 2024 01 01\"}",
    );
    assert!(!shopping_source_list.is_error);
    let mut shopping_sources: Vec<(String, String)> =
        serde_json::from_str::<Vec<(String, String)>>(&shopping_source_list.content)
            .expect("source list should parse");
    shopping_sources.sort();
    assert_eq!(
        shopping_sources,
        vec![
            ("Memory".to_string(), "grocery purchases".to_string()),
            (
                "Memory".to_string(),
                "price comparison follow up".to_string()
            ),
            ("Memory".to_string(), "shopping trip note".to_string()),
        ]
    );
    let shopping_trip_note_index =
        serde_json::from_str::<Vec<(String, String)>>(&shopping_source_list.content)
            .expect("source list should parse")
            .iter()
            .position(|entry| entry == &("Memory".to_string(), "shopping trip note".to_string()))
            .expect("shopping trip note source should be present");
    let shopping_source_content = registry.invoke(
            "get_source_content_for_memory",
            &format!(
                "{{\"memory_name\":\"shopping trip on 2024 01 01\",\"index\":{shopping_trip_note_index}}}"
            ),
        );
    assert!(!shopping_source_content.is_error);
    assert!(
        shopping_source_content
            .content
            .contains("#shopping trip note")
    );
    assert!(
        shopping_source_content
            .content
            .contains("I went to the store today, January 1")
    );

    let follow_up_source_list = registry.invoke(
        "get_source_list_for_memory",
        "{\"memory_name\":\"user s grocery price comparison follow up\"}",
    );
    assert!(!follow_up_source_list.is_error);
    let mut follow_up_sources: Vec<(String, String)> =
        serde_json::from_str::<Vec<(String, String)>>(&follow_up_source_list.content)
            .expect("source list should parse");
    follow_up_sources.sort();
    assert_eq!(follow_up_sources, shopping_sources);

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn create_memory_tool_limits_semantic_consolidation_to_top_three_clusters_per_run() {
    let unique = format!(
        "elroy-rs-app-semantic-memory-cluster-limit-{}",
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
    for (text, embedding) in [
        ("alpha trip detail one", "[1.0, 0.0]"),
        ("alpha trip detail two", "[0.999, 0.001]"),
        ("alpha trip detail three", "[0.998, 0.002]"),
        ("beta recipe detail one", "[0.0, 1.0]"),
        ("beta recipe detail two", "[0.001, 0.999]"),
        ("beta recipe detail three", "[0.002, 0.998]"),
        ("gamma project detail one", "[-1.0, 0.0]"),
        ("gamma project detail two", "[-0.999, 0.001]"),
        ("gamma project detail three", "[-0.998, 0.002]"),
        ("delta pricing detail one", "[0.0, -1.0]"),
        ("delta pricing detail two", "[0.12, -0.993]"),
        ("delta pricing detail three", "[0.24, -0.971]"),
    ] {
        server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(text.to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(r#"{{"data":[{{"embedding":{embedding}}}]}}"#))
            .create();
    }

    let _alpha_consolidation = server
            .mock("POST", "/responses")
            .match_body(mockito::Matcher::Regex("alpha trip one".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "# Memory Consolidation Reasoning\nMerged the alpha trip notes.\n\n## alpha consolidated memory\nThe alpha trip details belong in one memory."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let _beta_consolidation = server
            .mock("POST", "/responses")
            .match_body(mockito::Matcher::Regex("beta recipe one".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "# Memory Consolidation Reasoning\nMerged the beta recipe notes.\n\n## beta consolidated memory\nThe beta recipe details belong in one memory."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let _gamma_consolidation = server
            .mock("POST", "/responses")
            .match_body(mockito::Matcher::Regex("gamma project one".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "# Memory Consolidation Reasoning\nMerged the gamma project notes.\n\n## gamma consolidated memory\nThe gamma project details belong in one memory."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memories_between_consolidation = 12;
    config.min_memory_cluster_size = 3;
    config.max_memory_cluster_size = 5;
    config.memory_cluster_similarity_threshold = 0.05;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    config.embedding_model = "text-embedding-3-small".to_string();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

    let registry = build_live_tool_registry(&config);
    for (name, text) in [
        ("alpha trip one", "alpha trip detail one"),
        ("alpha trip two", "alpha trip detail two"),
        ("alpha trip three", "alpha trip detail three"),
        ("beta recipe one", "beta recipe detail one"),
        ("beta recipe two", "beta recipe detail two"),
        ("beta recipe three", "beta recipe detail three"),
        ("gamma project one", "gamma project detail one"),
        ("gamma project two", "gamma project detail two"),
        ("gamma project three", "gamma project detail three"),
        ("delta pricing one", "delta pricing detail one"),
        ("delta pricing two", "delta pricing detail two"),
        ("delta pricing three", "delta pricing detail three"),
    ] {
        let result = registry.invoke(
            "create_memory",
            &format!("{{\"name\":\"{name}\",\"text\":\"{text}\"}}"),
        );
        assert!(!result.is_error, "{name} should be created");
    }

    let connection = open_sqlite_connection(&database_path).expect("database should reopen");
    let active_memories =
        elroy_db::list_active_memories(&connection, 20).expect("active memories should list");
    let active_names = active_memories
        .iter()
        .map(|memory| memory.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(active_memories.len(), 6);
    assert!(active_names.contains(&"alpha consolidated memory"));
    assert!(active_names.contains(&"beta consolidated memory"));
    assert!(active_names.contains(&"gamma consolidated memory"));
    assert!(active_names.contains(&"delta pricing one"));
    assert!(active_names.contains(&"delta pricing two"));
    assert!(active_names.contains(&"delta pricing three"));
    assert!(!active_names.contains(&"alpha trip one"));
    assert!(!active_names.contains(&"beta recipe one"));
    assert!(!active_names.contains(&"gamma project one"));

    let tracker = load_memory_operation_tracker(&connection, LOCAL_USER_TOKEN)
        .expect("tracker should load")
        .expect("tracker should exist");
    assert_eq!(tracker.memories_since_consolidation, 0);

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn create_memory_tool_limits_each_semantic_cluster_to_densest_members() {
    let unique = format!(
        "elroy-rs-app-semantic-memory-max-cluster-size-{}",
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
    for (text, embedding) in [
        ("cluster core one", "[1.0, 0.0]"),
        ("cluster core two", "[0.999, 0.001]"),
        ("cluster core three", "[0.998, 0.002]"),
        ("cluster fringe note", "[0.97, 0.243]"),
    ] {
        server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(text.to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(r#"{{"data":[{{"embedding":{embedding}}}]}}"#))
            .create();
    }

    let _consolidation_mock = server
            .mock("POST", "/responses")
            .match_body(mockito::Matcher::Regex("cluster core one".to_string()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "# Memory Consolidation Reasoning\nI merged the three densest notes and left the looser fringe note for later.\n\n## cluster consolidated memory\nThe three core notes describe the same trip detail and belong in one memory."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memories_between_consolidation = 4;
    config.min_memory_cluster_size = 3;
    config.max_memory_cluster_size = 3;
    config.memory_cluster_similarity_threshold = 0.05;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    config.embedding_model = "text-embedding-3-small".to_string();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

    let registry = build_live_tool_registry(&config);
    for (name, text) in [
        ("Cluster core one", "cluster core one"),
        ("Cluster core two", "cluster core two"),
        ("Cluster core three", "cluster core three"),
        ("Cluster fringe note", "cluster fringe note"),
    ] {
        let result = registry.invoke(
            "create_memory",
            &format!("{{\"name\":\"{name}\",\"text\":\"{text}\"}}"),
        );
        assert!(!result.is_error, "{name} should be created");
    }

    let connection = open_sqlite_connection(&database_path).expect("database should reopen");
    let active_memories =
        elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
    assert_eq!(active_memories.len(), 2);
    assert!(
        active_memories
            .iter()
            .any(|memory| memory.name == "cluster consolidated memory")
    );
    assert!(
        active_memories
            .iter()
            .any(|memory| memory.name == "cluster fringe note")
    );

    let source_list = registry.invoke(
        "get_source_list_for_memory",
        "{\"memory_name\":\"cluster consolidated memory\"}",
    );
    assert!(!source_list.is_error);
    let mut source_entries: Vec<(String, String)> =
        serde_json::from_str::<Vec<(String, String)>>(&source_list.content)
            .expect("source list should parse");
    source_entries.sort();
    assert_eq!(
        source_entries,
        vec![
            ("Memory".to_string(), "cluster core one".to_string()),
            ("Memory".to_string(), "cluster core three".to_string()),
            ("Memory".to_string(), "cluster core two".to_string()),
        ]
    );

    assert!(
        memory_dir
            .join("archive")
            .join("cluster_core_one.md")
            .exists()
    );
    assert!(
        memory_dir
            .join("archive")
            .join("cluster_core_two.md")
            .exists()
    );
    assert!(
        memory_dir
            .join("archive")
            .join("cluster_core_three.md")
            .exists()
    );
    assert!(
        !memory_dir
            .join("archive")
            .join("cluster_fringe_note.md")
            .exists()
    );
    assert!(memory_dir.join("cluster_fringe_note.md").exists());

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn threshold_consolidation_removes_archived_source_memories_from_current_context() {
    let unique = format!(
        "elroy-rs-app-memory-consolidation-drops-context-{}",
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
    config.database_path = database_path;
    config.memories_between_consolidation = 3;
    elroy_db::bootstrap_database(&elroy_db::BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    for (name, text) in [
        ("Trip Note One", "I went shopping on New Year's Day."),
        ("Trip Note Two", "I went shopping on New Year's Day."),
    ] {
        let created = registry.invoke(
            "create_memory",
            &format!("{{\"name\":\"{name}\",\"text\":\"{text}\"}}"),
        );
        assert!(!created.is_error, "{name} should be created");
        let added = registry.invoke(
            "add_memory_to_current_context",
            &format!("{{\"memory_name\":\"{}\"}}", name.to_ascii_lowercase()),
        );
        assert!(!added.is_error, "{name} should be pinned");
    }

    let third = registry.invoke(
        "create_memory",
        "{\"name\":\"Trip Note Three\",\"text\":\"I went shopping on New Year's Day.\"}",
    );
    assert!(!third.is_error);

    let context = registry.invoke("show_context_messages", "{\"limit\":20}");
    assert!(!context.is_error);
    assert!(!context.content.contains("context-memory:trip note one"));
    assert!(!context.content.contains("context-memory:trip note two"));
    assert!(!context.content.contains("Trip Note One"));
    assert!(!context.content.contains("Trip Note Two"));

    let listed = registry.invoke("print_memories", "{\"n\":10}");
    assert!(!listed.is_error);
    assert!(listed.content.contains("trip note one"));
    assert!(!listed.content.contains("trip note two"));
    assert!(!listed.content.contains("trip note three"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn consolidate_memory_cluster_uses_python_style_prompt_contract() {
    struct ConsolidationPromptInspectionModel;

    impl ModelClient for ConsolidationPromptInspectionModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            assert!(prompt.contains("# Memory Consolidation Task"), "{prompt}");
            assert!(prompt.contains("## Dates and times"), "{prompt}");
            assert!(prompt.contains("Use ISO 8601 format"), "{prompt}");
            assert!(prompt.contains("## Synthesis Guidelines"), "{prompt}");
            assert!(
                    prompt.contains("Resolve contradictions instead of carrying conflicting claims forward unchanged."),
                    "{prompt}"
                );
            assert!(
                prompt.contains("prefer the most recent dated information."),
                "{prompt}"
            );
            assert!(
                prompt.contains("Call out recurring patterns or repeated events"),
                "{prompt}"
            );
            assert!(
                prompt.contains("Limit each new memory excerpt to 300 words."),
                "{prompt}"
            );
            assert!(prompt.contains("## Memory Title Guidelines"), "{prompt}");
            assert!(
                prompt.contains("# Memory Consolidation Reasoning"),
                "{prompt}"
            );
            assert!(prompt.contains("# Memory Consolidation Input"), "{prompt}");
            assert!(prompt.contains("## shopping trip note"), "{prompt}");
            assert!(
                prompt.contains("I went shopping at the store on New Year's Day"),
                "{prompt}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                    content: "# Memory Consolidation Reasoning\nI merged the two memories because they describe the same shopping trip.\n\n## Shopping trip on 2024-01-01\nUser went to the store on New Year's Day and bought some items."
                        .to_string(),
                }])
        }
    }

    let memories = vec![
        MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "shopping trip note".to_string(),
            file_path: "/tmp/shopping_trip_note.md".to_string(),
            body: "I went to the store today, January 1".to_string(),
            is_active: true,
            updated_at_unix: 1,
        },
        MemoryRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "new year s shopping".to_string(),
            file_path: "/tmp/new_year_s_shopping.md".to_string(),
            body: "I went shopping at the store on New Year's Day".to_string(),
            is_active: true,
            updated_at_unix: 2,
        },
    ];

    let outputs = crate::consolidate_memory_cluster_outputs(
        &memories,
        Some(&ConsolidationPromptInspectionModel),
    );
    assert_eq!(
        outputs,
        vec![crate::ConsolidatedMemoryOutput {
            name: "Shopping trip on 2024-01-01".to_string(),
            text: "User went to the store on New Year's Day and bought some items.".to_string(),
        }]
    );
}

#[test]
fn parse_consolidated_memory_response_accepts_python_style_markdown_multiple_outputs() {
    let outputs = crate::parse_consolidated_memory_response(
        "# Memory Consolidation Reasoning\n\
I split the overlapping memories into two focused follow-ups.\n\n\
## Shopping trip on 2024-01-01\n\
User went to the store on New Year's Day and bought some items.\n\n\
## User's grocery price comparison follow-up\n\
User still wants to compare grocery prices after the shopping trip.",
    )
    .expect("markdown response should parse");

    assert_eq!(
        outputs,
        vec![
            crate::ConsolidatedMemoryOutput {
                name: "Shopping trip on 2024-01-01".to_string(),
                text: "User went to the store on New Year's Day and bought some items.".to_string(),
            },
            crate::ConsolidatedMemoryOutput {
                name: "User's grocery price comparison follow-up".to_string(),
                text: "User still wants to compare grocery prices after the shopping trip."
                    .to_string(),
            },
        ]
    );
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
fn semantic_consolidation_source_fetch_includes_memories_beyond_old_500_cap() {
    let unique = format!(
        "elroy-rs-app-memory-consolidation-source-fetch-{}",
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

    for index in 0..503 {
        fs::write(
            memory_dir.join(format!("memory_{index:03}.md")),
            format!("Memory body {index}\n"),
        )
        .expect("memory file should be written");
    }

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    let fetched = crate::list_all_active_memories_in_scope(&connection, &memory_dir)
        .expect("all active memories in scope should load");

    assert_eq!(fetched.len(), 503);
    assert!(
        fetched.iter().any(|memory| memory.name == "memory 502"),
        "the consolidation source fetch should include memories beyond the old 500-row preload"
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn semantic_consolidation_can_reuse_persisted_embedding_cache() {
    let unique = format!(
        "elroy-rs-app-memory-consolidation-embedding-cache-{}",
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
        memory_dir.join("shopping_trip_note.md"),
        "I went to the store today, January 1\n",
    )
    .expect("first memory should be written");
    fs::write(
        memory_dir.join("new_year_s_shopping.md"),
        "I went shopping at the store on New Year's Day\n",
    )
    .expect("second memory should be written");
    fs::write(
        memory_dir.join("store_purchases.md"),
        "Today, New Year's Day, I bought some items at the store\n",
    )
    .expect("third memory should be written");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.min_memory_cluster_size = 3;
    config.max_memory_cluster_size = 5;
    config.memory_cluster_similarity_threshold = 0.1;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.embedding_model = "text-embedding-3-small".to_string();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());

    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let memories = elroy_recall::list_all_active_memories_in_scope(&connection, &memory_dir)
        .expect("memories should load");
    assert_eq!(memories.len(), 3);

    for memory in &memories {
        let embedding = match memory.name.as_str() {
            "shopping trip note" => vec![1.0, 0.0],
            "new year s shopping" => vec![0.99, 0.01],
            "store purchases" => vec![0.98, 0.02],
            other => panic!("unexpected memory name: {other}"),
        };
        upsert_memory_embedding(
            &connection,
            &memory.file_path,
            &embedding,
            &crate::memory_embedding_text(memory),
        )
        .expect("embedding cache should persist");
    }

    let mut server = mockito::Server::new();
    let _consolidation_mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"memories":[{"name":"Shopping trip on 2024-01-01","text":"User went to the store on New Year's Day and bought some items."}]}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));

    crate::consolidate_semantic_memory_clusters(
        &mut connection,
        &BootstrapPlan::from_config(&config),
        &crate::memory_consolidation_settings_from_app_config(&config),
    )
    .expect("semantic consolidation should succeed from cached embeddings");

    let reopened = open_sqlite_connection(&database_path).expect("database should reopen");
    let active_memories =
        elroy_db::list_active_memories(&reopened, 10).expect("active memories should list");
    assert_eq!(active_memories.len(), 1);
    assert_eq!(active_memories[0].name, "shopping trip on 2024 01 01");

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
    let missing_other = registry.invoke("show_memory", "{\"memory_name\":\"Other Only Memory\"}");

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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
fn load_snapshot_formats_persisted_tool_messages_and_skips_system_lines() {
    let unique = format!(
        "elroy-rs-app-load-snapshot-tool-lines-{}",
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
            ConversationMessage::new(MessageRole::System, "system prompt"),
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::new(
                MessageRole::Assistant,
                "<internal_thought>Need to think</internal_thought>Visible answer",
            ),
            ConversationMessage::new(
                MessageRole::Assistant,
                "<internal_thought>hidden only</internal_thought>",
            ),
            ConversationMessage::new(MessageRole::Tool, "stored tool output"),
        ],
    )
    .expect("messages should persist");
    drop(connection);

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;

    let runtime = AppRuntime::new(config);
    let snapshot = runtime.load_snapshot().expect("snapshot should load");

    assert_eq!(
        snapshot.conversation_lines,
        vec![
            "user: hello".to_string(),
            "assistant: Visible answer".to_string(),
            "tool result: stored tool output".to_string(),
        ]
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn load_snapshot_can_render_internal_thought_segments_when_enabled() {
    let unique = format!(
        "elroy-rs-app-load-snapshot-internal-thought-lines-{}",
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
            "<internal_thought>Need to think</internal_thought>Visible answer",
        )],
    )
    .expect("messages should persist");
    drop(connection);

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.show_internal_thought = true;

    let runtime = AppRuntime::new(config);
    let snapshot = runtime.load_snapshot().expect("snapshot should load");

    assert_eq!(
        snapshot.conversation_lines,
        vec![
            "thinking: Need to think".to_string(),
            "assistant: Visible answer".to_string(),
        ]
    );
    assert!(snapshot.show_internal_thought);

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
            source: TuiCommandSource::Slash,
        }
    );
    let help_snapshot = runtime
        .execute_command(
            &help_execution.command_name,
            &help_execution.display_name,
            &help_execution.values,
            &help_execution.source,
        )
        .expect("help command should execute");
    assert_eq!(help_snapshot.status, None);
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
            source: TuiCommandSource::Slash,
        }
    );
    let shown_snapshot = runtime
        .execute_command(
            &shown_execution.command_name,
            &shown_execution.display_name,
            &shown_execution.values,
            &shown_execution.source,
        )
        .expect("show_memory command should execute");
    assert_eq!(shown_snapshot.status, None);
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
    assert_eq!(missing_form.source, TuiCommandSource::Slash);

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
    assert_eq!(prefilled_form.source, TuiCommandSource::Slash);
    assert!(
        prefilled_form
            .parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .eq(["name", "text"])
    );
    assert_eq!(
        runtime
            .handle_slash_command("/missing_command")
            .expect("unknown slash command should fall back to plain chat"),
        TuiSlashCommandAction::NotHandled
    );
    assert_eq!(
        runtime
            .handle_slash_command("/")
            .expect("empty slash command should fall back to plain chat"),
        TuiSlashCommandAction::NotHandled
    );

    let submitted_snapshot = runtime
        .execute_command(
            "create_memory",
            "create_memory",
            &[
                ("name".to_string(), "trip".to_string()),
                ("text".to_string(), "Aisle seats.".to_string()),
            ],
            &TuiCommandSource::Slash,
        )
        .expect("command form submit should execute");
    assert_eq!(submitted_snapshot.status, None);
    assert!(
        submitted_snapshot
            .conversation_lines
            .last()
            .is_some_and(|line| line == "tool result: New memory created: trip")
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
    assert_eq!(form.source, TuiCommandSource::Palette);
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
    assert_eq!(show_task_form.source, TuiCommandSource::Palette);
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
            source: TuiCommandSource::Palette,
        }
    );
    let snapshot = runtime
        .execute_command(
            &execution.command_name,
            &execution.display_name,
            &execution.values,
            &execution.source,
        )
        .expect("get_help command should execute from the palette path");
    assert_eq!(snapshot.status, None);
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
            &TuiCommandSource::Palette,
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
fn live_tool_registry_can_dispatch_and_resume_codex_sessions() {
    let status_key = codex_background_status_key("thread-123");
    clear_background_status(&status_key);
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
        get_background_status_for_key(&status_key).as_deref(),
        Some("codex session thread-123 running...")
    );
    wait_for_codex_status(&database_path, "thread-123", "completed");
    wait_for_background_status_key_message(
        &status_key,
        "processing codex session thread-123 completion...",
    );
    thread::sleep(Duration::from_millis(250));
    assert!(get_background_status_for_key(&status_key).is_none());

    let resumed = registry.invoke(
        "resume_codex_session",
        "{\"session_id\":\"thread-123\",\"prompt\":\"follow up\"}",
    );
    assert!(!resumed.is_error);
    assert!(resumed.content.contains("\"status\":\"running\""));
    assert_eq!(
        get_background_status_for_key(&status_key).as_deref(),
        Some("codex session thread-123 running...")
    );
    wait_for_codex_status(&database_path, "thread-123", "completed");
    wait_for_background_status_key_message(
        &status_key,
        "processing codex session thread-123 completion...",
    );
    thread::sleep(Duration::from_millis(250));
    assert!(get_background_status_for_key(&status_key).is_none());

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
fn live_tool_registry_search_memories_can_prefer_semantic_memory_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-search-memories-semantic-priority-{}",
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
        memory_dir.join("gear_inventory.md"),
        "Review the storage locker spreadsheet.\n",
    )
    .expect("overlap memory should be written");
    fs::write(
        memory_dir.join("bands_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("semantic memory should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands memory matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("Memory | bands note | Pack resistance bands before drills.")
    );
    assert!(!search.content.contains("Memory | gear inventory |"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_use_fast_model_config_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-search-memories-fast-model-{}",
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
        memory_dir.join("gear_inventory.md"),
        "Review the storage locker spreadsheet.\n",
    )
    .expect("overlap memory should be written");
    fs::write(
        memory_dir.join("bands_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("semantic memory should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands memory matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("Memory | bands note | Pack resistance bands before drills.")
    );
    assert!(!search.content.contains("Memory | gear inventory |"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_prefer_semantic_due_item_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-search-memories-semantic-due-item-{}",
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
            agenda_dir.join("gear_inventory_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap due item should be written");
    fs::write(
            agenda_dir.join("bands_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before scrimmage\n---\n\nCarry resistance bands in the trunk.\n",
        )
        .expect("semantic due item should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands reminder matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("DueItem | bands reminder | Carry resistance bands in the trunk.")
    );
    assert!(
        !search
            .content
            .contains("DueItem | gear inventory reminder |")
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_surface_older_semantic_due_item_beyond_old_fetch_cap() {
    let unique = format!(
        "elroy-rs-app-search-memories-older-semantic-due-item-{}",
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

    for index in 0..7 {
        fs::write(
                agenda_dir.join(format!("recent_reminder_{index}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet {index}.\n"
                ),
            )
            .expect("recent due item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic due item should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,false,false,false,false,false,false,true],"reasoning":"Only the older training reminder matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..7 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_reminder_{index}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent due item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_reminder.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older due item timestamp should update");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("DueItem | old training reminder | Pack resistance bands before drills.")
    );
    assert!(!search.content.contains("DueItem | recent reminder 0 |"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_surface_older_semantic_due_item_via_embedding_without_relevance_model()
 {
    let unique = format!(
        "elroy-rs-app-search-memories-older-semantic-due-item-embedding-only-{}",
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

    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_reminder_{index:03}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet {index}.\n"
                ),
            )
            .expect("recent due item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic due item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_reminder_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent due item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_reminder.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older due item timestamp should update");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("DueItem | old training reminder | Pack resistance bands before drills.")
    );
    assert!(!search.content.contains("DueItem | recent reminder 000 |"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_surface_older_semantic_memory_via_embedding_candidates() {
    let unique = format!(
        "elroy-rs-app-search-memories-older-semantic-memory-embedding-{}",
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

    for index in 0..120 {
        fs::write(
            memory_dir.join(format!("recent_note_{index:03}.md")),
            format!("Review the storage locker spreadsheet {index}.\n"),
        )
        .expect("recent memory should be written");
    }
    fs::write(
        memory_dir.join("old_training_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut server = mockito::Server::new();
    let mut answers = vec![false; 100];
    answers[0] = true;
    let _relevance_mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": serde_json::json!({
                                "answers": answers,
                                "reasoning": "Only the older training-kit memory matches the user's intent."
                            }).to_string()
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let _query_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    memory_dir
                        .join(format!("recent_note_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent memory timestamp should update");
    }
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                memory_dir
                    .join("old_training_note.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older memory timestamp should update");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("Memory | old training note | Pack resistance bands before drills.")
    );
    assert!(
        !search.content.contains("recent note 000"),
        "{}",
        search.content
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_surface_older_semantic_memory_via_embedding_without_relevance_model()
 {
    let unique = format!(
        "elroy-rs-app-search-memories-older-semantic-memory-embedding-only-{}",
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

    for index in 0..120 {
        fs::write(
            memory_dir.join(format!("recent_note_{index:03}.md")),
            format!("Review the storage locker spreadsheet {index}.\n"),
        )
        .expect("recent memory should be written");
    }
    fs::write(
        memory_dir.join("old_training_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut server = mockito::Server::new();
    let _query_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    memory_dir
                        .join(format!("recent_note_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent memory timestamp should update");
    }
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                memory_dir
                    .join("old_training_note.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older memory timestamp should update");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("Memory | old training note | Pack resistance bands before drills.")
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_reuse_persisted_embedding_cache() {
    let unique = format!(
        "elroy-rs-app-search-memories-persisted-embedding-cache-{}",
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
        memory_dir.join("recent_planning_note.md"),
        "Review the storage locker spreadsheet.\n",
    )
    .expect("recent memory should be written");
    fs::write(
        memory_dir.join("recent_inventory_note.md"),
        "Double-check the equipment inventory list.\n",
    )
    .expect("second recent memory should be written");
    fs::write(
        memory_dir.join("old_training_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut server = mockito::Server::new();
    let query_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .expect(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    let memories = crate::list_active_memories_in_scope(&connection, &memory_dir, 10)
        .expect("memories should load");
    for memory in &memories {
        let embedding = match memory.name.as_str() {
            "old training note" => vec![1.0, 0.0],
            "recent planning note" => vec![0.0, 1.0],
            "recent inventory note" => vec![0.0, 0.95],
            other => panic!("unexpected memory name: {other}"),
        };
        upsert_memory_embedding(
            &connection,
            &memory.file_path,
            &embedding,
            &crate::memory_embedding_text(memory),
        )
        .expect("embedding cache should persist");
    }

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("Memory | old training note | Pack resistance bands before drills.")
    );
    assert!(
        !search.content.contains("recent planning note"),
        "{}",
        search.content
    );

    query_embedding_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_prefer_recent_semantic_memories_when_recency_weight_is_enabled()
 {
    let unique = format!(
        "elroy-rs-app-search-memories-recency-weight-{}",
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
        memory_dir.join("stale_exact_match.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("stale semantic memory should be written");
    fs::write(
        memory_dir.join("recent_warmup_note.md"),
        "Bring resistance bands for warmups.\n",
    )
    .expect("recent warmup memory should be written");
    fs::write(
        memory_dir.join("recent_practice_note.md"),
        "Bring resistance bands and cones to practice.\n",
    )
    .expect("recent practice memory should be written");

    let mut server = mockito::Server::new();
    let _query_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _stale_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_warmup_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Bring resistance bands for warmups".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.92, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_practice_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Bring resistance bands and cones to practice".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.93, 0.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.recency_weight = 0.1;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let now = Utc::now().timestamp();
    let connection = open_sqlite_connection(&database_path).expect("database should open");
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
            rusqlite::params![
                now - (5 * 365 * 24 * 60 * 60),
                memory_dir
                    .join("stale_exact_match.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("stale memory timestamp should update");
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
            rusqlite::params![
                now - (2 * 24 * 60 * 60),
                memory_dir
                    .join("recent_warmup_note.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("recent warmup timestamp should update");
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
            rusqlite::params![
                now - (24 * 60 * 60),
                memory_dir
                    .join("recent_practice_note.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("recent practice timestamp should update");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":2}",
    );

    assert!(!search.is_error);
    assert!(
        search.content.contains(
            "Memory | recent practice note | Bring resistance bands and cones to practice."
        )
    );
    assert!(
        search
            .content
            .contains("Memory | recent warmup note | Bring resistance bands for warmups.")
    );
    assert!(!search.content.contains("Memory | stale exact match |"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_prefer_semantic_agenda_item_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-search-memories-semantic-agenda-item-{}",
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
            agenda_dir.join("gear_inventory_plan.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap agenda item should be written");
    fs::write(
            agenda_dir.join("bands_plan.md"),
            "---\ndate: 2026-05-21\ncompleted: false\nstatus: created\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("semantic agenda item should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands agenda item matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let search = registry.invoke(
        "search_memories",
        "{\"query\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("AgendaItem | bands plan | Pack resistance bands before drills.")
    );
    assert!(
        !search
            .content
            .contains("AgendaItem | gear inventory plan |")
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_search_memories_can_surface_older_semantic_agenda_item_via_embedding_without_relevance_model()
 {
    let unique = format!(
        "elroy-rs-app-search-memories-older-semantic-agenda-item-embedding-only-{}",
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

    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_plan_{index:03}.md")),
                format!(
                    "---\ndate: 2026-05-{day:02}\ncompleted: false\nstatus: created\n---\n\nReview the storage locker spreadsheet {index}.\n",
                    day = (index % 28) + 1,
                ),
            )
            .expect("recent agenda item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_plan.md"),
            "---\ndate: 2026-04-01\ncompleted: false\nstatus: created\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic agenda item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_plan_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent agenda item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_plan.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older agenda item timestamp should update");

    let search = build_live_tool_registry(&config).invoke(
        "search_memories",
        "{\"query\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!search.is_error);
    assert!(
        search
            .content
            .contains("AgendaItem | old training plan | Pack resistance bands before drills.")
    );
    assert!(!search.content.contains("AgendaItem | recent plan 000 |"));

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
fn live_tool_registry_examine_memories_can_prefer_semantic_memory_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-examine-memories-semantic-priority-{}",
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
        memory_dir.join("gear_inventory.md"),
        "Review the storage locker spreadsheet.\n",
    )
    .expect("overlap memory should be written");
    fs::write(
        memory_dir.join("bands_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("semantic memory should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands memory matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let result = registry.invoke(
        "examine_memories",
        "{\"question\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!result.is_error);
    assert!(result.content.contains("# Memory: bands note"));
    assert!(
        result
            .content
            .contains("Pack resistance bands before drills.")
    );
    assert!(!result.content.contains("# Memory: gear inventory"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_examine_memories_can_prefer_semantic_due_item_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-examine-memories-semantic-due-item-{}",
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
            agenda_dir.join("gear_inventory_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap due item should be written");
    fs::write(
            agenda_dir.join("bands_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before scrimmage\n---\n\nCarry resistance bands in the trunk.\n",
        )
        .expect("semantic due item should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands reminder matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let result = registry.invoke(
        "examine_memories",
        "{\"question\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!result.is_error);
    assert!(result.content.contains("# Due Item: bands reminder"));
    assert!(
        result
            .content
            .contains("Carry resistance bands in the trunk.")
    );
    assert!(
        !result
            .content
            .contains("# Due Item: gear inventory reminder")
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_examine_memories_can_surface_older_semantic_due_item_via_embedding_without_relevance_model()
 {
    let unique = format!(
        "elroy-rs-app-examine-memories-older-semantic-due-item-embedding-only-{}",
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

    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_reminder_{index:03}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet {index}.\n"
                ),
            )
            .expect("recent due item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic due item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_reminder_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent due item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_reminder.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older due item timestamp should update");

    let registry = build_live_tool_registry(&config);
    let result = registry.invoke(
        "examine_memories",
        "{\"question\":\"What belongs in my workout kit?\",\"limit\":5}",
    );

    assert!(!result.is_error);
    assert!(result.content.contains("# Due Item: old training reminder"));
    assert!(
        result
            .content
            .contains("Pack resistance bands before drills.")
    );
    assert!(!result.content.contains("# Due Item: recent reminder 000"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn live_tool_registry_examine_memories_can_prefer_semantic_agenda_item_over_weaker_overlap() {
    let unique = format!(
        "elroy-rs-app-examine-memories-semantic-agenda-item-{}",
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
            agenda_dir.join("gear_inventory_plan.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap agenda item should be written");
    fs::write(
            agenda_dir.join("bands_plan.md"),
            "---\ndate: 2026-05-21\ncompleted: false\nstatus: created\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("semantic agenda item should be written");

    let mut server = mockito::Server::new();
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false,true],"reasoning":"Only the resistance-bands agenda item matches the user's intent."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let registry = build_live_tool_registry(&config);
    let result = registry.invoke(
        "examine_memories",
        "{\"question\":\"What gear should I bring?\",\"limit\":5}",
    );

    assert!(!result.is_error);
    assert!(result.content.contains("# Agenda Item: bands plan"));
    assert!(
        result
            .content
            .contains("Pack resistance bands before drills.")
    );
    assert!(
        !result
            .content
            .contains("# Agenda Item: gear inventory plan")
    );

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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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

    let active_due_items = list_active_due_items(&connection, 10).expect("due items should list");
    assert!(
        !active_due_items
            .iter()
            .any(|item| item.name == "medicine reminder")
    );
    assert!(!agenda_dir.join("medicine_reminder.md").exists());

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_surfaces_and_cleans_up_due_tasks() {
    let unique = format!(
        "elroy-rs-app-due-task-prompt-integration-{}",
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
    create_task_file_with_schedule(
        &agenda_dir,
        "medicine reminder",
        "Take your daily medicine",
        None,
        Some("2000-01-01T09:00:00"),
        None,
    )
    .expect("due task file should be written");

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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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

    let active_due_items = list_active_due_items(&connection, 10).expect("due items should list");
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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

    let active_due_items = list_active_due_items(&connection, 10).expect("due items should list");
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
    let tracker = elroy_db::get_or_create_memory_operation_tracker(&mut reopened, LOCAL_USER_TOKEN)
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
fn run_prompt_with_model_and_registry_does_not_duplicate_hybrid_due_item_via_contextual_path() {
    struct HybridDueItemDedupModel;

    impl ModelClient for HybridDueItemDedupModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "Can we talk about work?");
            let tool_messages = request
                .transcript
                .iter()
                .filter_map(|message| {
                    (message.role == MessageRole::Tool)
                        .then_some(message.content.as_deref())
                        .flatten()
                })
                .collect::<Vec<_>>();
            let hybrid_mentions = tool_messages
                .iter()
                .filter(|content| content.contains("Hybrid reminder text"))
                .count();
            assert_eq!(hybrid_mentions, 1, "{tool_messages:#?}");
            Ok(vec![StreamEvent::AssistantResponse {
                content: "A hybrid reminder is due: Hybrid reminder text.".to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-hybrid-due-item-dedup-{}",
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
        "Can we talk about work?",
        &HybridDueItemDedupModel,
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
    struct NoRecallClassifierModel;

    impl ModelClient for NoRecallClassifierModel {
        fn next_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            Ok(vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"needs_recall":false,"reasoning":"Use the contextual reminder path only."}"#
                        .to_string(),
            }])
        }
    }

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

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "I just got the payroll email.",
        &ContextualDueItemModel,
        recall_model_clients(Some(&NoRecallClassifierModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            if content.to_ascii_lowercase().contains("reply to payroll")
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

    let second_events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "I'm following up after that payroll email now.",
        &ContextualDueItemModel,
        recall_model_clients(Some(&NoRecallClassifierModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
fn run_prompt_with_model_and_registry_can_broaden_contextual_due_item_surfacing() {
    struct SemanticContextualDueItemModel;

    impl ModelClient for SemanticContextualDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            if request
                .transcript
                .iter()
                .any(|message| message.role == MessageRole::Tool)
            {
                assert_eq!(request.user_message, "What gear should I bring?");
                assert!(request.transcript.iter().any(|message| {
                    message.role == MessageRole::Tool
                        && message.content.as_deref().is_some_and(|content| {
                            let normalized = content.to_ascii_lowercase();
                            normalized.contains("practice reminder")
                                && normalized.contains("bring the resistance bands")
                                && normalized.contains("before basketball practice")
                        })
                }));
                return Ok(vec![StreamEvent::AssistantResponse {
                    content: "You should bring the resistance bands.".to_string(),
                }]);
            }

            Ok(vec![StreamEvent::AssistantResponse {
                    content: r#"{"answers":[true],"reasoning":"This reminder is relevant even though the wording differs."}"#
                        .to_string(),
                }])
        }
    }

    let unique = format!(
        "elroy-rs-app-contextual-due-item-semantic-prompt-{}",
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
            agenda_dir.join("practice_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nBring the resistance bands\n",
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

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What gear should I bring?",
        &SemanticContextualDueItemModel,
        recall_model_clients(Some(&SemanticContextualDueItemModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_can_prefer_semantic_contextual_due_item_over_weaker_overlap()
{
    struct SemanticPriorityContextualDueItemModel;
    struct SemanticReminderRelevanceModel;

    impl ModelClient for SemanticPriorityContextualDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What gear should I bring?");
            let tool_content = request
                .transcript
                .iter()
                .filter(|message| message.role == MessageRole::Tool)
                .filter_map(|message| message.content.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                tool_content.contains("practice reminder")
                    && tool_content.contains("Bring the resistance bands"),
                "{tool_content}"
            );
            assert!(!tool_content.contains("Gear Inventory"), "{tool_content}");
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You should bring the resistance bands.".to_string(),
            }])
        }
    }

    impl ModelClient for SemanticReminderRelevanceModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            let answers = if prompt.contains("0. # Practice Reminder") {
                vec![true, false]
            } else {
                vec![false, true]
            };
            Ok(vec![StreamEvent::AssistantResponse {
                content: serde_json::json!({
                    "answers": answers,
                    "reasoning": "Only the resistance-bands reminder matches the user's intent."
                })
                .to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-contextual-due-item-semantic-priority-{}",
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
            agenda_dir.join("gear_inventory.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap due item file should be written");
    fs::write(
            agenda_dir.join("practice_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nBring the resistance bands\n",
        )
        .expect("semantic due item file should be written");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What gear should I bring?",
        &SemanticPriorityContextualDueItemModel,
        recall_model_clients(Some(&SemanticReminderRelevanceModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_does_not_duplicate_contextual_due_items_already_in_fast_recall()
 {
    struct NonDuplicatingDueItemModel;

    impl ModelClient for NonDuplicatingDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "I just got the payroll email");
            let due_item_tool_mentions = request
                .transcript
                .iter()
                .filter(|message| message.role == MessageRole::Tool)
                .filter(|message| {
                    message.content.as_deref().is_some_and(|content| {
                        content.to_ascii_lowercase().contains("payroll follow up")
                    })
                })
                .count();
            assert_eq!(
                due_item_tool_mentions, 1,
                "same due item should not be surfaced by both fast recall and contextual due-item injection in one turn"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You should reply to payroll.".to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-contextual-due-item-no-duplicate-{}",
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

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "I just got the payroll email",
        &NonDuplicatingDueItemModel,
        recall_model_clients(None),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.to_ascii_lowercase().contains("reply to payroll")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_does_not_duplicate_contextual_due_items_already_pinned_in_current_context()
 {
    struct CurrentContextDueItemModel;
    struct NoRecallClassifierModel;

    impl ModelClient for CurrentContextDueItemModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "I just got the payroll email");
            let due_item_tool_mentions = request
                .transcript
                .iter()
                .filter(|message| message.role == MessageRole::Tool)
                .filter(|message| {
                    message.content.as_deref().is_some_and(|content| {
                        content.to_ascii_lowercase().contains("payroll follow up")
                    })
                })
                .count();
            assert_eq!(
                due_item_tool_mentions, 1,
                "a due item already pinned in current context should not be re-injected by contextual due-item surfacing"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "You should reply to payroll.".to_string(),
            }])
        }
    }

    impl ModelClient for NoRecallClassifierModel {
        fn next_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            Ok(vec![StreamEvent::AssistantResponse {
                    content: r#"{"needs_recall":false,"reasoning":"Contextual reminder should come only from current context here."}"#
                        .to_string(),
                }])
        }
    }

    let unique = format!(
        "elroy-rs-app-contextual-due-item-current-context-no-duplicate-{}",
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

    let pinned_due_item =
        elroy_db::find_active_agenda_item_by_name(&connection, "payroll follow up")
            .expect("due item lookup should succeed")
            .expect("due item should exist");
    elroy_db::replace_context_messages(
        &mut connection,
        LOCAL_USER_TOKEN,
        &context_due_item_tool_messages(&pinned_due_item),
    )
    .expect("pinned due item context should persist");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "I just got the payroll email",
        &CurrentContextDueItemModel,
        recall_model_clients(Some(&NoRecallClassifierModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            if content.contains("reply to payroll")
    )));

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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: config.reflect,
        },
    )
    .expect("second prompt should succeed");

    let memories = elroy_db::list_active_memories(&connection, 10).expect("memories should list");
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: config.reflect,
        },
    )
    .expect("second prompt should succeed");

    let memories = elroy_db::list_active_memories(&connection, 10).expect("memories should list");
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
fn refresh_context_if_needed_persists_model_authored_summary_when_provider_available() {
    let unique = format!(
        "elroy-rs-app-context-refresh-model-summary-{}",
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
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "I reminded the user about payroll and kept the conversation focused."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");

    let mut transcript = vec![ConversationMessage::new(MessageRole::System, "system")];
    for index in 0..8 {
        transcript.push(ConversationMessage::new(
            MessageRole::User,
            format!("user {index} mentioned payroll and planning repeated repeated"),
        ));
        transcript.push(ConversationMessage::new(
            MessageRole::Assistant,
            format!("assistant {index} replied about payroll and planning repeated repeated"),
        ));
    }
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());

    let refreshed = refresh_context_if_needed(
        &mut connection,
        &config,
        &BootstrapPlan::from_config(&config),
    )
    .expect("context refresh should succeed");

    assert!(refreshed);

    let stored =
        elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert_eq!(
        summary_content,
        "Recent conversation summary: I reminded the user about payroll and kept the conversation focused."
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn app_runtime_refresh_context_persists_model_authored_summary_when_provider_available() {
    let unique = format!(
        "elroy-rs-app-runtime-context-refresh-model-summary-{}",
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
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "I reminded the user about payroll and kept the conversation focused."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");

    let mut transcript = vec![ConversationMessage::new(MessageRole::System, "system")];
    for index in 0..8 {
        transcript.push(ConversationMessage::new(
            MessageRole::User,
            format!("user {index} mentioned payroll and planning repeated repeated"),
        ));
        transcript.push(ConversationMessage::new(
            MessageRole::Assistant,
            format!("assistant {index} replied about payroll and planning repeated repeated"),
        ));
    }
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");
    drop(connection);

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());

    let runtime = AppRuntime::new(config.clone());
    let refreshed = runtime
        .refresh_context_if_needed()
        .expect("context refresh should succeed");

    assert!(refreshed);

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert_eq!(
        summary_content,
        "Recent conversation summary: I reminded the user about payroll and kept the conversation focused."
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn refresh_context_if_needed_falls_back_to_deterministic_summary_when_model_returns_empty_text() {
    let unique = format!(
        "elroy-rs-app-context-refresh-summary-fallback-{}",
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
    let _mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": ""
                    }]
                }]
            })
            .to_string(),
        )
        .create();

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
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());

    let refreshed = refresh_context_if_needed(
        &mut connection,
        &config,
        &BootstrapPlan::from_config(&config),
    )
    .expect("context refresh should succeed");

    assert!(refreshed);

    let stored =
        elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert!(summary_content.starts_with("Recent conversation summary:"));
    assert!(summary_content.contains("user 7 words repeated"));
    assert!(summary_content.contains("assistant 7 words repeated"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn refresh_context_if_needed_can_use_fast_model_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-context-refresh-fast-model-{}",
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
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "I condensed the earlier practice discussion into a short first-person summary."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");

    let mut transcript = vec![ConversationMessage::new(MessageRole::System, "system")];
    for index in 0..8 {
        transcript.push(ConversationMessage::new(
            MessageRole::User,
            format!("user {index} mentioned practice planning repeated repeated"),
        ));
        transcript.push(ConversationMessage::new(
            MessageRole::Assistant,
            format!("assistant {index} replied about practice planning repeated repeated"),
        ));
    }
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));

    let refreshed = refresh_context_if_needed(
        &mut connection,
        &config,
        &BootstrapPlan::from_config(&config),
    )
    .expect("context refresh should succeed");

    assert!(refreshed);

    let stored =
        elroy_db::load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert_eq!(
        summary_content,
        "Recent conversation summary: I condensed the earlier practice discussion into a short first-person summary."
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn app_runtime_refresh_context_can_use_fast_model_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-runtime-context-refresh-fast-model-{}",
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
    let _mock = server
            .mock("POST", "/responses")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "I condensed the earlier practice discussion into a short first-person summary."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");

    let mut transcript = vec![ConversationMessage::new(MessageRole::System, "system")];
    for index in 0..8 {
        transcript.push(ConversationMessage::new(
            MessageRole::User,
            format!("user {index} mentioned practice planning repeated repeated"),
        ));
        transcript.push(ConversationMessage::new(
            MessageRole::Assistant,
            format!("assistant {index} replied about practice planning repeated repeated"),
        ));
    }
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");
    drop(connection);

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", server.url()));

    let runtime = AppRuntime::new(config.clone());
    let refreshed = runtime
        .refresh_context_if_needed()
        .expect("context refresh should succeed");

    assert!(refreshed);

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert_eq!(
        summary_content,
        "Recent conversation summary: I condensed the earlier practice discussion into a short first-person summary."
    );

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn app_runtime_refresh_context_falls_back_to_deterministic_summary_when_model_returns_empty_text() {
    let unique = format!(
        "elroy-rs-app-runtime-context-refresh-fallback-{}",
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
    let _mock = server
        .mock("POST", "/responses")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": ""
                    }]
                }]
            })
            .to_string(),
        )
        .create();

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
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &transcript)
        .expect("messages should persist");
    drop(connection);

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.max_tokens = 40;
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());

    let runtime = AppRuntime::new(config.clone());
    let refreshed = runtime
        .refresh_context_if_needed()
        .expect("context refresh should succeed");

    assert!(refreshed);

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let summary_content = stored[stored.len() - 1]
        .content
        .as_deref()
        .expect("summary tool content should exist");
    assert!(summary_content.starts_with("Recent conversation summary:"));
    assert!(summary_content.contains("user 7 words repeated"));
    assert!(summary_content.contains("assistant 7 words repeated"));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_use_fast_model_for_prompt_time_recall_classifier_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-model-recall-classifier-{}",
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
        memory_dir.join("python_library.md"),
        "# Python Library\n\nYou mentioned the requests library for Python projects.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let fast_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .expect(1)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"needs_recall":false,"reasoning":"This follow-up does not need memory recall."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "What was that library you mentioned?".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Main chat model replied without injected recall."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let mut connection =
        open_sqlite_connection(&config.database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    elroy_db::replace_context_messages(
        &mut connection,
        LOCAL_USER_TOKEN,
        &[ConversationMessage::new(
            MessageRole::Assistant,
            "You should look at the requests library.",
        )],
    )
    .expect("messages should persist");
    drop(connection);

    let result = runtime
        .process_message(
            "What was that library you mentioned?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(!result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Main chat model replied without injected recall."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(!stored.iter().any(|message| {
        message.role == MessageRole::Tool
            && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
    }));

    fast_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_use_fast_model_for_reflective_recall_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-model-reflective-recall-{}",
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
        "# Basketball Form\n\nBring the resistance bands to practice.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"The recalled memory is relevant."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let reflective_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Recalled Memory Content".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"is_relevant":true,"content":"I remember that the user should bring the resistance bands to practice."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Bring the resistance bands to practice."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What should I remember before basketball practice?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands to practice."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(
        stored
            .iter()
            .any(|message| { message_matches_tool_call_id(message, "bootstrap-memory-recall") })
    );
    assert!(stored.iter().any(|message| {
        message.content.as_deref().is_some_and(|content| {
            content
                .contains("I remember that the user should bring the resistance bands to practice.")
                && content.contains("\"recall_metadata\"")
        })
    }));

    relevance_mock.assert();
    reflective_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_use_embedding_only_for_reflective_due_item_recall() {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-only-reflective-due-item-recall-{}",
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
    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_reminder_{index:03}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet {index}.\n"
                ),
            )
            .expect("recent due item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic due item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "I also recall these due items may matter".to_string(),
        ))
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills\\.".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "You should pack the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_reminder_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent due item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_reminder.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older due item timestamp should update");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "You should pack the resistance bands before drills."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(stored.iter().any(|message| {
        message_matches_tool_call_id(message, "bootstrap-memory-recall")
            && message.content.as_deref().is_some_and(|content| {
                content.contains("I also recall these due items may matter")
                    && content.contains("old training reminder")
                    && content.contains("\"memory_type\": \"AgendaItem\"")
            })
    }));

    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_does_not_persist_irrelevant_reflective_recall() {
    let unique = format!(
        "elroy-rs-app-process-message-reflective-recall-suppression-{}",
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

    let mut fast_server = mockito::Server::new();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"The recalled memory is relevant enough to inspect."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let reflective_mock = fast_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer fast-test-key")
        .match_body(mockito::Matcher::Regex(
            "Recalled Memory Content".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": r#"{"is_relevant":false,"content":null}"#
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "No prior reminder looks especially relevant."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What should I remember before basketball practice?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(!result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "No prior reminder looks especially relevant."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(
        !stored
            .iter()
            .any(|message| { message_matches_tool_call_id(message, "bootstrap-memory-recall") })
    );

    relevance_mock.assert();
    reflective_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_does_not_duplicate_persisted_reflective_recall_on_later_turn() {
    let unique = format!(
        "elroy-rs-app-process-message-reflective-recall-dedupe-runtime-{}",
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
        memory_dir.join("practice_gear.md"),
        "# Practice Gear\n\nPack resistance bands before training.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .expect(1)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"The practice gear memory is relevant."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let reflective_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Recalled Memory Content".to_string(),
            ))
            .expect(1)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"is_relevant":true,"content":"I remember that the user should pack resistance bands before training."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let first_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "I remember that the user should pack resistance bands before training\\.".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Pack the resistance bands."
                }]
            })
            .to_string(),
        )
        .create();
    let second_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Should I remember that same practice gear again\\?".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "That same practice gear reminder is already in context."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let first_result = runtime
        .process_message(
            "What should I remember before practice?",
            MessageProcessOptions::default(),
        )
        .expect("first prompt should succeed");
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));

    let second_result = runtime
        .process_message(
            "Should I remember that same practice gear again?",
            MessageProcessOptions::default(),
        )
        .expect("second prompt should succeed");
    assert!(!second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "That same practice gear reminder is already in context."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let bootstrap_recall_messages = stored
        .iter()
        .filter(|message| message_matches_tool_call_id(message, "bootstrap-memory-recall"))
        .count();
    assert_eq!(bootstrap_recall_messages, 2);

    relevance_mock.assert();
    reflective_mock.assert();
    first_chat_mock.assert();
    second_chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_use_fast_model_for_prompt_time_fast_recall_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-model-fast-recall-{}",
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
        memory_dir.join("practice_gear.md"),
        "# Practice Gear\n\nPack resistance bands before training.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"This candidate is relevant even though the wording differs."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before training\\.".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Bring the resistance bands."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What gear should I bring to practice?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(!result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands."
    )));

    relevance_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_surface_older_semantic_memory_via_embedding_candidates() {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-recall-{}",
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
    for index in 0..120 {
        fs::write(
            memory_dir.join(format!("recent_note_{index:03}.md")),
            format!("Review the storage locker spreadsheet {index}.\n"),
        )
        .expect("recent memory should be written");
    }
    fs::write(
        memory_dir.join("old_training_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut fast_server = mockito::Server::new();
    let mut answers = vec![false; 100];
    answers[0] = true;
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": serde_json::json!({
                                "answers": answers,
                                "reasoning": "Only the older training-kit memory matches the user's intent."
                            }).to_string()
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let _query_embedding_mock = fast_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = fast_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = fast_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills\\.".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Bring the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.embedding_model_api_key = Some("fast-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    memory_dir
                        .join(format!("recent_note_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent memory timestamp should update");
    }
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                memory_dir
                    .join("old_training_note.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older memory timestamp should update");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands before drills."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(stored.iter().any(|message| {
        message_matches_tool_call_id(message, "bootstrap-memory-recall")
            && message.content.as_deref().is_some_and(|content| {
                content.contains("old training note")
                    && content.contains("Pack resistance bands before drills.")
            })
    }));

    relevance_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_reuse_persisted_embedding_cache_for_prompt_time_recall() {
    let unique = format!(
        "elroy-rs-app-process-message-persisted-embedding-cache-{}",
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
        memory_dir.join("recent_planning_note.md"),
        "Review the storage locker spreadsheet.\n",
    )
    .expect("recent memory should be written");
    fs::write(
        memory_dir.join("recent_inventory_note.md"),
        "Double-check the equipment inventory list.\n",
    )
    .expect("second recent memory should be written");
    fs::write(
        memory_dir.join("old_training_note.md"),
        "Pack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut embedding_server = mockito::Server::new();
    let query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills\\.".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Bring the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    let memories = crate::list_active_memories_in_scope(&connection, &memory_dir, 10)
        .expect("memories should load");
    for memory in &memories {
        let embedding = match memory.name.as_str() {
            "old training note" => vec![1.0, 0.0],
            "recent planning note" => vec![0.0, 1.0],
            "recent inventory note" => vec![0.0, 0.95],
            other => panic!("unexpected memory name: {other}"),
        };
        upsert_memory_embedding(
            &connection,
            &memory.file_path,
            &embedding,
            &crate::memory_embedding_text(memory),
        )
        .expect("embedding cache should persist");
    }

    let runtime = AppRuntime::new(config);
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands before drills."
    )));

    query_embedding_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_surface_older_semantic_due_item_via_embedding_without_relevance_model() {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-due-item-recall-{}",
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
    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_reminder_{index:03}.md")),
                format!(
                    "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet {index}.\n"
                ),
            )
            .expect("recent due item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic due item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills\\.".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Pack the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_reminder_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent due item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_reminder.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older due item timestamp should update");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Pack the resistance bands before drills."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(stored.iter().any(|message| {
        message_matches_tool_call_id(message, "bootstrap-memory-recall")
            && message.content.as_deref().is_some_and(|content| {
                content.contains("old training reminder")
                    && content.contains("Pack resistance bands before drills.")
            })
    }));

    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_surface_older_semantic_agenda_item_via_embedding_without_relevance_model() {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-agenda-item-recall-{}",
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
    for index in 0..120 {
        fs::write(
                agenda_dir.join(format!("recent_plan_{index:03}.md")),
                format!(
                    "---\ndate: 2026-05-{day:02}\ncompleted: false\nstatus: created\n---\n\nReview the storage locker spreadsheet {index}.\n",
                    day = (index % 28) + 1,
                ),
            )
            .expect("recent agenda item should be written");
    }
    fs::write(
            agenda_dir.join("old_training_plan.md"),
            "---\ndate: 2026-04-01\ncompleted: false\nstatus: created\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("older semantic agenda item should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the storage locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills\\.".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Bring the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE agenda_items SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    agenda_dir
                        .join(format!("recent_plan_{index:03}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent agenda item timestamp should update");
    }
    connection
        .execute(
            "UPDATE agenda_items SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![
                agenda_dir
                    .join("old_training_plan.md")
                    .display()
                    .to_string(),
            ],
        )
        .expect("older agenda item timestamp should update");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands before drills."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(stored.iter().any(|message| {
        message_matches_tool_call_id(message, "bootstrap-memory-recall")
            && message.content.as_deref().is_some_and(|content| {
                content.contains("old training plan")
                    && content.contains("Pack resistance bands before drills.")
            })
    }));

    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_inject_and_persist_prompt_time_fast_recall() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-recall-runtime-{}",
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
        memory_dir.join("practice_gear.md"),
        "# Practice Gear\n\nPack resistance bands before training.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let fast_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .expect(1)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"The practice gear memory is relevant."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before training\\.".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "Bring the resistance bands."
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What gear should I bring to practice?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Bring the resistance bands."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(
        stored
            .iter()
            .any(|message| { message_matches_tool_call_id(message, "bootstrap-memory-recall") })
    );
    assert!(stored.iter().any(|message| {
        message
            .content
            .as_deref()
            .is_some_and(|content| content.contains("\"recall_metadata\""))
    }));

    fast_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_does_not_duplicate_persisted_prompt_time_fast_recall_on_later_turn() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-recall-dedupe-runtime-{}",
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
        memory_dir.join("practice_gear.md"),
        "# Practice Gear\n\nPack resistance bands before training.\n",
    )
    .expect("memory file should be written");

    let mut fast_server = mockito::Server::new();
    let fast_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .expect(1)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"The practice gear memory is relevant."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let first_chat_mock = chat_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before training\\.".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "Bring the resistance bands."
                    }]
                }]
            })
            .to_string(),
        )
        .create();
    let second_chat_mock = chat_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Regex(
            "Should I remember the same practice gear again\\?".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "The same practice gear reminder is still in context."
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let first_result = runtime
        .process_message(
            "What gear should I bring to practice?",
            MessageProcessOptions::default(),
        )
        .expect("first prompt should succeed");
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));

    let second_result = runtime
        .process_message(
            "Should I remember the same practice gear again?",
            MessageProcessOptions::default(),
        )
        .expect("second prompt should succeed");
    assert!(!second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "The same practice gear reminder is still in context."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let bootstrap_recall_messages = stored
        .iter()
        .filter(|message| message_matches_tool_call_id(message, "bootstrap-memory-recall"))
        .count();
    assert_eq!(bootstrap_recall_messages, 2);

    fast_mock.assert();
    first_chat_mock.assert();
    second_chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_inject_and_persist_mixed_fast_recall() {
    let unique = format!(
        "elroy-rs-app-process-message-mixed-fast-recall-runtime-{}",
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

    let mut fast_server = mockito::Server::new();
    let fast_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .expect(4)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true,true,true],"reasoning":"All three recalled items are relevant to basketball practice."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex(
                    "Remember to follow through on your shot\\.".to_string(),
                ),
                mockito::Matcher::Regex("Bring the resistance bands".to_string()),
                mockito::Matcher::Regex(
                    "Focus on basketball practice footwork and follow-through".to_string(),
                ),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "Remember your follow-through, bring the resistance bands, and review the drill plan."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What should I remember for basketball practice?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(result.events.iter().any(|event| matches!(
            event,
            StreamEvent::AssistantResponse { content }
                if content == "Remember your follow-through, bring the resistance bands, and review the drill plan."
        )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(
        stored
            .iter()
            .any(|message| { message_matches_tool_call_id(message, "bootstrap-memory-recall") })
    );
    assert!(stored.iter().any(|message| {
        message.content.as_deref().is_some_and(|content| {
            content.contains("basketball form")
                && content.contains("practice reminder")
                && content.contains("drill plan")
                && content.contains("\"memory_type\": \"Memory\"")
                && content.contains("\"memory_type\": \"AgendaItem\"")
        })
    }));

    fast_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_use_fast_model_for_contextual_due_item_selection_when_chat_model_differs() {
    let unique = format!(
        "elroy-rs-app-process-message-fast-model-contextual-due-item-{}",
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
            agenda_dir.join("practice_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nBring the resistance bands\n",
        )
        .expect("contextual due item file should be written");

    let mut fast_server = mockito::Server::new();
    let classifier_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Analyze if this message requires recalling information from long-term memory"
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"needs_recall":false,"reasoning":"This should come from contextual reminder surfacing instead of memory recall."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"This reminder is relevant even though the wording differs."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Bring the resistance bands".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "You should bring the resistance bands."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What gear should I bring?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "You should bring the resistance bands."
    )));

    classifier_mock.assert();
    relevance_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_surface_future_hybrid_contextual_due_item_via_embedding_without_relevance_model()
 {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-contextual-hybrid-due-item-{}",
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
            agenda_dir.join("workout_kit_review.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the workout locker spreadsheet.\n",
        )
        .expect("overlap contextual due item file should be written");
    fs::write(
            agenda_dir.join("practice_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2099-05-20T09:00:00\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("future hybrid due item file should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _overlap_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the workout locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let _chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Pack the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Pack the resistance bands before drills."
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_does_not_duplicate_persisted_contextual_due_item_on_later_turn() {
    let unique = format!(
        "elroy-rs-app-process-message-contextual-due-item-dedupe-runtime-{}",
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

    let mut fast_server = mockito::Server::new();
    let classifier_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Analyze if this message requires recalling information from long-term memory"
                    .to_string(),
            ))
            .expect(2)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"needs_recall":false,"reasoning":"This should come from contextual reminder surfacing instead of memory recall."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let relevance_mock = fast_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer fast-test-key")
        .match_body(mockito::Matcher::Regex(
            "Your job is to determine which candidate recall items are relevant to a query\\."
                .to_string(),
        ))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": r#"{"answers":[true],"reasoning":"This reminder is relevant."}"#
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let first_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex("Reply to payroll".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "You should reply to payroll."
                }]
            })
            .to_string(),
        )
        .create();
    let second_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "I just got another payroll email".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "That payroll reminder is already in context."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let first_result = runtime
        .process_message(
            "I just got the payroll email",
            MessageProcessOptions::default(),
        )
        .expect("first prompt should succeed");
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "You should reply to payroll."
    )));

    let second_result = runtime
        .process_message(
            "I just got another payroll email",
            MessageProcessOptions::default(),
        )
        .expect("second prompt should succeed");
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "That payroll reminder is already in context."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let contextual_due_item_messages = stored
        .iter()
        .filter(|message| {
            message_matches_tool_call_id(
                message,
                &context_due_item_tool_call_id("payroll follow up"),
            )
        })
        .count();
    assert_eq!(
        contextual_due_item_messages, 2,
        "the contextual due item tool-call/result pair should only be persisted once across turns"
    );

    classifier_mock.assert();
    relevance_mock.assert();
    first_chat_mock.assert();
    second_chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_surfaces_contextual_task_without_deleting_or_duplicating_it() {
    let unique = format!(
        "elroy-rs-app-process-message-contextual-task-runtime-{}",
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
    let task_path = agenda_dir.join("payroll_follow_up.md");
    fs::write(
            &task_path,
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\ntrigger_context: after payroll email\n---\n\nReply to payroll\n",
        )
        .expect("contextual task file should be written");

    let mut fast_server = mockito::Server::new();
    let classifier_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Analyze if this message requires recalling information from long-term memory"
                    .to_string(),
            ))
            .expect(2)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"needs_recall":false,"reasoning":"This should come from contextual task surfacing instead of memory recall."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();
    let relevance_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .match_body(mockito::Matcher::Regex(
                "Your job is to determine which candidate recall items are relevant to a query\\."
                    .to_string(),
            ))
            .expect(2)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[true],"reasoning":"This triggered task is relevant."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let first_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex("Reply to payroll".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "You should reply to payroll."
                }]
            })
            .to_string(),
        )
        .create();
    let second_chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "I just got another payroll email".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "That payroll task is already in context."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let first_result = runtime
        .process_message(
            "I just got the payroll email",
            MessageProcessOptions::default(),
        )
        .expect("first prompt should succeed");
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(first_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "You should reply to payroll."
    )));

    let second_result = runtime
        .process_message(
            "I just got another payroll email",
            MessageProcessOptions::default(),
        )
        .expect("second prompt should succeed");
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "classifying recall..."
    )));
    assert!(second_result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "That payroll task is already in context."
    )));

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    let contextual_due_item_messages = stored
        .iter()
        .filter(|message| {
            message_matches_tool_call_id(
                message,
                &context_due_item_tool_call_id("payroll follow up"),
            )
        })
        .count();
    assert_eq!(
        contextual_due_item_messages, 2,
        "the contextual task tool-call/result pair should only be persisted once across turns"
    );

    let active_task_count: i64 = reopened
            .query_row(
                "SELECT COUNT(*) FROM agenda_items WHERE name = ?1 AND status = 'created' AND is_active = 1",
                rusqlite::params!["payroll follow up"],
                |row| row.get(0),
            )
            .expect("active task rows should query");
    assert_eq!(active_task_count, 1);
    assert!(
        task_path.exists(),
        "contextual task file should remain on disk"
    );

    classifier_mock.assert();
    relevance_mock.assert();
    first_chat_mock.assert();
    second_chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_can_surface_semantic_contextual_task_via_embedding_without_relevance_model() {
    let unique = format!(
        "elroy-rs-app-process-message-embedding-contextual-task-{}",
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
    let semantic_task_path = agenda_dir.join("practice_packing_task.md");
    fs::write(
            agenda_dir.join("workout_kit_review.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the workout locker spreadsheet.\n",
        )
        .expect("overlap contextual task file should be written");
    fs::write(
            &semantic_task_path,
            "---\ndate: 2026-05-21\ncompleted: false\nstatus: created\ntrigger_context: before basketball practice\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("semantic contextual task file should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _overlap_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the workout locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/messages")
        .match_header("x-api-key", "anthropic-test-key")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .expect_at_least(1)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": "Pack the resistance bands before drills."
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.chat_model = "claude-sonnet-4-20250514".to_string();
    config.anthropic_api_key = Some("anthropic-test-key".to_string());
    config.anthropic_base_url = format!("{}/messages", chat_server.url());
    config.embedding_model_api_key = Some("embedding-test-key".to_string());
    config.embedding_model_api_base = Some(format!("{}/embeddings", embedding_server.url()));
    config.openai_api_key = None;
    config.fast_model_api_key = None;
    config.memory_recall_classifier_enabled = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What belongs in my workout kit?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content == "Pack the resistance bands before drills."
    )));

    let reopened = open_sqlite_connection(&config.database_path).expect("database should reopen");
    let active_task_count: i64 = reopened
            .query_row(
                "SELECT COUNT(*) FROM agenda_items WHERE name = ?1 AND status = 'created' AND is_active = 1",
                rusqlite::params!["practice packing task"],
                |row| row.get(0),
            )
            .expect("active task rows should query");
    assert_eq!(active_task_count, 1);
    assert!(
        semantic_task_path.exists(),
        "semantic contextual task file should remain on disk"
    );

    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_surfaces_and_cleans_up_due_items_end_to_end() {
    let unique = format!(
        "elroy-rs-app-process-message-due-item-cleanup-{}",
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

    let mut server = mockito::Server::new();
    let first_mock = server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Regex(
            "Take your daily medicine".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "function_call",
                    "call_id": "call-delete-due-item",
                    "name": "delete_due_item",
                    "arguments": "{\"name\":\"medicine reminder\"}"
                }]
            })
            .to_string(),
        )
        .create();
    let second_mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::Regex(
                "function_call_output".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "You had a reminder to take your daily medicine, and I've cleared it for you."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "Hi, how are you doing today?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallRequested(call)
            if call.name == "delete_due_item"
                && call.arguments_json == "{\"name\":\"medicine reminder\"}"
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantToolResult { content, is_error }
            if !is_error
                && content.contains("Due item 'medicine reminder' has been deleted.")
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("take your daily medicine")
    )));

    let mut connection =
        open_sqlite_connection(&config.database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let active_due_items = list_active_due_items(&connection, 10).expect("due items should list");
    assert!(
        !active_due_items
            .iter()
            .any(|item| item.name == "medicine reminder")
    );
    assert!(!agenda_dir.join("medicine_reminder.md").exists());

    first_mock.assert();
    second_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_surfaces_multiple_due_items_end_to_end() {
    let unique = format!(
        "elroy-rs-app-process-message-multiple-due-items-{}",
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

    let mut server = mockito::Server::new();
    let mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::Regex("First due reminder".to_string()),
                mockito::Matcher::Regex("Second due reminder".to_string()),
            ]))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "You have two reminders due: First due reminder and Second due reminder."
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message(
            "What's on my schedule today?",
            MessageProcessOptions::default(),
        )
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("First due reminder")
                && content.contains("Second due reminder")
    )));

    mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_skips_future_due_item_context_end_to_end() {
    let unique = format!(
        "elroy-rs-app-process-message-future-due-item-{}",
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

    let mut fast_server = mockito::Server::new();
    let fast_mock = fast_server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer fast-test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": r#"{"answers":[false],"reasoning":"The future reminder is not relevant to the weather question."}"#
                        }]
                    }]
                })
                .to_string(),
            )
            .create();

    let mut chat_server = mockito::Server::new();
    let chat_mock = chat_server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "Weather looks calm today."
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", chat_server.url());
    config.fast_model = Some("gpt-5.4-mini".to_string());
    config.fast_model_api_key = Some("fast-test-key".to_string());
    config.fast_model_api_base = Some(format!("{}/responses", fast_server.url()));
    config.memory_recall_classifier_enabled = false;
    config.reflect = false;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config.clone());
    let result = runtime
        .process_message("How's the weather today?", MessageProcessOptions::default())
        .expect("prompt should succeed");

    assert!(!result.events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallRequested(call) if call.name == "delete_due_item"
    )));
    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content } if content == "Weather looks calm today."
    )));

    let mut connection =
        open_sqlite_connection(&config.database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let active_due_items = list_active_due_items(&connection, 10).expect("due items should list");
    assert!(
        active_due_items
            .iter()
            .any(|item| item.name == "future reminder")
    );

    fast_mock.assert();
    chat_mock.assert();
    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn process_message_surfaces_hybrid_due_item_when_time_due_end_to_end() {
    let unique = format!(
        "elroy-rs-app-process-message-hybrid-due-item-{}",
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

    let mut server = mockito::Server::new();
    let mock = server
        .mock("POST", "/responses")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Regex("Hybrid reminder text".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "A hybrid reminder is due: Hybrid reminder text."
                    }]
                }]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.openai_api_key = Some("test-key".to_string());
    config.openai_base_url = format!("{}/responses", server.url());
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let runtime = AppRuntime::new(config);
    let result = runtime
        .process_message("What's happening?", MessageProcessOptions::default())
        .expect("prompt should succeed");

    assert!(result.events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("Hybrid reminder text")
    )));

    mock.assert();
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

    assert!(formatted.starts_with("Conversation Summary"));
    assert!(formatted.contains("User ("));
    assert!(formatted.contains("): I need to finish payroll"));
    assert!(formatted.contains("Elroy ("));
    assert!(formatted.contains("): I should look that up."));
    assert!(formatted.contains("Elroy TOOL CALL REQUEST ("));
    assert!(formatted.contains("function name: search_memories"));
    assert!(formatted.contains("arguments: {\"query\":\"payroll\"}"));
    assert!(formatted.contains("TOOL CALL RESULT ("));
    assert!(formatted.contains("): Found payroll reminder."));
    assert!(formatted.contains("Messages from "));
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
            ConversationMessage::new(MessageRole::Assistant, "You mentioned a payroll deadline."),
        ],
    )
    .expect("summary should be generated");

    assert_eq!(
        summary,
        "Recent conversation summary: I reminded the user about payroll and the tone stayed focused."
    );
}

#[test]
fn summarize_context_messages_with_model_adds_python_style_word_limit_instruction() {
    struct SummaryPromptInspectionModel;

    impl ModelClient for SummaryPromptInspectionModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert!(request.user_message.contains("Conversation Summary"));
            assert!(
                request
                    .user_message
                    .contains("Your word limit is 300. DO NOT EXCEED IT."),
                "{}",
                request.user_message
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "I stayed focused on payroll.".to_string(),
            }])
        }
    }

    let summary = summarize_context_messages_with_model(
        &SummaryPromptInspectionModel,
        "Elroy",
        "User",
        &[
            ConversationMessage::new(MessageRole::User, "I need to finish payroll"),
            ConversationMessage::new(MessageRole::Assistant, "You mentioned a payroll deadline."),
        ],
    )
    .expect("summary should be generated");

    assert_eq!(
        summary,
        "Recent conversation summary: I stayed focused on payroll."
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
    let before_memories = crate::list_active_memories_in_scope(&connection, &config.memory_dir, 10)
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
    let active_memories = crate::list_active_memories_in_scope(&reopened, &config.memory_dir, 10)
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
    let before_memories = crate::list_active_memories_in_scope(&connection, &config.memory_dir, 10)
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
    let active_memories = crate::list_active_memories_in_scope(&reopened, &config.memory_dir, 10)
        .expect("active memories should load");
    assert_eq!(active_memories.len(), 1);

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn append_startup_session_context_persists_hidden_python_style_bootstrap_pair() {
    let unique = format!(
        "elroy-rs-app-startup-session-context-{}",
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

    let runtime = AppRuntime::new(config.clone());
    let mut connection =
        open_sqlite_connection(&config.database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let first_today_user = ConversationMessage {
        role: MessageRole::User,
        content: Some("morning check-in".to_string()),
        chat_model: None,
        id: None,
        created_at_unix: Utc::now().timestamp() - 120,
        tool_calls: None,
        tool_call_id: None,
    };
    elroy_db::replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &[first_today_user])
        .expect("messages should persist");
    drop(connection);

    runtime
        .append_startup_session_context()
        .expect("startup session context should persist");

    let mut reopened =
        open_sqlite_connection(&config.database_path).expect("database should reopen");
    let stored = elroy_db::load_context_messages(&mut reopened, LOCAL_USER_TOKEN).expect("load ok");
    assert!(stored.len() >= 3);
    assert!(stored.iter().any(|message| {
        is_bootstrap_session_context_message(message)
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| calls.iter().any(|call| call.name == "get_session_context"))
    }));
    let session_context_result = stored
        .iter()
        .find(|message| {
            message.role == MessageRole::Tool
                && message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("bootstrap-session-context:"))
        })
        .expect("tool result should persist");
    let content = session_context_result
        .content
        .as_deref()
        .expect("tool result should have content");
    assert!(content.contains("Current date/time:"));
    assert!(content.contains("has logged in."));
    assert!(content.contains("I first started chatting"));

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
        content: r#"{"needs_recall":false,"reasoning":"This is only a lightweight follow-up."}"#
            .to_string(),
    }]]);
    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What was that library you mentioned?",
        &NoRecallPromptModel,
        recall_model_clients(Some(&classifier)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        recall_model_clients(Some(&classifier_and_filter)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        recall_model_clients(Some(&classifier_and_filter)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
fn run_prompt_with_model_and_registry_can_prefer_semantic_recall_candidate_over_weaker_overlap() {
    struct SemanticRecallPromptModel;
    struct SemanticRecallRelevanceModel;

    impl ModelClient for SemanticRecallPromptModel {
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
            assert!(
                recall_payload.contains("resistance bands"),
                "{recall_payload}"
            );
            assert!(
                !recall_payload.contains("Gear Inventory"),
                "{recall_payload}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Bring the resistance bands.".to_string(),
            }])
        }
    }

    impl ModelClient for SemanticRecallRelevanceModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            let answers =
                if prompt.find("resistance bands") < prompt.find("storage locker spreadsheet") {
                    vec![true, false]
                } else {
                    vec![false, true]
                };
            Ok(vec![StreamEvent::AssistantResponse {
                content: serde_json::json!({
                    "answers": answers,
                    "reasoning": "Only the resistance-bands memory matches the user's intent."
                })
                .to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-recall-semantic-priority-{}",
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
        memory_dir.join("gear_inventory.md"),
        "# Gear Inventory\n\nReview the storage locker spreadsheet.\n",
    )
    .expect("overlap memory file should be written");
    fs::write(
        memory_dir.join("bands_note.md"),
        "# Training Kit\n\nPack resistance bands before drills.\n",
    )
    .expect("semantic memory file should be written");
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What gear should I bring to practice?",
        &SemanticRecallPromptModel,
        recall_model_clients(Some(&SemanticRecallRelevanceModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_can_surface_older_semantic_memory_beyond_old_candidate_cap() {
    struct OlderSemanticRecallPromptModel;
    struct OlderSemanticRecallRelevanceModel;

    impl ModelClient for OlderSemanticRecallPromptModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What belongs in my workout kit?");
            let recall_payload = request
                .transcript
                .iter()
                .find(|message| {
                    message.role == MessageRole::Tool
                        && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                })
                .and_then(|message| message.content.as_deref())
                .expect("fast recall payload should be injected");
            assert!(
                recall_payload.contains("Resistance Bands"),
                "{recall_payload}"
            );
            assert!(
                !recall_payload.contains("Recent Memory 00"),
                "{recall_payload}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Pack the resistance bands.".to_string(),
            }])
        }
    }

    impl ModelClient for OlderSemanticRecallRelevanceModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            if prompt.contains("needs_recall") {
                return Ok(vec![StreamEvent::AssistantResponse {
                        content: r#"{"needs_recall":true,"reasoning":"The user is asking what belongs in the workout kit."}"#
                            .to_string(),
                    }]);
            }
            assert!(prompt.contains("Resistance Bands"), "{prompt}");
            let answers = prompt
                .lines()
                .filter_map(|line| {
                    let (index, candidate) = line.split_once(". ")?;
                    index
                        .chars()
                        .all(|character| character.is_ascii_digit())
                        .then_some(candidate.contains("resistance bands"))
                })
                .collect::<Vec<_>>();
            assert!(answers.iter().any(|answer| *answer), "{prompt}");
            Ok(vec![StreamEvent::AssistantResponse {
                content: serde_json::json!({
                    "answers": answers,
                    "reasoning": "Only the older training-kit memory matches the user's intent."
                })
                .to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-recall-older-semantic-memory-{}",
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

    for index in 0..60 {
        fs::write(
            memory_dir.join(format!("recent_memory_{index:02}.md")),
            format!(
                "# Recent Memory {index:02}\n\nReview the workout locker spreadsheet {index}.\n"
            ),
        )
        .expect("recent memory should be written");
    }
    fs::write(
        memory_dir.join("resistance_bands.md"),
        "# Resistance Bands\n\nPack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();

    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..60 {
        connection
            .execute(
                "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    memory_dir
                        .join(format!("recent_memory_{index:02}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent memory timestamp should update");
    }
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![memory_dir.join("resistance_bands.md").display().to_string(),],
        )
        .expect("older memory timestamp should update");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What belongs in my workout kit?",
        &OlderSemanticRecallPromptModel,
        recall_model_clients(Some(&OlderSemanticRecallRelevanceModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_can_surface_older_semantic_memory_via_embedding_without_relevance_model()
 {
    struct OlderSemanticEmbeddingOnlyPromptModel;

    impl ModelClient for OlderSemanticEmbeddingOnlyPromptModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(request.user_message, "What belongs in my workout kit?");
            let recall_payload = request
                .transcript
                .iter()
                .find(|message| {
                    message.role == MessageRole::Tool
                        && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                })
                .and_then(|message| message.content.as_deref())
                .expect("fast recall payload should be injected");
            assert!(
                recall_payload.contains("Resistance Bands"),
                "{recall_payload}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Pack the resistance bands.".to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-recall-older-semantic-memory-embedding-only-{}",
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

    for index in 0..120 {
        fs::write(
            memory_dir.join(format!("recent_memory_{index:02}.md")),
            format!(
                "# Recent Memory {index:02}\n\nReview the workout locker spreadsheet {index}.\n"
            ),
        )
        .expect("recent memory should be written");
    }
    fs::write(
        memory_dir.join("resistance_bands.md"),
        "# Resistance Bands\n\nPack resistance bands before drills.\n",
    )
    .expect("older semantic memory should be written");

    let mut embedding_server = mockito::Server::new();
    let _query_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _semantic_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_embedding_mock = embedding_server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Review the workout locker spreadsheet".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.0, 1.0]}]
            })
            .to_string(),
        )
        .create();

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir.clone();
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    for index in 0..120 {
        connection
            .execute(
                "UPDATE memories SET updated_at_unix = ?1 WHERE file_path = ?2",
                rusqlite::params![
                    10_000_i64 - index as i64,
                    memory_dir
                        .join(format!("recent_memory_{index:02}.md"))
                        .display()
                        .to_string(),
                ],
            )
            .expect("recent memory timestamp should update");
    }
    connection
        .execute(
            "UPDATE memories SET updated_at_unix = 1 WHERE file_path = ?1",
            rusqlite::params![memory_dir.join("resistance_bands.md").display().to_string(),],
        )
        .expect("older memory timestamp should update");

    let embedding_client = best_effort_embedding_client(Some(&EmbeddingProviderConfig {
        model: "text-embedding-3-small".to_string(),
        api_key: "embedding-test-key".to_string(),
        base_url: format!("{}/embeddings", embedding_server.url()),
        timeout_seconds: 60,
    }))
    .expect("embedding client should build");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What belongs in my workout kit?",
        &OlderSemanticEmbeddingOnlyPromptModel,
        RecallModelClients {
            classifier_model: None,
            embedding_client: Some(&embedding_client),
            embedding_distance_threshold: Some(
                config.l2_memory_relevance_distance_threshold as f32,
            ),
            recency_weight: config.recency_weight as f32,
            reflection_max_words: config.memory_reflection_max_words,
        },
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: false,
            memory_recall_classifier_window: 3,
            reflect: false,
        },
    )
    .expect("prompt should succeed");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_can_prefer_semantic_due_and_agenda_recall_candidates_over_weaker_overlap()
 {
    struct MixedSemanticRecallPromptModel;
    struct MixedSemanticRecallRelevanceModel;

    impl ModelClient for MixedSemanticRecallPromptModel {
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
            assert!(
                recall_payload.contains("bands reminder"),
                "{recall_payload}"
            );
            assert!(recall_payload.contains("bands plan"), "{recall_payload}");
            assert!(
                !recall_payload.contains("gear inventory reminder"),
                "{recall_payload}"
            );
            assert!(
                !recall_payload.contains("gear inventory plan"),
                "{recall_payload}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Bring the resistance bands and check the bands plan.".to_string(),
            }])
        }
    }

    impl ModelClient for MixedSemanticRecallRelevanceModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            if prompt.contains("needs_recall") {
                return Ok(vec![StreamEvent::AssistantResponse {
                        content: r#"{"needs_recall":true,"reasoning":"The user is asking about practice gear."}"#
                            .to_string(),
                    }]);
            }
            let answers = if prompt.contains("bands reminder")
                && prompt.contains("gear inventory reminder")
            {
                if prompt.find("bands reminder") < prompt.find("gear inventory reminder") {
                    vec![true, false]
                } else {
                    vec![false, true]
                }
            } else if prompt.contains("bands plan") && prompt.contains("gear inventory plan") {
                if prompt.find("bands plan") < prompt.find("gear inventory plan") {
                    vec![true, false]
                } else {
                    vec![false, true]
                }
            } else {
                vec![true]
            };
            Ok(vec![StreamEvent::AssistantResponse {
                content: serde_json::json!({
                    "answers": answers,
                    "reasoning": "Only the resistance-bands candidates match the user's intent."
                })
                .to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-recall-semantic-mixed-priority-{}",
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
    config.agenda_dir = agenda_dir.clone();
    config.database_path = database_path.clone();

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    fs::write(
            agenda_dir.join("gear_inventory_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: after equipment handoff\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap due item should be written");
    fs::write(
            agenda_dir.join("bands_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: before scrimmage\n---\n\nCarry resistance bands in the trunk.\n",
        )
        .expect("semantic due item should be written");
    fs::write(
            agenda_dir.join("gear_inventory_plan.md"),
            "---\ndate: 2026-05-20\ncompleted: false\nstatus: created\n---\n\nReview the storage locker spreadsheet.\n",
        )
        .expect("overlap agenda item should be written");
    fs::write(
            agenda_dir.join("bands_plan.md"),
            "---\ndate: 2026-05-21\ncompleted: false\nstatus: created\n---\n\nPack resistance bands before drills.\n",
        )
        .expect("semantic agenda item should be written");
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What gear should I bring to practice?",
        &MixedSemanticRecallPromptModel,
        recall_model_clients(Some(&MixedSemanticRecallRelevanceModel)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
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
fn run_prompt_with_model_and_registry_can_inject_model_authored_reflective_recall() {
    struct ReflectiveRuntimePromptModel;

    impl ModelClient for ReflectiveRuntimePromptModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(
                request.user_message,
                "What should I remember before basketball practice?"
            );
            let recall_payload = request
                .transcript
                .iter()
                .find(|message| {
                    message.role == MessageRole::Tool
                        && message.tool_call_id.as_deref() == Some("bootstrap-memory-recall")
                })
                .and_then(|message| message.content.as_deref())
                .expect("reflective recall payload should be injected");
            assert!(
                recall_payload.contains(
                    "I remember that the user should bring the resistance bands to practice."
                ),
                "{recall_payload}"
            );
            assert!(
                !recall_payload.contains("I remember these memory details may be relevant"),
                "{recall_payload}"
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Bring the resistance bands to practice.".to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-reflective-runtime-authored-{}",
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
        "# Basketball Form\n\nBring the resistance bands to practice.\n",
    )
    .expect("memory file should be written");

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let reflective_model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The recalled memory is relevant."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":true,"content":"I remember that the user should bring the resistance bands to practice."}"#.to_string(),
            }],
        ]);

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What should I remember before basketball practice?",
        &ReflectiveRuntimePromptModel,
        recall_model_clients(Some(&reflective_model)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: false,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: true,
        },
    )
    .expect("prompt should succeed");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("resistance bands")
    )));

    fs::remove_dir_all(home).expect("home should be removed");
}

#[test]
fn run_prompt_with_model_and_registry_can_suppress_irrelevant_reflective_recall() {
    struct ReflectiveSuppressionPromptModel;

    impl ModelClient for ReflectiveSuppressionPromptModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            assert_eq!(
                request.user_message,
                "What should I remember before basketball practice?"
            );
            assert!(
                request.transcript.iter().all(|message| {
                    message.tool_call_id.as_deref() != Some("bootstrap-memory-recall")
                }),
                "{:#?}",
                request.transcript
            );
            Ok(vec![StreamEvent::AssistantResponse {
                content: "No prior reminder looks especially relevant.".to_string(),
            }])
        }
    }

    let unique = format!(
        "elroy-rs-app-reflective-runtime-suppression-{}",
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

    let mut config = AppConfig::defaults();
    config.home_dir = home.clone();
    config.memory_dir = memory_dir;
    config.agenda_dir = agenda_dir;
    config.database_path = database_path.clone();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
        .expect("bootstrap should succeed");

    let reflective_model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The recalled memory is relevant enough to inspect."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":false,"content":null}"#.to_string(),
            }],
        ]);

    let mut connection = open_sqlite_connection(&database_path).expect("database should open");
    run_migrations(&mut connection).expect("migrations should run");
    let events = run_prompt_with_model_and_registry_internal(
        &mut connection,
        "What should I remember before basketball practice?",
        &ReflectiveSuppressionPromptModel,
        recall_model_clients(Some(&reflective_model)),
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
            memory_consolidation_settings: Some(
                crate::memory_consolidation_settings_from_app_config(&config),
            ),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: false,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: true,
        },
    )
    .expect("prompt should succeed");

    assert!(!events.iter().any(|event| matches!(
        event,
        StreamEvent::StatusUpdate { content } if content == "fetching memories..."
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::AssistantResponse { content }
            if content.contains("No prior reminder")
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

    let recalled =
        select_recalled_memories("I am heading to basketball practice", &memories, &[], 3);

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
    assert!(payload.contains("The latest user message is: I am heading to basketball practice"));
}

#[test]
fn reflective_recall_uses_model_authored_content_when_available() {
    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The recalled memory is relevant."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The due item is relevant."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":true,"content":"I remember that the user should bring the resistance bands to practice."}"#.to_string(),
            }],
        ]);
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
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
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
        payload.contains("I remember that the user should bring the resistance bands to practice.")
    );
    assert!(!payload.contains("I remember these memory details may be relevant"));
}

#[test]
fn reflective_recall_can_be_suppressed_when_model_marks_it_irrelevant() {
    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The recalled memory is relevant enough to inspect."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":false,"content":null}"#.to_string(),
            }],
        ]);

    let messages = recall_memory_context_messages_with_decision(
        config.memory_recall_classifier_window,
        config.reflect,
        "I am heading to basketball practice",
        true,
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
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
fn reflective_recall_prompt_uses_configured_word_limit() {
    struct ReflectivePromptInspectionModel {
        prompt: Arc<Mutex<Option<String>>>,
    }

    impl ModelClient for ReflectivePromptInspectionModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            *self.prompt.lock().expect("prompt lock should succeed") =
                Some(request.user_message.to_string());
            Ok(vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"is_relevant":true,"content":"I should remind the user about payroll."}"#
                        .to_string(),
            }])
        }
    }

    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    config.memory_reflection_max_words = 42;
    let captured_prompt = Arc::new(Mutex::new(None));
    let model = ReflectivePromptInspectionModel {
        prompt: Arc::clone(&captured_prompt),
    };

    let messages = recall_memory_context_messages_with_decision(
        config.memory_recall_classifier_window,
        config.reflect,
        "What should I remember about payroll?",
        true,
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        RecallContext {
            transcript: &[],
            memories: &[MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "payroll".to_string(),
                file_path: "/tmp/payroll.md".to_string(),
                body: "Remember to finish payroll before Friday.".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }],
            due_items: &[],
            agenda_items: &[],
        },
    );

    assert_eq!(messages.len(), 2);
    let prompt = captured_prompt
        .lock()
        .expect("prompt lock should succeed")
        .clone()
        .expect("reflective recall prompt should be captured");
    assert!(prompt.contains("no more than 42 words."), "{}", prompt);
}

#[test]
fn reflective_recall_truncates_model_authored_content_to_configured_word_limit() {
    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    config.memory_reflection_max_words = 5;
    let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"The recalled memory is relevant enough to inspect."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":true,"content":"one two three four five six seven"}"#
                    .to_string(),
            }],
        ]);

    let messages = recall_memory_context_messages_with_decision(
        config.memory_recall_classifier_window,
        config.reflect,
        "What should I remember about payroll?",
        true,
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        RecallContext {
            transcript: &[],
            memories: &[MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "payroll".to_string(),
                file_path: "/tmp/payroll.md".to_string(),
                body: "Remember to finish payroll before Friday.".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }],
            due_items: &[],
            agenda_items: &[],
        },
    );

    let payload = messages[1]
        .content
        .as_deref()
        .expect("reflective recall payload should exist");
    assert!(payload.contains("one two three four five"));
    assert!(!payload.contains("six seven"));
}

#[test]
fn reflective_recall_truncates_fallback_content_to_configured_word_limit() {
    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    config.memory_reflection_max_words = 6;

    let messages = recall_memory_context_messages_with_decision(
        config.memory_recall_classifier_window,
        config.reflect,
        "Should I ask about the marathon date?",
        true,
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: None,
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        RecallContext {
            transcript: &[],
            memories: &[MemoryRecord {
                id: 1,
                legacy_frontmatter_id: None,
                name: "marathon".to_string(),
                file_path: "/tmp/marathon.md".to_string(),
                body: "User wants to train for a marathon and asked for encouragement.".to_string(),
                is_active: true,
                updated_at_unix: 10,
            }],
            due_items: &[],
            agenda_items: &[],
        },
    );

    let payload = messages[1]
        .content
        .as_deref()
        .expect("reflective recall payload should exist");
    let parsed = serde_json::from_str::<serde_json::Value>(payload).expect("payload should parse");
    let content = parsed
        .get("content")
        .and_then(|value| value.as_str())
        .expect("reflective recall content should exist");
    assert_eq!(content.split_whitespace().count(), 6);
}

#[test]
fn reflective_recall_can_broaden_candidates_beyond_overlap() {
    let mut config = AppConfig::defaults();
    config.memory_recall_classifier_enabled = false;
    config.reflect = true;
    let model = FakeModel::new(vec![
            vec![StreamEvent::AssistantResponse {
                content: r#"{"answers":[true],"reasoning":"This due item is semantically relevant."}"#
                    .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content:
                    r#"{"answers":[true],"reasoning":"This agenda item is semantically relevant."}"#
                        .to_string(),
            }],
            vec![StreamEvent::AssistantResponse {
                content: r#"{"is_relevant":true,"content":"I remember that the user should bring the resistance bands and review the equipment checklist before practice."}"#.to_string(),
            }],
        ]);
    let messages = recall_memory_context_messages_with_decision(
        config.memory_recall_classifier_window,
        config.reflect,
        "What gear should I bring to practice?",
        true,
        config.memory_reflection_max_words,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        RecallContext {
            transcript: &[],
            memories: &[],
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
                name: "packing checklist".to_string(),
                file_path: "/tmp/packing_checklist.md".to_string(),
                agenda_date: Some("2026-05-20".to_string()),
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
            }],
        },
    );

    assert_eq!(messages.len(), 2);
    let payload = messages[1]
        .content
        .as_deref()
        .expect("reflective recall payload should exist");
    assert!(payload.contains("practice reminder"));
    assert!(payload.contains("packing checklist"));
    assert!(payload.contains("resistance bands"));
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
fn recall_memory_context_messages_do_not_skip_same_name_memory_with_different_id() {
    let transcript = synthetic_tool_context_messages(
        "bootstrap-memory-recall",
        "get_fast_recall",
        "{}",
        r##"{
  "content": "# practice plan\nOld warmup advice",
  "recall_metadata": [
    {
      "memory_type": "Memory",
      "memory_id": 1,
      "name": "practice plan"
    }
  ]
}"##,
    );

    let messages = recall_memory_context_messages(
        false,
        3,
        false,
        "What should I do before basketball drills?",
        RecallContext {
            transcript: &transcript,
            memories: &[MemoryRecord {
                id: 2,
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
    assert!(
        messages[1]
            .content
            .as_deref()
            .is_some_and(|content| content.contains("Warm up before basketball drills"))
    );
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

    let parsed = parse_recalled_item_refs(
        r#"[{"name":"Basketball Form"},{"name":"Sleep Routine"}]"#,
        "Memory",
    );
    let names = recalled_memory_names(&transcript);

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].id, None);
    assert!(names.contains("basketball form"));
    assert!(names.contains("sleep routine"));
}

#[test]
fn parse_reflective_recall_metadata_names_from_transcript() {
    let transcript = vec![ConversationMessage::tool_result(
        "bootstrap-memory-recall",
        r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"Memory","memory_id":2,"name":"Sleep Routine"}]}"#,
    )];

    let parsed = parse_recalled_item_refs(
        r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"Memory","memory_id":2,"name":"Sleep Routine"}]}"#,
        "Memory",
    );
    let names = recalled_memory_names(&transcript);

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].id, Some(1));
    assert_eq!(parsed[1].id, Some(2));
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

    let parsed = parse_recalled_item_refs(
        r#"{"content":"I remember these details may be relevant.","recall_metadata":[{"memory_type":"Memory","memory_id":1,"name":"Basketball Form"},{"memory_type":"AgendaItem","memory_id":9,"name":"Payroll Followup"},{"memory_type":"AgendaItem","memory_id":10,"name":"Drill Plan"}]}"#,
        "AgendaItem",
    );
    let names = recalled_item_refs_by_type(&transcript, "AgendaItem")
        .into_iter()
        .map(|item| item.name)
        .collect::<HashSet<_>>();

    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].id, Some(9));
    assert_eq!(parsed[1].id, Some(10));
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
    let already_recalled = vec![RecalledItemRef {
        id: None,
        name: "basketball form".to_string(),
    }];

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
        &[],
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_due_items = select_relevant_recall_due_items(
        "What workout gear should I bring?",
        &due_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_agenda_items = select_relevant_recall_agenda_items(
        "What workout gear should I bring?",
        &agenda_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
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
            content: r#"{"answers":[true],"reasoning":"This agenda item matches semantically."}"#
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
        &[],
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_due_items = select_relevant_recall_due_items(
        "What gear should I bring to practice?",
        &due_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_agenda_items = select_relevant_recall_agenda_items(
        "What gear should I bring to practice?",
        &agenda_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );

    assert_eq!(relevant_memories.len(), 1);
    assert_eq!(relevant_memories[0].name, "practice gear");
    assert_eq!(relevant_due_items.len(), 1);
    assert_eq!(relevant_due_items[0].name, "bring backup cleats");
    assert_eq!(relevant_agenda_items.len(), 1);
    assert_eq!(relevant_agenda_items[0].name, "practice packing list");
}

#[test]
fn select_relevant_recall_helpers_can_prefer_semantic_matches_over_weaker_overlap_candidates() {
    struct SemanticPriorityRelevanceModel;

    impl ModelClient for SemanticPriorityRelevanceModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, elroy_core::ModelClientError> {
            let prompt = request.user_message;
            let answers = if prompt.contains("resistance bands")
                && prompt.contains("storage locker spreadsheet")
            {
                if prompt.find("resistance bands") < prompt.find("storage locker spreadsheet") {
                    vec![true, false]
                } else {
                    vec![false, true]
                }
            } else if prompt.contains("Carry resistance bands in the trunk.")
                && prompt.contains("Review the storage locker spreadsheet.")
            {
                if prompt.find("Carry resistance bands in the trunk.")
                    < prompt.find("Review the storage locker spreadsheet.")
                {
                    vec![true, false]
                } else {
                    vec![false, true]
                }
            } else if prompt.contains("Pack resistance bands before drills.")
                && prompt.contains("Review the storage locker spreadsheet.")
            {
                if prompt.find("Pack resistance bands before drills.")
                    < prompt.find("Review the storage locker spreadsheet.")
                {
                    vec![true, false]
                } else {
                    vec![false, true]
                }
            } else {
                panic!("unexpected relevance prompt: {prompt}");
            };
            Ok(vec![StreamEvent::AssistantResponse {
                content: serde_json::json!({
                    "answers": answers,
                    "reasoning": "Only the semantic candidate matches the user's intent."
                })
                .to_string(),
            }])
        }
    }

    let memories = vec![
        MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "Gear Inventory".to_string(),
            file_path: "/tmp/gear_inventory.md".to_string(),
            body: "Review the storage locker spreadsheet.".to_string(),
            is_active: true,
            updated_at_unix: 10,
        },
        MemoryRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "Training Kit".to_string(),
            file_path: "/tmp/practice_gear.md".to_string(),
            body: "Pack resistance bands before drills.".to_string(),
            is_active: true,
            updated_at_unix: 20,
        },
    ];
    let due_items = vec![
        AgendaItemRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "Gear Inventory Reminder".to_string(),
            file_path: "/tmp/gear_inventory_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Review the storage locker spreadsheet.".to_string(),
            trigger_datetime: None,
            trigger_context: Some("after equipment handoff".to_string()),
            is_active: true,
            updated_at_unix: 11,
        },
        AgendaItemRecord {
            id: 4,
            legacy_frontmatter_id: None,
            name: "Locker Reminder".to_string(),
            file_path: "/tmp/practice_reminder.md".to_string(),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Carry resistance bands in the trunk.".to_string(),
            trigger_datetime: None,
            trigger_context: Some("before scrimmage".to_string()),
            is_active: true,
            updated_at_unix: 21,
        },
    ];
    let agenda_items = vec![
        AgendaItemRecord {
            id: 5,
            legacy_frontmatter_id: None,
            name: "Gear Inventory Agenda".to_string(),
            file_path: "/tmp/gear_inventory_agenda.md".to_string(),
            agenda_date: Some("2026-05-21".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Review the storage locker spreadsheet.".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 12,
        },
        AgendaItemRecord {
            id: 6,
            legacy_frontmatter_id: None,
            name: "Training Kit List".to_string(),
            file_path: "/tmp/practice_packing_list.md".to_string(),
            agenda_date: Some("2026-05-22".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: "Pack resistance bands before drills.".to_string(),
            trigger_datetime: None,
            trigger_context: None,
            is_active: true,
            updated_at_unix: 22,
        },
    ];

    let relevant_memories = select_relevant_recall_memories(
        "What gear should I bring to practice?",
        &memories,
        &[],
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&SemanticPriorityRelevanceModel),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_due_items = select_relevant_recall_due_items(
        "What gear should I bring to practice?",
        &due_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&SemanticPriorityRelevanceModel),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let relevant_agenda_items = select_relevant_recall_agenda_items(
        "What gear should I bring to practice?",
        &agenda_items,
        RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&SemanticPriorityRelevanceModel),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );

    assert_eq!(relevant_memories.len(), 1);
    assert_eq!(relevant_memories[0].name, "Training Kit");
    assert_eq!(relevant_due_items.len(), 1);
    assert_eq!(relevant_due_items[0].name, "Locker Reminder");
    assert_eq!(relevant_agenda_items.len(), 1);
    assert_eq!(relevant_agenda_items[0].name, "Training Kit List");
}

#[test]
fn select_relevant_recall_memories_can_prefer_newer_semantic_matches_when_recency_weight_is_enabled()
 {
    let now = Utc::now().timestamp();
    let memories = vec![
        MemoryRecord {
            id: 1,
            legacy_frontmatter_id: None,
            name: "stale exact match".to_string(),
            file_path: "/tmp/stale_exact_match.md".to_string(),
            body: "Pack resistance bands before drills.".to_string(),
            is_active: true,
            updated_at_unix: now - (5 * 365 * 24 * 60 * 60),
        },
        MemoryRecord {
            id: 2,
            legacy_frontmatter_id: None,
            name: "recent warmup note".to_string(),
            file_path: "/tmp/recent_warmup_note.md".to_string(),
            body: "Bring resistance bands for warmups.".to_string(),
            is_active: true,
            updated_at_unix: now - (2 * 24 * 60 * 60),
        },
        MemoryRecord {
            id: 3,
            legacy_frontmatter_id: None,
            name: "recent practice note".to_string(),
            file_path: "/tmp/recent_practice_note.md".to_string(),
            body: "Bring resistance bands and cones to practice.".to_string(),
            is_active: true,
            updated_at_unix: now - (24 * 60 * 60),
        },
    ];

    let mut server = mockito::Server::new();
    let _query_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::PartialJson(serde_json::json!({
            "input": "What belongs in my workout kit?"
        })))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _stale_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Pack resistance bands before drills".to_string(),
        ))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [1.0, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_warmup_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Bring resistance bands for warmups".to_string(),
        ))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.92, 0.0]}]
            })
            .to_string(),
        )
        .create();
    let _recent_practice_embedding_mock = server
        .mock("POST", "/embeddings")
        .match_body(mockito::Matcher::Regex(
            "Bring resistance bands and cones to practice".to_string(),
        ))
        .expect(2)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({
                "data": [{"embedding": [0.93, 0.0]}]
            })
            .to_string(),
        )
        .create();

    let embedding_client = best_effort_embedding_client(Some(&EmbeddingProviderConfig {
        model: "text-embedding-3-small".to_string(),
        api_key: "embedding-test-key".to_string(),
        base_url: format!("{}/embeddings", server.url()),
        timeout_seconds: 60,
    }))
    .expect("embedding client should build");

    let without_recency_weight = select_relevant_recall_memories(
        "What belongs in my workout kit?",
        &memories,
        &[],
        RecallSelectionClients {
            limit: 2,
            relevance_model: None,
            embedding_client: Some(&embedding_client),
            embedding_distance_threshold: Some(0.5),
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );
    let with_recency_weight = select_relevant_recall_memories(
        "What belongs in my workout kit?",
        &memories,
        &[],
        RecallSelectionClients {
            limit: 2,
            relevance_model: None,
            embedding_client: Some(&embedding_client),
            embedding_distance_threshold: Some(0.5),
            recency_weight: 0.1,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
    );

    assert_eq!(without_recency_weight.len(), 2);
    assert_eq!(without_recency_weight[0].name, "stale exact match");
    assert_eq!(without_recency_weight[1].name, "recent practice note");

    assert_eq!(with_recency_weight.len(), 2);
    assert_eq!(with_recency_weight[0].name, "recent practice note");
    assert_eq!(with_recency_weight[1].name, "recent warmup note");
}

#[test]
fn recall_memory_context_messages_can_surface_older_semantic_memory_candidate_beyond_old_candidate_cap()
 {
    let transcript = vec![ConversationMessage::new(
        MessageRole::Assistant,
        "What should I remember for practice?",
    )];
    let mut memories = (0..40)
        .map(|index| MemoryRecord {
            id: index + 1,
            legacy_frontmatter_id: None,
            name: format!("Recent Memory {index}"),
            file_path: format!("/tmp/recent_memory_{index}.md"),
            body: format!("Review the workout locker spreadsheet {index}."),
            is_active: true,
            updated_at_unix: 100 - index,
        })
        .collect::<Vec<_>>();
    memories.push(MemoryRecord {
        id: 99,
        legacy_frontmatter_id: None,
        name: "Resistance Bands".to_string(),
        file_path: "/tmp/resistance_bands.md".to_string(),
        body: "Pack resistance bands before drills.".to_string(),
        is_active: true,
        updated_at_unix: 1,
    });
    let mut answers = vec![false; 41];
    *answers.last_mut().expect("answers should exist") = true;
    let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
        content: serde_json::json!({
            "answers": answers,
            "reasoning": "Only the older resistance-bands memory is semantically relevant."
        })
        .to_string(),
    }]]);

    let messages = recall_memory_context_messages_with_decision(
        3,
        false,
        "What gear should I bring to practice?",
        true,
        100,
        crate::RecallSelectionClients {
            limit: 2,
            relevance_model: Some(&model),
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        RecallContext {
            transcript: &transcript,
            memories: &memories,
            due_items: &[],
            agenda_items: &[],
        },
    );

    let joined_tool_content = messages
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .filter_map(|message| message.content.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined_tool_content.contains("Resistance Bands"));
    assert!(joined_tool_content.contains("resistance bands"));
    assert!(!joined_tool_content.contains("Recent Memory 0"));
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
        let record =
            elroy_codex::get_codex_session_by_thread_id(&connection, LOCAL_USER_TOKEN, session_id)
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

fn wait_for_background_status_key_message(status_key: &str, expected_status: &str) {
    let started = Instant::now();
    loop {
        if get_background_status_for_key(status_key).as_deref() == Some(expected_status) {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "timed out waiting for background status {expected_status}, last status {:?}",
            get_background_status_for_key(status_key)
        );
        thread::sleep(Duration::from_millis(25));
    }
}
