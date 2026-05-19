// Memory recall, candidate selection, reflective recall, and consolidation orchestration.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use chrono::{Local, Utc};
use elroy_core::memory_store::{
    archive_memory_file, create_memory_file_with_frontmatter, read_memory_parts,
};
use elroy_core::{ConversationRequest, LiveProviderModel, ModelClient, excerpt};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, LOCAL_USER_TOKEN, MemoryEmbeddingRecord, MemoryRecord,
    get_or_create_memory_operation_tracker, list_all_active_memories, load_context_messages,
    load_memory_embeddings_for_paths, load_messages_by_ids, open_sqlite_connection,
    replace_context_messages, run_migrations, save_memory_operation_tracker,
    search_active_memories, upsert_memory_embedding,
};
use elroy_llm::{
    ConversationMessage, EmbeddingProviderConfig, LiveEmbeddingClient, LiveModelClient,
    MessageRole, ProviderConfig, StreamEvent, ToolCall,
};
use elroy_user::effective_persona;
use serde_json::Value;

use elroy_config::{
    AppConfig, LlmProvider, embedding_provider_config_from_app_config,
    fast_provider_config_from_app_config,
};
use elroy_context::load_validated_runtime_transcript;
use elroy_tools::ToolExecutionResult;

pub const CONTEXT_MESSAGE_SOURCE_TYPE: &str = "ContextMessageSet";
pub const MEMORY_SOURCE_TYPE: &str = "Memory";
pub const MEMORY_WORD_COUNT_LIMIT: usize = 300;
pub const MEMORY_CONSOLIDATION_CLUSTER_LIMIT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySourceType {
    ContextMessageSet,
    Memory,
}

pub struct MemorySource {
    pub name: String,
    pub source_type: MemorySourceType,
    pub message_ids: Vec<i64>,
    pub path: Option<String>,
}

pub fn get_source_list_from_frontmatter(frontmatter: Option<&str>) -> Vec<MemorySource> {
    let mut sources = Vec::new();
    if let Some(message_ids) = parse_context_message_source_ids(frontmatter) {
        sources.push(MemorySource {
            name: "Context messages".to_string(),
            source_type: MemorySourceType::ContextMessageSet,
            message_ids,
            path: None,
        });
    }
    if let Some(memory_sources) = parse_memory_sources(frontmatter) {
        for (name, path) in memory_sources {
            sources.push(MemorySource {
                name,
                source_type: MemorySourceType::Memory,
                message_ids: Vec::new(),
                path: Some(path),
            });
        }
    }
    sources
}

pub fn memory_consolidation_settings_from_app_config(
    config: &AppConfig,
) -> MemoryConsolidationSettings {
    MemoryConsolidationSettings {
        memory_cluster_similarity_threshold: config.memory_cluster_similarity_threshold,
        max_memory_cluster_size: config.max_memory_cluster_size,
        min_memory_cluster_size: config.min_memory_cluster_size,
        fast_provider_config: fast_provider_config_from_app_config(config).ok(),
        embedding_provider_config: embedding_provider_config_from_app_config(config).ok(),
    }
}
pub fn create_consolidated_memory_from_config(
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

pub fn create_consolidated_memory_from_plan(
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

    create_consolidated_memories_from_records(
        bootstrap_plan,
        &[ConsolidatedMemoryOutput {
            name: name.to_string(),
            text: text.to_string(),
        }],
        &source_memories,
    )
    .map(|mut created| created.remove(0))
}
pub fn sync_memory_context_after_mutation(
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
    let old_tool_call_id = context_memory_tool_call_id(old_name);
    let mut updated_transcript = transcript
        .into_iter()
        .filter(|message| !message_matches_tool_call_id(message, &old_tool_call_id))
        .collect::<Vec<_>>();

    if let Some(current_name) = current_name
        && let Some(memory) =
            find_active_memory_by_name_in_scope(&connection, current_name, &config.memory_dir)?
        && !transcript_contains_context_memory(&updated_transcript, &memory.name)
    {
        updated_transcript.extend(context_memory_tool_messages(&memory));
    }

    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

pub fn sync_task_context_after_mutation(
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
        && let Some(task) = elroy_db::find_active_agenda_item_by_name(&connection, current_name)?
            .filter(|item| item.trigger_datetime.is_none() && item.trigger_context.is_none())
        && !transcript_contains_context_task(&updated_transcript, &task.name)
    {
        updated_transcript.extend(context_task_tool_messages(&task));
    }

    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

pub fn sync_due_item_context_after_mutation(
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
    let old_tool_call_id = context_due_item_tool_call_id(old_name);
    let mut updated_transcript = transcript
        .into_iter()
        .filter(|message| !message_matches_tool_call_id(message, &old_tool_call_id))
        .collect::<Vec<_>>();

    if let Some(current_name) = current_name
        && let Some(due_item) =
            elroy_db::find_active_agenda_item_by_name(&connection, current_name)?
                .filter(|item| item.trigger_datetime.is_some() || item.trigger_context.is_some())
        && !transcript_contains_context_due_item(&updated_transcript, &due_item.name)
    {
        updated_transcript.extend(context_due_item_tool_messages(&due_item));
    }

    replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

pub fn mutate_memory_file_from_config(
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
        sync_memory_context_after_mutation(config, &memory.name, Some(&memory.name))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(())
    }) {
        Ok(()) => ToolExecutionResult::success(
            serde_json::json!({"updated": true, "name": memory.name, "file_path": memory.file_path})
                .to_string(),
        ),
        Err(error) => ToolExecutionResult::error(format!("memory mutation failed: {error}")),
    }
}

pub fn archive_memory_file_from_config(
    config: &AppConfig,
    name: &str,
    operation: impl FnOnce(&Path) -> std::io::Result<PathBuf>,
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
    match operation(Path::new(&memory.file_path)).and_then(|new_path| {
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        remove_context_memory_messages_by_names(
            &mut connection,
            std::slice::from_ref(&memory.name),
        )
        .map_err(|error| std::io::Error::other(error.to_string()))?;
        Ok(new_path)
    }) {
        Ok(new_path) => ToolExecutionResult::success(
            serde_json::json!({
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
pub fn find_active_memory_by_name_in_scope(
    connection: &rusqlite::Connection,
    name: &str,
    memory_dir: &Path,
) -> rusqlite::Result<Option<MemoryRecord>> {
    Ok(
        list_active_memories_in_scope(connection, memory_dir, 10_000)?
            .into_iter()
            .find(|memory| memory.name.eq_ignore_ascii_case(name)),
    )
}

pub fn list_active_memories_in_scope(
    connection: &rusqlite::Connection,
    memory_dir: &Path,
    limit: usize,
) -> rusqlite::Result<Vec<MemoryRecord>> {
    Ok(list_all_active_memories_in_scope(connection, memory_dir)?
        .into_iter()
        .take(limit)
        .collect())
}

pub fn search_active_memories_in_scope(
    connection: &rusqlite::Connection,
    memory_dir: &Path,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<MemoryRecord>> {
    Ok(search_active_memories(connection, query, 10_000)?
        .into_iter()
        .filter(|memory| Path::new(&memory.file_path).starts_with(memory_dir))
        .take(limit)
        .collect())
}

pub fn format_memory_listing(memories: &[MemoryRecord]) -> String {
    if memories.is_empty() {
        return "No memories found.".to_string();
    }

    let mut lines = vec!["Memories".to_string()];
    for memory in memories.iter().rev() {
        lines.push(format!(
            "- {} | Text: {}",
            memory.name.replace('_', " "),
            excerpt(&memory.body, 180)
        ));
    }
    lines.join("\n")
}

pub fn get_source_list_for_memory_from_config(
    config: &AppConfig,
    name: &str,
) -> anyhow::Result<String> {
    let connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut open_sqlite_connection(&config.database_path)?)?;
    let memory = find_active_memory_by_name_in_scope(&connection, name, &config.memory_dir)?
        .ok_or_else(|| anyhow::anyhow!("Memory '{name}' not found for the current user."))?;

    let (frontmatter, _) = read_memory_parts(Path::new(&memory.file_path))?;
    let sources = get_source_list_from_frontmatter(frontmatter.as_deref());

    let payload = sources
        .into_iter()
        .map(|source| {
            let source_type = match source.source_type {
                MemorySourceType::ContextMessageSet => CONTEXT_MESSAGE_SOURCE_TYPE,
                MemorySourceType::Memory => MEMORY_SOURCE_TYPE,
            };
            let id = if source.message_ids.is_empty() {
                source.name
            } else {
                source.message_ids[0].to_string() // Simplified for now as per previous behavior
            };
            serde_json::json!([source_type, id])
        })
        .collect::<Vec<_>>();

    Ok(serde_json::to_string(&payload)?)
}

pub fn get_source_content_for_memory_from_config(
    config: &AppConfig,
    name: &str,
    index: usize,
) -> anyhow::Result<String> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let memory = find_active_memory_by_name_in_scope(&connection, name, &config.memory_dir)?
        .ok_or_else(|| anyhow::anyhow!("Memory '{name}' not found for the current user."))?;

    let (frontmatter, _body) = read_memory_parts(Path::new(&memory.file_path))?;
    let sources = get_source_list_from_frontmatter(frontmatter.as_deref());

    if sources.is_empty() {
        return Ok(format!(
            "No sources found for memory '{}'",
            name.to_lowercase()
        ));
    }

    if index >= sources.len() {
        anyhow::bail!(
            "Index {index} out of range. Available indices: {:?}",
            (0..sources.len()).collect::<Vec<_>>()
        );
    }

    let source = &sources[index];
    match source.source_type {
        MemorySourceType::ContextMessageSet => {
            let messages = load_messages_by_ids(&connection, &source.message_ids)?;
            Ok(format_context_message_source_content(&messages))
        }
        MemorySourceType::Memory => {
            let source_path = source
                .path
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("missing path for memory source"))?;
            let source_body = if let Ok(_content) = std::fs::read_to_string(source_path) {
                let (_, body) = read_memory_parts(Path::new(source_path))?;
                body
            } else {
                format!(
                    "Error: Unable to read source memory file for '{}'",
                    source.name
                )
            };
            Ok(format_memory_file_source_content(
                &source.name,
                &source_body,
            ))
        }
    }
}

// ── helper functions (moved from elroy-app) ───────────────────────────────────

pub fn format_memory_detail(memory: &elroy_db::MemoryRecord) -> String {
    format!("#{}\n{}", memory.name, memory.body)
}

pub fn format_due_item_detail(item: &elroy_db::AgendaItemRecord) -> String {
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

pub fn format_agenda_item_detail(item: &elroy_db::AgendaItemRecord) -> String {
    let mut text = format!("# Agenda Item: {}\n\n{}", item.name, item.body.trim());
    if let Some(agenda_date) = item.agenda_date.as_deref() {
        text.push_str(&format!("\n\nAgenda date: {agenda_date}"));
    }
    text
}

pub fn format_memory_examination(memory: &MemoryRecord) -> String {
    format!(
        "# Memory: {}\n\n*to view the source content this memory is based on, call tool `get_source_content_for_memory({}, idx)`\n\n{}",
        memory.name,
        memory.name,
        memory.body.trim()
    )
}

pub fn format_memory_search_results(
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

pub fn format_agenda_item_recall_detail(item: &AgendaItemRecord) -> String {
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

pub fn synthetic_tool_context_messages(
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

pub fn context_memory_tool_call_id(name: &str) -> String {
    format!("context-memory:{}", name.to_ascii_lowercase())
}

pub fn context_due_item_tool_call_id(name: &str) -> String {
    format!("context-due-item:{}", name.to_ascii_lowercase())
}

pub fn context_task_tool_call_id(name: &str) -> String {
    format!("context-task:{}", name.to_ascii_lowercase())
}

pub fn context_due_item_tool_messages(item: &AgendaItemRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&serde_json::json!({
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

pub fn context_task_tool_messages(item: &AgendaItemRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&serde_json::json!({
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

pub fn context_memory_tool_messages(memory: &elroy_db::MemoryRecord) -> Vec<ConversationMessage> {
    let content = serde_json::to_string_pretty(&serde_json::json!({
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

pub fn parse_sidebar_trigger_datetime(value: &str) -> Option<chrono::NaiveDateTime> {
    chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M").ok())
}

// ── types ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
pub struct RecallModelClients<'a> {
    pub classifier_model: Option<&'a dyn ModelClient>,
    pub embedding_client: Option<&'a LiveEmbeddingClient>,
    pub embedding_distance_threshold: Option<f32>,
    pub recency_weight: f32,
    pub reflection_max_words: usize,
}

pub fn recall_model_clients(classifier_model: Option<&dyn ModelClient>) -> RecallModelClients<'_> {
    RecallModelClients {
        classifier_model,
        embedding_client: None,
        embedding_distance_threshold: None,
        recency_weight: 0.0,
        reflection_max_words: 100,
    }
}

pub struct RecallContext<'a> {
    pub transcript: &'a [ConversationMessage],
    pub memories: &'a [MemoryRecord],
    pub due_items: &'a [AgendaItemRecord],
    pub agenda_items: &'a [AgendaItemRecord],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecallDecision {
    pub needs_recall: bool,
    pub reasoning: String,
    pub used_llm: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecalledItemRef {
    pub id: Option<i64>,
    pub name: String,
}

#[derive(Clone, Copy)]
pub struct RecallSelectionClients<'a> {
    pub limit: usize,
    pub relevance_model: Option<&'a dyn ModelClient>,
    pub embedding_client: Option<&'a LiveEmbeddingClient>,
    pub embedding_distance_threshold: Option<f32>,
    pub recency_weight: f32,
    pub connection: Option<&'a rusqlite::Connection>,
    pub query_embedding: Option<&'a [f32]>,
    pub now_iso: Option<&'a str>,
}

pub struct ReflectiveRecallPromptInputs<'a> {
    pub memories: &'a [&'a MemoryRecord],
    pub due_items: &'a [&'a AgendaItemRecord],
    pub agenda_items: &'a [&'a AgendaItemRecord],
    pub prompt: &'a str,
    pub recent_context: &'a [String],
    pub reflection_max_words: usize,
}

// ── transcript helpers ────────────────────────────────────────────────────────

pub fn transcript_contains_context_memory(
    transcript: &[ConversationMessage],
    memory_name: &str,
) -> bool {
    let tool_call_id = context_memory_tool_call_id(memory_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub fn transcript_contains_context_due_item(
    transcript: &[ConversationMessage],
    due_item_name: &str,
) -> bool {
    let tool_call_id = context_due_item_tool_call_id(due_item_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub fn transcript_contains_context_task(
    transcript: &[ConversationMessage],
    task_name: &str,
) -> bool {
    let tool_call_id = context_task_tool_call_id(task_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub fn transcript_contains_recalled_agenda_item(
    transcript: &[ConversationMessage],
    agenda_item_id: i64,
    agenda_item_name: &str,
) -> bool {
    transcript_contains_context_due_item(transcript, agenda_item_name)
        || recalled_item_matches(
            &recalled_item_refs_by_type(transcript, "AgendaItem"),
            agenda_item_id,
            agenda_item_name,
        )
}

pub fn message_matches_tool_call_id(message: &ConversationMessage, tool_call_id: &str) -> bool {
    message.tool_call_id.as_deref() == Some(tool_call_id)
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| tool_calls.iter().any(|call| call.id == tool_call_id))
}

// ── main recall entry points ──────────────────────────────────────────────────

pub fn recall_memory_context_messages_with_decision(
    memory_recall_classifier_window: usize,
    reflect: bool,
    prompt: &str,
    should_recall: bool,
    reflection_max_words: usize,
    selection_clients: RecallSelectionClients<'_>,
    context: RecallContext<'_>,
) -> Vec<ConversationMessage> {
    if !should_recall {
        return Vec::new();
    }

    let recall_query =
        build_recall_query(prompt, context.transcript, memory_recall_classifier_window);
    let shared_query_embedding = selection_clients
        .embedding_client
        .and_then(|client| client.embed(&recall_query).ok());
    let already_recalled_memories = recalled_item_refs_by_type(context.transcript, "Memory");
    let recalled = select_relevant_recall_memories(
        &recall_query,
        context.memories,
        &already_recalled_memories,
        RecallSelectionClients {
            limit: 2,
            query_embedding: shared_query_embedding.as_deref(),
            now_iso: selection_clients.now_iso,
            ..selection_clients
        },
    );
    let already_recalled_agenda_items =
        recalled_item_refs_by_type(context.transcript, "AgendaItem");
    let fast_due_items = if reflect {
        Vec::new()
    } else {
        select_relevant_recall_due_items(
            &recall_query,
            context.due_items,
            RecallSelectionClients {
                limit: 2,
                query_embedding: shared_query_embedding.as_deref(),
                ..selection_clients
            },
        )
        .into_iter()
        .filter(|item| !recalled_item_matches(&already_recalled_agenda_items, item.id, &item.name))
        .collect()
    };
    let fast_agenda_items = if reflect {
        Vec::new()
    } else {
        select_relevant_recall_agenda_items(
            &recall_query,
            context.agenda_items,
            RecallSelectionClients {
                limit: 2,
                query_embedding: shared_query_embedding.as_deref(),
                ..selection_clients
            },
        )
        .into_iter()
        .filter(|item| !recalled_item_matches(&already_recalled_agenda_items, item.id, &item.name))
        .collect()
    };
    let reflective_due_items = if reflect {
        select_relevant_recall_due_items(
            &recall_query,
            context.due_items,
            RecallSelectionClients {
                limit: 2,
                query_embedding: shared_query_embedding.as_deref(),
                ..selection_clients
            },
        )
        .into_iter()
        .filter(|item| !recalled_item_matches(&already_recalled_agenda_items, item.id, &item.name))
        .collect()
    } else {
        Vec::new()
    };
    let reflective_agenda_items = if reflect {
        select_relevant_recall_agenda_items(
            &recall_query,
            context.agenda_items,
            RecallSelectionClients {
                limit: 2,
                query_embedding: shared_query_embedding.as_deref(),
                ..selection_clients
            },
        )
        .into_iter()
        .filter(|item| !recalled_item_matches(&already_recalled_agenda_items, item.id, &item.name))
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
            selection_clients.relevance_model,
            ReflectiveRecallPromptInputs {
                memories: &recalled,
                due_items: &reflective_due_items,
                agenda_items: &reflective_agenda_items,
                prompt,
                recent_context: &recent_context,
                reflection_max_words,
            },
            &fallback_content,
        ) else {
            return Vec::new();
        };
        let content = serde_json::to_string_pretty(&serde_json::json!({
            "content": content,
            "recall_metadata": recalled.iter().map(|memory| {
                serde_json::json!({
                    "memory_type": "Memory",
                    "memory_id": memory.id,
                    "name": memory.name,
                })
            }).chain(reflective_due_items.iter().map(|item| {
                serde_json::json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            })).chain(reflective_agenda_items.iter().map(|item| {
                serde_json::json!({
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

    let content = serde_json::to_string_pretty(&serde_json::json!({
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
                serde_json::json!({
                    "memory_type": "Memory",
                    "memory_id": memory.id,
                    "name": memory.name,
                })
            })
            .chain(fast_due_items.iter().map(|item| {
                serde_json::json!({
                    "memory_type": "AgendaItem",
                    "memory_id": item.id,
                    "name": item.name,
                })
            }))
            .chain(fast_agenda_items.iter().map(|item| {
                serde_json::json!({
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

pub fn recall_memory_context_messages(
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
        100,
        RecallSelectionClients {
            limit: 2,
            relevance_model: None,
            embedding_client: None,
            embedding_distance_threshold: None,
            recency_weight: 0.0,
            connection: None,
            query_embedding: None,
            now_iso: None,
        },
        context,
    )
}

pub fn recall_due_item_context_messages(
    prompt: &str,
    transcript: &[ConversationMessage],
    due_items: &[AgendaItemRecord],
    now_iso: &str,
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<ConversationMessage> {
    let recall_query = build_recall_query(prompt, transcript, 6);
    let recalled = select_relevant_contextual_due_items(
        &recall_query,
        due_items,
        now_iso,
        RecallSelectionClients {
            limit: 2,
            ..selection_clients
        },
    );
    if recalled.is_empty() {
        return Vec::new();
    }
    recalled
        .into_iter()
        .filter(|item| !transcript_contains_recalled_agenda_item(transcript, item.id, &item.name))
        .flat_map(context_due_item_tool_messages)
        .collect()
}

pub fn memory_recall_status_updates_with_decision(
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

pub fn memory_recall_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
) -> Vec<StreamEvent> {
    let used_llm_classifier =
        memory_recall_classifier_enabled && !should_skip_memory_recall(prompt);
    memory_recall_status_updates_with_decision(used_llm_classifier, fetched_memories)
}

pub fn prompt_prelude_status_updates_with_decision(
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

pub fn prompt_prelude_status_updates(
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

pub fn should_skip_memory_recall(prompt: &str) -> bool {
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

pub fn parse_memory_recall_decision(response: &str) -> Option<(bool, String)> {
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

pub fn parse_relevance_filter_response(response: &str) -> Option<Vec<bool>> {
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

const SEMANTIC_RECALL_CANDIDATE_LIMIT: usize = 100;
const SEMANTIC_RECALL_SOURCE_FETCH_LIMIT: usize = 1_000;

pub fn semantic_recall_enabled(
    relevance_model: Option<&dyn ModelClient>,
    embedding_client: Option<&LiveEmbeddingClient>,
) -> bool {
    relevance_model.is_some() || embedding_client.is_some()
}

pub fn semantic_recall_source_fetch_limit(
    base_limit: usize,
    relevance_model: Option<&dyn ModelClient>,
    embedding_client: Option<&LiveEmbeddingClient>,
) -> usize {
    if semantic_recall_enabled(relevance_model, embedding_client) {
        base_limit.max(SEMANTIC_RECALL_SOURCE_FETCH_LIMIT)
    } else {
        base_limit
    }
}

fn semantic_recall_candidate_limit(
    limit: usize,
    relevance_model: Option<&dyn ModelClient>,
    embedding_client: Option<&LiveEmbeddingClient>,
) -> usize {
    if semantic_recall_enabled(relevance_model, embedding_client) {
        limit.saturating_mul(3).max(SEMANTIC_RECALL_CANDIDATE_LIMIT)
    } else {
        limit.saturating_mul(3).max(limit)
    }
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

fn l2_distance(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    Some(
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| {
                let delta = lhs - rhs;
                delta * delta
            })
            .sum(),
    )
}

fn embedding_rank_candidates<'a, T>(
    query: &str,
    candidates: impl IntoIterator<Item = &'a T>,
    selection_clients: RecallSelectionClients<'_>,
    extraction_fn: impl Fn(&T) -> String,
    file_path_fn: impl Fn(&T) -> &str,
    updated_at_fn: impl Fn(&T) -> i64,
) -> Vec<&'a T> {
    let Some(embedding_client) = selection_clients.embedding_client else {
        return Vec::new();
    };
    let owned_query_embedding;
    let query_embedding = if let Some(query_embedding) = selection_clients.query_embedding {
        query_embedding
    } else {
        let Ok(embedding) = embedding_client.embed(query) else {
            return Vec::new();
        };
        owned_query_embedding = embedding;
        owned_query_embedding.as_slice()
    };
    let now_unix = Utc::now().timestamp();
    let candidates = candidates.into_iter().collect::<Vec<_>>();
    let mut embedding_cache = selection_clients
        .connection
        .and_then(|connection| {
            let paths = candidates
                .iter()
                .map(|candidate| file_path_fn(candidate).to_string())
                .collect::<Vec<_>>();
            load_memory_embeddings_for_paths(connection, &paths).ok()
        })
        .unwrap_or_default();

    let mut ranked = candidates
        .into_iter()
        .filter_map(|candidate| {
            let embedding_text = extraction_fn(candidate);
            let file_path = file_path_fn(candidate);
            let embedding = if let Some(cached) = embedding_cache.get(file_path) {
                if cached.embedding_text == embedding_text {
                    Some(cached.embedding.clone())
                } else {
                    None
                }
            } else {
                None
            }
            .or_else(|| {
                let embedding = embedding_client.embed(&embedding_text).ok()?;
                if let Some(connection) = selection_clients.connection {
                    let _ =
                        upsert_memory_embedding(connection, file_path, &embedding, &embedding_text);
                    embedding_cache.insert(
                        file_path.to_string(),
                        MemoryEmbeddingRecord {
                            file_path: file_path.to_string(),
                            embedding: embedding.clone(),
                            embedding_text: embedding_text.clone(),
                            created_at_unix: 0,
                            updated_at_unix: 0,
                        },
                    );
                }
                Some(embedding)
            })?;
            let distance = l2_distance(query_embedding, &embedding)?;
            if let Some(distance_threshold) = selection_clients.embedding_distance_threshold
                && distance > distance_threshold
            {
                return None;
            }
            let adjusted_distance = distance
                + recency_penalty(
                    updated_at_fn(candidate),
                    now_unix,
                    selection_clients.recency_weight,
                );
            Some((adjusted_distance, distance, candidate))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(
        |(left_adjusted, left_distance, _), (right_adjusted, right_distance, _)| {
            left_adjusted
                .total_cmp(right_adjusted)
                .then_with(|| left_distance.total_cmp(right_distance))
        },
    );
    ranked
        .into_iter()
        .map(|(_, _, candidate)| candidate)
        .take(selection_clients.limit)
        .collect()
}

fn recency_penalty(updated_at_unix: i64, now_unix: i64, recency_weight: f32) -> f32 {
    if recency_weight <= 0.0 {
        return 0.0;
    }
    let age_seconds = now_unix.saturating_sub(updated_at_unix).max(0) as f32;
    let age_years = age_seconds / (86_400.0 * 365.0);
    recency_weight * age_years
}

pub fn select_relevant_recall_memories<'a>(
    query: &str,
    memories: &'a [MemoryRecord],
    already_recalled: &[RecalledItemRef],
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a MemoryRecord> {
    let candidate_limit = semantic_recall_candidate_limit(
        selection_clients.limit,
        selection_clients.relevance_model,
        selection_clients.embedding_client,
    );
    let overlap_candidates =
        select_recalled_memories(query, memories, already_recalled, candidate_limit);
    let candidates = if selection_clients.relevance_model.is_some() {
        let mut merged_candidates = overlap_candidates;
        for candidate in embedding_rank_candidates(
            query,
            memories
                .iter()
                .filter(|memory| !recalled_item_matches(already_recalled, memory.id, &memory.name)),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            memory_embedding_text,
            |memory| memory.file_path.as_str(),
            |memory| memory.updated_at_unix,
        ) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        for candidate in recent_memory_candidates(memories, already_recalled, candidate_limit) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        merged_candidates
    } else if selection_clients.embedding_client.is_some() {
        let embedding_candidates = embedding_rank_candidates(
            query,
            memories
                .iter()
                .filter(|memory| !recalled_item_matches(already_recalled, memory.id, &memory.name)),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            memory_embedding_text,
            |memory| memory.file_path.as_str(),
            |memory| memory.updated_at_unix,
        );
        if embedding_candidates.is_empty() {
            overlap_candidates
        } else {
            embedding_candidates
        }
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(
        selection_clients.relevance_model,
        query,
        candidates,
        memory_embedding_text,
    )
    .into_iter()
    .take(selection_clients.limit)
    .collect()
}

pub fn select_relevant_recall_due_items<'a>(
    query: &str,
    due_items: &'a [AgendaItemRecord],
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = semantic_recall_candidate_limit(
        selection_clients.limit,
        selection_clients.relevance_model,
        selection_clients.embedding_client,
    );
    let overlap_candidates =
        select_due_items_by_overlap(query, due_items, candidate_limit, selection_clients.now_iso);
    let candidates = if selection_clients.relevance_model.is_some() {
        let mut merged_candidates = overlap_candidates;
        for candidate in embedding_rank_candidates(
            query,
            due_items.iter(),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        ) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        for candidate in recent_due_item_candidates(due_items, candidate_limit) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        merged_candidates
    } else if selection_clients.embedding_client.is_some() {
        let embedding_candidates = embedding_rank_candidates(
            query,
            due_items.iter(),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        );
        if embedding_candidates.is_empty() {
            overlap_candidates
        } else {
            embedding_candidates
        }
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(
        selection_clients.relevance_model,
        query,
        candidates,
        agenda_item_embedding_text,
    )
    .into_iter()
    .take(selection_clients.limit)
    .collect()
}

pub fn select_relevant_contextual_due_items<'a>(
    query: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = semantic_recall_candidate_limit(
        selection_clients.limit,
        selection_clients.relevance_model,
        selection_clients.embedding_client,
    );
    let overlap_candidates =
        select_due_items_by_overlap(query, due_items, candidate_limit, Some(now_iso));
    let candidates = if selection_clients.relevance_model.is_some() {
        let mut merged_candidates = overlap_candidates;
        for candidate in embedding_rank_candidates(
            query,
            due_items
                .iter()
                .filter(|item| item.trigger_context.is_some()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        ) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        for candidate in recent_contextual_due_item_candidates(due_items, candidate_limit, now_iso)
        {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        merged_candidates
    } else if selection_clients.embedding_client.is_some() {
        let embedding_candidates = embedding_rank_candidates(
            query,
            due_items
                .iter()
                .filter(|item| item.trigger_context.is_some()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        );
        if embedding_candidates.is_empty() {
            overlap_candidates
        } else {
            embedding_candidates
        }
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(
        selection_clients.relevance_model,
        query,
        candidates,
        agenda_item_embedding_text,
    )
    .into_iter()
    .take(selection_clients.limit)
    .collect()
}

pub fn select_relevant_recall_agenda_items<'a>(
    query: &str,
    agenda_items: &'a [AgendaItemRecord],
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = semantic_recall_candidate_limit(
        selection_clients.limit,
        selection_clients.relevance_model,
        selection_clients.embedding_client,
    );
    let overlap_candidates = select_agenda_items_by_overlap(query, agenda_items, candidate_limit);
    let candidates = if selection_clients.relevance_model.is_some() {
        let mut merged_candidates = overlap_candidates;
        for candidate in embedding_rank_candidates(
            query,
            agenda_items
                .iter()
                .filter(|item| item.trigger_datetime.is_none() && item.trigger_context.is_none()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        ) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        for candidate in recent_agenda_item_candidates(agenda_items, candidate_limit) {
            if merged_candidates
                .iter()
                .any(|existing| existing.id == candidate.id)
            {
                continue;
            }
            merged_candidates.push(candidate);
            if merged_candidates.len() >= candidate_limit {
                break;
            }
        }
        merged_candidates
    } else if selection_clients.embedding_client.is_some() {
        let embedding_candidates = embedding_rank_candidates(
            query,
            agenda_items
                .iter()
                .filter(|item| item.trigger_datetime.is_none() && item.trigger_context.is_none()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
            agenda_item_embedding_text,
            |item| item.file_path.as_str(),
            |item| item.updated_at_unix,
        );
        if embedding_candidates.is_empty() {
            overlap_candidates
        } else {
            embedding_candidates
        }
    } else {
        overlap_candidates
    };
    filter_candidates_for_relevance(
        selection_clients.relevance_model,
        query,
        candidates,
        agenda_item_embedding_text,
    )
    .into_iter()
    .take(selection_clients.limit)
    .collect()
}

fn recent_memory_candidates<'a>(
    memories: &'a [MemoryRecord],
    already_recalled: &[RecalledItemRef],
    limit: usize,
) -> Vec<&'a MemoryRecord> {
    let mut candidates = memories
        .iter()
        .filter(|memory| !recalled_item_matches(already_recalled, memory.id, &memory.name))
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

fn recent_contextual_due_item_candidates<'a>(
    due_items: &'a [AgendaItemRecord],
    limit: usize,
    now_iso: &str,
) -> Vec<&'a AgendaItemRecord> {
    let mut candidates = due_items
        .iter()
        .filter(|item| item.trigger_context.is_some())
        .filter(|item| {
            item.trigger_datetime
                .as_deref()
                .is_none_or(|trigger_datetime| trigger_datetime > now_iso)
        })
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

pub fn classify_memory_recall_with_model(
    model: &dyn ModelClient,
    current_message: &str,
    recent_messages: &[ConversationMessage],
    window_size: usize,
) -> anyhow::Result<MemoryRecallDecision> {
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
        return Err(anyhow!("memory recall classifier returned invalid JSON"));
    };
    Ok(MemoryRecallDecision {
        needs_recall,
        reasoning,
        used_llm: true,
    })
}

pub fn determine_memory_recall_decision(
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

pub fn build_recall_query(
    prompt: &str,
    transcript: &[ConversationMessage],
    window: usize,
) -> String {
    let mut parts = recent_recall_context(transcript, window);
    parts.push(prompt.trim().to_string());
    parts.retain(|part| !part.is_empty());
    parts.join("\n")
}

pub fn recent_recall_context(transcript: &[ConversationMessage], window: usize) -> Vec<String> {
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
    memories: &[&MemoryRecord],
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
    memories: &[&MemoryRecord],
    due_items: &[&AgendaItemRecord],
    agenda_items: &[&AgendaItemRecord],
    prompt: &str,
    recent_context: &[String],
    reflection_max_words: usize,
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

    let mut body = String::from(
        "I am considering recalled context, as well as the transcript of a recent conversation. I am:\n\
- Re-stating the most relevant context from the recalled content\n\
- Reflecting on how the recalled content relates to the conversation transcript\n\n\
Specific examples are most helpful. For example, if the recalled content is:\n\n\
\"USER mentioned that when playing basketball, they struggle to remember to follow through on their shots.\"\n\n\
and the conversation transcript includes:\n\
\"USER: I'm going to play basketball next week\"\n\n\
a good response would be:\n\
\"I remember that USER struggles to remember to follow through on their shots when playing basketball. I should remind USER about following through on their shots for next week's game.\"\n\n\
My response will be in the first person, and will be transmitted to an AI assistant to inform their response. My response will NOT be transmitted to the user.\n\n",
    );
    body.push_str(&format!(
        "My response is brief and to the point, no more than {reflection_max_words} words.\n\n"
    ));
    body.push_str("Recalled Memory Content\n\n");
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

fn truncate_reflective_recall_content(content: &str, reflection_max_words: usize) -> String {
    let words = content.split_whitespace().collect::<Vec<_>>();
    if words.len() <= reflection_max_words {
        return content.trim().to_string();
    }
    words[..reflection_max_words].join(" ")
}

pub fn parse_reflective_recall_model_response(response: &str) -> Option<(bool, Option<String>)> {
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
    inputs: ReflectiveRecallPromptInputs<'_>,
    fallback_content: &str,
) -> Option<String> {
    let truncated_fallback =
        truncate_reflective_recall_content(fallback_content, inputs.reflection_max_words);
    let Some(model) = model else {
        return Some(truncated_fallback);
    };
    let model_prompt = build_reflective_recall_prompt(
        inputs.memories,
        inputs.due_items,
        inputs.agenda_items,
        inputs.prompt,
        inputs.recent_context,
        inputs.reflection_max_words,
    );
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
        Some((true, Some(content))) if !content.trim().is_empty() => Some(
            truncate_reflective_recall_content(&content, inputs.reflection_max_words),
        ),
        _ => Some(truncated_fallback),
    }
}

pub fn recalled_memory_names(transcript: &[ConversationMessage]) -> HashSet<String> {
    recalled_item_refs_by_type(transcript, "Memory")
        .into_iter()
        .map(|item| item.name)
        .collect()
}

pub fn recalled_item_refs_by_type(
    transcript: &[ConversationMessage],
    memory_type: &str,
) -> Vec<RecalledItemRef> {
    transcript
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .filter_map(|message| message.content.as_deref())
        .flat_map(|content| parse_recalled_item_refs(content, memory_type))
        .collect()
}

pub fn select_due_items_by_overlap<'a>(
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
            if skip_time_due_before.is_some_and(|now_iso| {
                item.trigger_datetime
                    .as_deref()
                    .is_some_and(|trigger_datetime| trigger_datetime <= now_iso)
            }) {
                return None;
            }

            let trigger_context = item.trigger_context.as_deref().unwrap_or("");
            let mut haystack = String::with_capacity(
                item.name.len() + item.body.len() + trigger_context.len() + 2,
            );
            haystack.push_str(&item.name);
            haystack.push(' ');
            haystack.push_str(&item.body);
            if !trigger_context.is_empty() {
                haystack.push(' ');
                haystack.push_str(trigger_context);
            }
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

pub fn select_recalled_due_items<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    select_due_items_by_overlap(prompt, due_items, limit, Some(now_iso))
}

pub fn parse_recalled_item_refs(content: &str, desired_memory_type: &str) -> Vec<RecalledItemRef> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
        return Vec::new();
    };

    if let Some(items) = value.as_array() {
        return items
            .iter()
            .filter_map(|item| {
                item.get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(|name| RecalledItemRef {
                        id: item.get("memory_id").and_then(serde_json::Value::as_i64),
                        name: name.to_ascii_lowercase(),
                    })
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
                    .map(|name| RecalledItemRef {
                        id: item.get("memory_id").and_then(serde_json::Value::as_i64),
                        name: name.to_ascii_lowercase(),
                    })
            })?
        })
        .collect()
}

pub fn recalled_item_matches(
    recalled_items: &[RecalledItemRef],
    item_id: i64,
    item_name: &str,
) -> bool {
    let normalized_name = item_name.to_ascii_lowercase();
    recalled_items.iter().any(|recalled| {
        recalled.id == Some(item_id) || (recalled.id.is_none() && recalled.name == normalized_name)
    })
}

pub fn select_recalled_memories<'a>(
    prompt: &str,
    memories: &'a [MemoryRecord],
    already_recalled: &[RecalledItemRef],
    limit: usize,
) -> Vec<&'a MemoryRecord> {
    let prompt_tokens = significant_tokens(prompt);
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored = memories
        .iter()
        .filter_map(|memory| {
            if recalled_item_matches(already_recalled, memory.id, &memory.name) {
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

pub fn significant_tokens(text: &str) -> HashSet<String> {
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

// ── consolidation (moved from elroy-app/src/consolidation.rs) ────────────────

pub fn best_effort_embedding_client(
    provider_config: Option<&EmbeddingProviderConfig>,
) -> Option<LiveEmbeddingClient> {
    provider_config.and_then(|config| LiveEmbeddingClient::new(config.clone()).ok())
}

pub fn best_effort_provider_model(
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

pub fn formulate_memory_from_transcript(transcript: &[ConversationMessage]) -> (String, String) {
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

pub fn list_all_active_memories_in_scope(
    connection: &rusqlite::Connection,
    memory_dir: &Path,
) -> rusqlite::Result<Vec<MemoryRecord>> {
    let memory_dir = memory_dir
        .canonicalize()
        .unwrap_or_else(|_| memory_dir.to_path_buf());
    Ok(list_all_active_memories(connection)?
        .into_iter()
        .filter(|memory| {
            let path = Path::new(&memory.file_path);
            let path = if path.is_relative() {
                memory_dir.join(path)
            } else {
                path.to_path_buf()
            };
            let path = path.canonicalize().unwrap_or(path);
            path.starts_with(&memory_dir)
        })
        .collect())
}

pub fn remove_context_memory_messages_by_names(
    connection: &mut rusqlite::Connection,
    memory_names: &[String],
) -> anyhow::Result<()> {
    if memory_names.is_empty() {
        return Ok(());
    }

    let tool_call_ids = memory_names
        .iter()
        .map(|name| context_memory_tool_call_id(name))
        .collect::<Vec<_>>();
    let transcript = load_context_messages(connection, LOCAL_USER_TOKEN)?;
    let updated_transcript = transcript
        .into_iter()
        .filter(|message| {
            !tool_call_ids
                .iter()
                .any(|tool_call_id| message_matches_tool_call_id(message, tool_call_id))
        })
        .collect::<Vec<_>>();
    replace_context_messages(connection, LOCAL_USER_TOKEN, &updated_transcript)?;
    Ok(())
}

pub fn create_consolidated_memories_from_records(
    bootstrap_plan: &BootstrapPlan,
    outputs: &[ConsolidatedMemoryOutput],
    source_memories: &[MemoryRecord],
) -> std::io::Result<Vec<PathBuf>> {
    if outputs.is_empty() {
        return Err(std::io::Error::other(
            "at least one consolidated memory output is required",
        ));
    }

    let archive_dir = bootstrap_plan.memory_dir.join("archive");
    let mut archived_sources = Vec::new();
    for memory in source_memories {
        let archived_path = archive_memory_file(Path::new(&memory.file_path), &archive_dir)?;
        archived_sources.push((memory.name.clone(), archived_path));
    }

    let frontmatter = memory_source_frontmatter(
        &archived_sources
            .iter()
            .map(|(source_name, path)| (source_name.as_str(), path.as_path()))
            .collect::<Vec<_>>(),
    );
    let mut created = Vec::new();
    for output in outputs {
        created.push(create_memory_file_with_frontmatter(
            &bootstrap_plan.memory_dir,
            &output.name,
            &output.text,
            frontmatter.as_deref(),
        )?);
    }
    elroy_db::bootstrap_database(bootstrap_plan)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let mut connection = open_sqlite_connection(&bootstrap_plan.database_path)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    run_migrations(&mut connection).map_err(|e| std::io::Error::other(e.to_string()))?;
    remove_context_memory_messages_by_names(
        &mut connection,
        &source_memories
            .iter()
            .map(|memory| memory.name.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(created)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryConsolidationSettings {
    pub memory_cluster_similarity_threshold: f64,
    pub max_memory_cluster_size: usize,
    pub min_memory_cluster_size: usize,
    pub fast_provider_config: Option<ProviderConfig>,
    pub embedding_provider_config: Option<EmbeddingProviderConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsolidatedMemoryOutput {
    pub name: String,
    pub text: String,
}

pub fn run_auto_memory_if_needed(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
    existing_transcript_len: usize,
    transcript: &[ConversationMessage],
    messages_between_memory: usize,
) -> anyhow::Result<()> {
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
    elroy_db::bootstrap_database(bootstrap_plan).map_err(|error| anyhow::anyhow!("{error}"))?;
    *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
    record_memory_creation_and_maybe_consolidate(
        connection,
        bootstrap_plan,
        memories_between_consolidation,
        memory_consolidation_settings,
    )?;
    Ok(())
}

pub fn record_memory_creation_and_maybe_consolidate(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
) -> anyhow::Result<()> {
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

    consolidate_memories(connection, bootstrap_plan, memory_consolidation_settings)?;
    tracker.memories_since_consolidation = 0;
    tracker.updated_at_unix = Utc::now().timestamp();
    save_memory_operation_tracker(connection, &tracker)?;
    Ok(())
}

fn consolidate_memories(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
) -> anyhow::Result<()> {
    consolidate_exact_duplicate_memories(connection, bootstrap_plan)?;
    if let Some(settings) = memory_consolidation_settings {
        consolidate_semantic_memory_clusters(connection, bootstrap_plan, settings)?;
    }
    Ok(())
}

pub fn consolidate_exact_duplicate_memories(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
) -> anyhow::Result<()> {
    let active_memories =
        list_all_active_memories_in_scope(connection, &bootstrap_plan.memory_dir)?;
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
        let canonical = group
            .first()
            .ok_or_else(|| anyhow::anyhow!("duplicate memory group was unexpectedly empty"))?;
        create_consolidated_memories_from_records(
            bootstrap_plan,
            &[ConsolidatedMemoryOutput {
                name: canonical.name.clone(),
                text: canonical.body.clone(),
            }],
            &group,
        )?;
        *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
        run_migrations(connection)?;
    }

    Ok(())
}

pub fn consolidate_semantic_memory_clusters(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    settings: &MemoryConsolidationSettings,
) -> anyhow::Result<()> {
    if settings.min_memory_cluster_size < 2 || settings.max_memory_cluster_size < 2 {
        return Ok(());
    }

    let embedding_client =
        best_effort_embedding_client(settings.embedding_provider_config.as_ref());
    let Some(embedding_client) = embedding_client else {
        return Ok(());
    };
    let fast_model = best_effort_provider_model(settings.fast_provider_config.as_ref(), "Elroy");

    let memories = list_all_active_memories_in_scope(connection, &bootstrap_plan.memory_dir)?;
    if memories.len() < settings.min_memory_cluster_size {
        return Ok(());
    }

    let embedded_memories =
        load_or_create_cached_memory_embeddings(connection, &memories, &embedding_client)?;
    if embedded_memories.len() < settings.min_memory_cluster_size {
        return Ok(());
    }

    let embeddings = embedded_memories
        .iter()
        .map(|(_, embedding)| embedding.clone())
        .collect::<Vec<_>>();
    let memories = embedded_memories
        .into_iter()
        .map(|(memory, _)| memory)
        .collect::<Vec<_>>();
    let clusters = semantic_memory_clusters(
        &embeddings,
        settings.memory_cluster_similarity_threshold as f32,
        settings.min_memory_cluster_size,
        settings.max_memory_cluster_size,
    );

    for cluster in clusters
        .into_iter()
        .take(MEMORY_CONSOLIDATION_CLUSTER_LIMIT)
    {
        let cluster_memories = cluster
            .into_iter()
            .filter_map(|index| memories.get(index).cloned())
            .collect::<Vec<_>>();
        if cluster_memories.len() < settings.min_memory_cluster_size {
            continue;
        }
        let outputs = consolidate_memory_cluster_outputs(
            &cluster_memories,
            fast_model.as_ref().map(|model| model as &dyn ModelClient),
        );
        if outputs.is_empty() {
            continue;
        }
        create_consolidated_memories_from_records(bootstrap_plan, &outputs, &cluster_memories)?;
        *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
        run_migrations(connection)?;
    }

    Ok(())
}

pub fn update_outdated_or_incorrect_memory_from_config(
    config: &AppConfig,
    memory_name: &str,
    update_text: &str,
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
    let memory =
        match find_active_memory_by_name_in_scope(&connection, memory_name, &config.memory_dir) {
            Ok(Some(memory)) => memory,
            Ok(None) => {
                return ToolExecutionResult::success(format!("Memory '{memory_name}' not found"));
            }
            Err(error) => {
                return ToolExecutionResult::error(format!("database query failed: {error}"));
            }
        };
    let path = Path::new(&memory.file_path);
    let existing = match read_memory_parts(path) {
        Ok((_, body)) => body,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to read memory file: {error}"));
        }
    };

    let mut updated = existing;
    if !updated.is_empty() {
        updated.push_str("\n\n");
    }
    updated.push_str(&format!(
        "Update ({}):\n{}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        update_text.trim()
    ));

    let archive_dir = config.memory_dir.join("archive");
    match archive_memory_file(path, &archive_dir) {
        Ok(archived_path) => {
            let frontmatter = memory_source_frontmatter(&[(memory_name, &archived_path)]);
            match create_memory_file_with_frontmatter(
                &config.memory_dir,
                memory_name,
                &updated,
                frontmatter.as_deref(),
            ) {
                Ok(_) => {
                    if let Err(error) =
                        elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))
                    {
                        return ToolExecutionResult::error(format!(
                            "failed to rebuild derived state: {error}"
                        ));
                    }
                    if let Err(error) =
                        sync_memory_context_after_mutation(config, memory_name, Some(memory_name))
                    {
                        return ToolExecutionResult::error(format!(
                            "failed to sync context: {error}"
                        ));
                    }
                    ToolExecutionResult::success(format!(
                        "Memory '{}' has been updated",
                        memory_name
                    ))
                }
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to create updated memory file: {error}"
                )),
            }
        }
        Err(error) => ToolExecutionResult::error(format!("failed to archive old memory: {error}")),
    }
}

pub fn search_memories_from_config(
    config: &AppConfig,
    query: &str,
    limit: usize,
) -> ToolExecutionResult {
    let limit = limit.min(2);
    let mut connection = match open_sqlite_connection(&config.database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    if let Err(error) = run_migrations(&mut connection) {
        return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
    }

    let provider_config = fast_provider_config_from_app_config(config).ok();
    let embedding_provider_config = embedding_provider_config_from_app_config(config).ok();

    let relevance_model =
        best_effort_provider_model(provider_config.as_ref(), &config.assistant_name);
    let embedding_client = best_effort_embedding_client(embedding_provider_config.as_ref());

    let query_embedding = embedding_client
        .as_ref()
        .and_then(|client| client.embed(query).ok());

    let selection_clients = RecallSelectionClients {
        limit,
        embedding_distance_threshold: Some(config.l2_memory_relevance_distance_threshold as f32),
        recency_weight: config.recency_weight as f32,
        connection: Some(&connection),
        relevance_model: relevance_model.as_ref().map(|m| m as &dyn ModelClient),
        embedding_client: embedding_client.as_ref(),
        query_embedding: query_embedding.as_deref(),
        now_iso: None,
    };

    let source_fetch_limit = semantic_recall_source_fetch_limit(
        limit * 3,
        relevance_model
            .as_ref()
            .map(|model| model as &dyn ModelClient),
        embedding_client.as_ref(),
    );

    let memories = if semantic_recall_enabled(
        relevance_model
            .as_ref()
            .map(|model| model as &dyn ModelClient),
        embedding_client.as_ref(),
    ) {
        match list_all_active_memories_in_scope(&connection, &config.memory_dir) {
            Ok(memories) => memories,
            Err(error) => {
                return ToolExecutionResult::error(format!("database query failed: {error}"));
            }
        }
    } else {
        match search_active_memories_in_scope(&connection, &config.memory_dir, query, 10_000) {
            Ok(memories) => memories,
            Err(error) => {
                return ToolExecutionResult::error(format!("database query failed: {error}"));
            }
        }
    };

    let due_items = match elroy_db::list_active_due_items(&connection, source_fetch_limit) {
        Ok(items) => items,
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    let agenda_items =
        match elroy_db::list_active_plain_agenda_items(&connection, source_fetch_limit) {
            Ok(items) => items,
            Err(error) => {
                return ToolExecutionResult::error(format!("database query failed: {error}"));
            }
        };

    let relevant_memories =
        select_relevant_recall_memories(query, &memories, &[], selection_clients);
    let relevant_due_items = select_relevant_recall_due_items(query, &due_items, selection_clients);
    let relevant_agenda_items =
        select_relevant_recall_agenda_items(query, &agenda_items, selection_clients);

    ToolExecutionResult::success(format_memory_search_results(
        &relevant_memories,
        &relevant_due_items,
        &relevant_agenda_items,
    ))
}
pub fn format_due_item_examination(item: &elroy_db::AgendaItemRecord) -> String {
    let mut text = format!("# Due Item: {}\n\n{}", item.name, item.body.trim());
    if let Some(trigger_datetime) = item.trigger_datetime.as_deref() {
        text.push_str(&format!("\n\nScheduled for: {trigger_datetime}"));
    }
    if let Some(trigger_context) = item.trigger_context.as_deref() {
        text.push_str(&format!("\n\nTrigger context: {trigger_context}"));
    }
    text
}

pub fn examine_memories_from_config(config: &AppConfig, question: &str) -> ToolExecutionResult {
    let mut connection = match open_sqlite_connection(&config.database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    if let Err(error) = run_migrations(&mut connection) {
        return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
    }

    let provider_config = fast_provider_config_from_app_config(config).ok();
    let embedding_provider_config = embedding_provider_config_from_app_config(config).ok();

    let relevance_model =
        best_effort_provider_model(provider_config.as_ref(), &config.assistant_name);
    let embedding_client = best_effort_embedding_client(embedding_provider_config.as_ref());

    let query_embedding = embedding_client
        .as_ref()
        .and_then(|client| client.embed(question).ok());

    let selection_clients = RecallSelectionClients {
        limit: 2,
        embedding_distance_threshold: Some(config.l2_memory_relevance_distance_threshold as f32),
        recency_weight: config.recency_weight as f32,
        connection: Some(&connection),
        relevance_model: relevance_model.as_ref().map(|m| m as &dyn ModelClient),
        embedding_client: embedding_client.as_ref(),
        query_embedding: query_embedding.as_deref(),
        now_iso: None,
    };

    let memories = match list_all_active_memories_in_scope(&connection, &config.memory_dir) {
        Ok(memories) => memories,
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    let due_items = match elroy_db::list_active_due_items(&connection, 1_000) {
        Ok(items) => items,
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };
    let agenda_items = match elroy_db::list_active_plain_agenda_items(&connection, 1_000) {
        Ok(items) => items,
        Err(error) => return ToolExecutionResult::error(format!("database query failed: {error}")),
    };

    let recalled_memories =
        select_relevant_recall_memories(question, &memories, &[], selection_clients);

    let recalled_due_items =
        select_relevant_recall_due_items(question, &due_items, selection_clients);

    let recalled_agenda_items =
        select_relevant_recall_agenda_items(question, &agenda_items, selection_clients);

    let mut reports = Vec::new();

    if !recalled_memories.is_empty() {
        reports.push("### RELEVANT MEMORIES".to_string());
        for memory in recalled_memories {
            reports.push(format_memory_examination(memory));
        }
    }

    if !recalled_due_items.is_empty() {
        reports.push("### RELEVANT DUE ITEMS".to_string());
        for item in recalled_due_items {
            reports.push(format_due_item_examination(item));
        }
    }

    if !recalled_agenda_items.is_empty() {
        reports.push("### RELEVANT AGENDA ITEMS".to_string());
        for item in recalled_agenda_items {
            reports.push(format_agenda_item_detail(item));
        }
    }

    if reports.is_empty() {
        ToolExecutionResult::success("No relevant memories found".to_string())
    } else {
        ToolExecutionResult::success(reports.join("\n\n"))
    }
}

fn load_or_create_cached_memory_embeddings(
    connection: &rusqlite::Connection,
    memories: &[MemoryRecord],
    embedding_client: &LiveEmbeddingClient,
) -> anyhow::Result<Vec<(MemoryRecord, Vec<f32>)>> {
    let embedding_cache = load_memory_embeddings_for_paths(
        connection,
        &memories
            .iter()
            .map(|memory| memory.file_path.clone())
            .collect::<Vec<_>>(),
    )?;

    let mut embedded_memories = Vec::new();
    for memory in memories {
        let embedding_text = memory_embedding_text(memory);
        if let Some(cached) = embedding_cache.get(&memory.file_path)
            && cached.embedding_text == embedding_text
        {
            embedded_memories.push((memory.clone(), cached.embedding.clone()));
            continue;
        }

        let Ok(embedding) = embedding_client.embed(&embedding_text) else {
            continue;
        };
        upsert_memory_embedding(connection, &memory.file_path, &embedding, &embedding_text)?;
        embedded_memories.push((memory.clone(), embedding));
    }
    Ok(embedded_memories)
}

pub fn memory_embedding_text(memory: &MemoryRecord) -> String {
    format!("# {}\n{}", memory.name, memory.body.trim())
}

fn semantic_memory_clusters(
    embeddings: &[Vec<f32>],
    distance_threshold: f32,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> Vec<Vec<usize>> {
    if embeddings.len() < min_cluster_size {
        return Vec::new();
    }

    let mut visited = vec![false; embeddings.len()];
    let mut assigned = vec![false; embeddings.len()];
    let mut clusters = Vec::new();

    for index in 0..embeddings.len() {
        if visited[index] {
            continue;
        }
        visited[index] = true;
        let neighbors = semantic_memory_neighbors(embeddings, index, distance_threshold);
        if neighbors.len() + 1 < min_cluster_size {
            continue;
        }

        let mut cluster = vec![index];
        assigned[index] = true;
        let mut seeds = VecDeque::from(neighbors);
        while let Some(point) = seeds.pop_front() {
            if !visited[point] {
                visited[point] = true;
                let point_neighbors =
                    semantic_memory_neighbors(embeddings, point, distance_threshold);
                if point_neighbors.len() + 1 >= min_cluster_size {
                    for neighbor in point_neighbors {
                        if !seeds.contains(&neighbor) {
                            seeds.push_back(neighbor);
                        }
                    }
                }
            }
            if !assigned[point] {
                assigned[point] = true;
                cluster.push(point);
            }
        }

        if cluster.len() > max_cluster_size {
            cluster = densest_memory_cluster_indices(embeddings, &cluster, max_cluster_size);
        }
        clusters.push(cluster);
    }

    clusters.sort_by(|left, right| {
        right.len().cmp(&left.len()).then_with(|| {
            mean_cluster_distance(embeddings, left)
                .total_cmp(&mean_cluster_distance(embeddings, right))
        })
    });
    clusters
}

fn semantic_memory_neighbors(
    embeddings: &[Vec<f32>],
    index: usize,
    distance_threshold: f32,
) -> Vec<usize> {
    embeddings
        .iter()
        .enumerate()
        .filter_map(|(candidate_index, candidate)| {
            if candidate_index == index {
                return None;
            }
            let distance = cosine_distance(&embeddings[index], candidate)?;
            (distance <= distance_threshold).then_some(candidate_index)
        })
        .collect()
}

fn densest_memory_cluster_indices(
    embeddings: &[Vec<f32>],
    cluster: &[usize],
    limit: usize,
) -> Vec<usize> {
    if cluster.len() <= limit {
        return cluster.to_vec();
    }

    let mut scored = cluster
        .iter()
        .map(|&index| {
            let distances = cluster
                .iter()
                .filter(|&&other| other != index)
                .filter_map(|&other| cosine_distance(&embeddings[index], &embeddings[other]))
                .collect::<Vec<_>>();
            let mean_distance = if distances.is_empty() {
                0.0
            } else {
                distances.iter().sum::<f32>() / distances.len() as f32
            };
            (mean_distance, index)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(left, _), (right, _)| left.total_cmp(right));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, index)| index)
        .collect()
}

fn mean_cluster_distance(embeddings: &[Vec<f32>], cluster: &[usize]) -> f32 {
    if cluster.len() < 2 {
        return 0.0;
    }

    let mut distances = Vec::new();
    for left in 0..cluster.len() {
        for right in left + 1..cluster.len() {
            if let Some(distance) =
                cosine_distance(&embeddings[cluster[left]], &embeddings[cluster[right]])
            {
                distances.push(distance);
            }
        }
    }

    if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    }
}

fn cosine_distance(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        dot += lhs * rhs;
        left_norm += lhs * lhs;
        right_norm += rhs * rhs;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return None;
    }
    Some(1.0 - (dot / (left_norm.sqrt() * right_norm.sqrt())))
}

pub fn consolidate_memory_cluster_outputs(
    memories: &[MemoryRecord],
    model: Option<&dyn ModelClient>,
) -> Vec<ConsolidatedMemoryOutput> {
    let Some(model) = model else {
        return fallback_consolidated_memory_outputs(memories);
    };

    let prompt = build_memory_consolidation_prompt(memories);
    let Ok(events) = model.next_events(ConversationRequest {
        user_message: &prompt,
        tools: &[],
        transcript: &[ConversationMessage::new(MessageRole::User, prompt.clone())],
        force_tool: None,
    }) else {
        return fallback_consolidated_memory_outputs(memories);
    };
    let response = events
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>();
    parse_consolidated_memory_response(&response)
        .filter(|outputs| !outputs.is_empty())
        .unwrap_or_else(|| fallback_consolidated_memory_outputs(memories))
}

fn build_memory_consolidation_prompt(memories: &[MemoryRecord]) -> String {
    format!(
        "# Memory Consolidation Task\n\n\
Your task is to consolidate or reorganize two or more memory excerpts. These excerpts have been flagged as having overlapping or redundant content and require consolidation or reorganization.\n\n\
Each excerpt has the following characteristics:\n\
- They are written from the first-person perspective of an AI assistant.\n\
- They consist of a title and a main body.\n\n\
If the excerpts cover the same topic, consolidate them into a single, cohesive memory. If they address distinct topics, create separate, reorganized memories for each.\n\n\
## Dates and times\n\n\
The memories being consolidated can be from any time in the past. Note that the current time is {}, or {} UTC\n\n\
Use ISO 8601 format for dates and times to ensure references remain unambiguous in future retrievals.\n\n\
## Synthesis Guidelines\n\n\
- Resolve contradictions instead of carrying conflicting claims forward unchanged.\n\
- When dates or times are present and memories disagree, prefer the most recent dated information.\n\
- Call out recurring patterns or repeated events when the excerpts support that inference.\n\
- If the excerpts describe the same topic, produce a single synthesis rather than a list of lightly edited duplicates.\n\n\
## Style Guidelines\n\n\
- Limit each new memory excerpt to {} words.\n\n\
## Memory Title Guidelines\n\n\
Examples of effective and ineffective memory titles are provided:\n\n\
**Ineffective:**\n\
- UserFoo's project progress and personal goals: 'Personal goals' is too vague; two topics are referenced.\n\n\
**Effective:**\n\
- UserFoo's project on building a treehouse: Specific and topic-focused.\n\
- UserFoo's goal to be more thoughtful in conversation: Specifies a clear goal.\n\n\
**Ineffective:**\n\
- UserFoo's weekend plans: 'Weekend plans' lacks specificity, and dates should be in ISO 8601 format.\n\n\
**Effective:**\n\
- UserFoo's plan to attend a concert on 2022-02-11: Specific with a defined date.\n\n\
**Ineffective:**\n\
- UserFoo's preferred name and well-being: Covers two distinct topics; 'well-being' is generic.\n\n\
**Effective:**\n\
- UserFoo's preferred name: Focused on a single topic.\n\
- UserFoo's feeling of rejuvenation after rest: Clarifies the topic.\n\n\
## Formatting\n\n\
Respond in Markdown format using this structure:\n\n\
```markdown\n\
# Memory Consolidation Reasoning\n\
Provide a clear explanation of the consolidation or reorganization choices. Justify which information was included or omitted, and detail organizational strategies and considerations.\n\n\
## Memory Title 1\n\
Include all pertinent content from the original memories for the specified topic. Optionally, add reflections on how the assistant should respond to this information, along with any open questions the memory poses.\n\n\
## Memory Title 2  (If necessary)\n\
Detail the content for a second memory, should distinct topics require individual consolidation. Repeat as needed.\n\
```\n\n\
If the excerpts are about the same topic, usually return a single consolidated memory. If they are distinct, return multiple reorganized memories.\n\n\
# Memory Consolidation Input\n{}",
        Local::now().to_rfc3339(),
        Utc::now().to_rfc3339(),
        MEMORY_WORD_COUNT_LIMIT,
        format_memory_cluster_for_prompt(memories)
    )
}

pub fn parse_consolidated_memory_response(response: &str) -> Option<Vec<ConsolidatedMemoryOutput>> {
    let normalized = strip_outer_code_fence(response.trim());
    if let Some(outputs) = parse_markdown_consolidated_memory_response(&normalized) {
        return Some(outputs);
    }
    let value = serde_json::from_str::<Value>(normalized.trim()).ok()?;
    let memories = value.get("memories")?.as_array()?;
    Some(
        memories
            .iter()
            .filter_map(|memory| {
                let name = memory.get("name")?.as_str()?.trim();
                let text = memory.get("text")?.as_str()?.trim();
                if name.is_empty() || text.is_empty() {
                    return None;
                }
                Some(ConsolidatedMemoryOutput {
                    name: name.to_string(),
                    text: text.to_string(),
                })
            })
            .collect(),
    )
}

fn strip_outer_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut lines = trimmed.lines();
    let _opening = lines.next();
    let remainder = lines.collect::<Vec<_>>();
    let mut end = remainder.len();
    while end > 0 && remainder[end - 1].trim().is_empty() {
        end -= 1;
    }
    if end > 0 && remainder[end - 1].trim_start().starts_with("```") {
        end -= 1;
    }
    remainder[..end].join("\n")
}

fn parse_markdown_consolidated_memory_response(
    response: &str,
) -> Option<Vec<ConsolidatedMemoryOutput>> {
    let mut outputs = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_body = Vec::new();

    let flush_current = |outputs: &mut Vec<ConsolidatedMemoryOutput>,
                         current_name: &mut Option<String>,
                         current_body: &mut Vec<String>| {
        let Some(name) = current_name.take() else {
            return;
        };
        let text = current_body.join("\n").trim().to_string();
        current_body.clear();
        if text.is_empty() {
            return;
        }
        outputs.push(ConsolidatedMemoryOutput { name, text });
    };

    for line in response.lines() {
        let trimmed = line.trim_end();
        if let Some(heading) = trimmed.strip_prefix("## ") {
            flush_current(&mut outputs, &mut current_name, &mut current_body);
            let title = heading.trim();
            if !title.is_empty() {
                current_name = Some(title.to_string());
            }
            continue;
        }
        if current_name.is_some() {
            current_body.push(trimmed.to_string());
        }
    }
    flush_current(&mut outputs, &mut current_name, &mut current_body);

    if outputs.is_empty() {
        None
    } else {
        Some(outputs)
    }
}

fn fallback_consolidated_memory_outputs(
    memories: &[MemoryRecord],
) -> Vec<ConsolidatedMemoryOutput> {
    let Some(primary) = memories.first() else {
        return Vec::new();
    };
    let mut seen = HashSet::new();
    let mut parts = Vec::new();
    for memory in memories {
        let body = memory.body.trim();
        if body.is_empty() || !seen.insert(body.to_string()) {
            continue;
        }
        parts.push(body.to_string());
    }
    if parts.is_empty() {
        parts.push(primary.body.trim().to_string());
    }
    vec![ConsolidatedMemoryOutput {
        name: primary.name.clone(),
        text: parts.join("\n\n"),
    }]
}

fn format_memory_cluster_for_prompt(memories: &[MemoryRecord]) -> String {
    memories
        .iter()
        .map(|memory| format!("## {}\n{}", memory.name, memory.body.trim()))
        .collect::<Vec<_>>()
        .join("\n\n")
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

pub fn create_memory_file_from_context_messages(
    memory_dir: &Path,
    name: &str,
    text: &str,
    context_messages: &[ConversationMessage],
) -> std::io::Result<PathBuf> {
    let frontmatter = context_message_source_frontmatter(context_messages);
    create_memory_file_with_frontmatter(memory_dir, name, text, frontmatter.as_deref())
}

pub fn context_message_source_frontmatter(
    context_messages: &[ConversationMessage],
) -> Option<String> {
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

pub fn parse_context_message_source_ids(frontmatter: Option<&str>) -> Option<Vec<i64>> {
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

pub fn memory_source_frontmatter(memory_sources: &[(&str, &Path)]) -> Option<String> {
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

pub fn parse_memory_sources(frontmatter: Option<&str>) -> Option<Vec<(String, String)>> {
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

pub fn list_memory_sources(frontmatter: Option<&str>) -> Vec<(String, String)> {
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

pub fn format_context_message_source_content(messages: &[ConversationMessage]) -> String {
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

pub fn format_memory_file_source_content(source_name: &str, source_body: &str) -> String {
    format!("#{source_name}\n{source_body}")
}
