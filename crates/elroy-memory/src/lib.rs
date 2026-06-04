use anyhow::anyhow;
use chrono::{Local, Utc};
use elroy_agenda::create_agenda_file;
use elroy_agenda::tools::{derive_agenda_item_name, parse_trigger_datetime_for_validation};
use elroy_config::{AppConfig, provider_config_from_app_config};
use elroy_core::{ConversationRequest, ModelClient};
use elroy_db::{
    AgendaItemRecord, BootstrapPlan, MemoryRecord, find_active_agenda_item_by_name,
    find_active_memory_by_name, load_memories_by_ids, open_sqlite_connection, run_migrations,
};
use elroy_llm::{ConversationMessage, MessageRole, StreamEvent};
use elroy_recall::{
    RecallSelectionClients, augment_text_from_config, best_effort_embedding_client,
    best_effort_provider_model, find_active_memory_by_name_in_scope,
    get_source_content_for_memory_from_config, list_active_memories_in_scope,
    list_all_active_memories_in_scope, list_memory_sources, select_relevant_recall_agenda_items,
    select_relevant_recall_due_items, select_relevant_recall_memories, semantic_recall_enabled,
    semantic_recall_source_fetch_limit, sync_due_item_context_after_mutation,
    sync_memory_context_after_mutation,
};

pub use elroy_core::memory_store::*;

pub mod tools;

#[derive(Debug, Clone, PartialEq)]
pub enum IngestedMemoItem {
    Memory(MemoryRecord),
    DueItem(AgendaItemRecord),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelevantRecallItem {
    Memory(MemoryRecord),
    DueItem(AgendaItemRecord),
    AgendaItem(AgendaItemRecord),
}

fn collect_assistant_response(events: Vec<StreamEvent>) -> String {
    events
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::AssistantResponse { content } => Some(content),
            _ => None,
        })
        .collect::<String>()
}

fn parse_memory_request(value: &serde_json::Value) -> Option<(String, String)> {
    let request = value.get("create_memory_request")?;
    if request.is_null() {
        return None;
    }
    let name = request.get("name")?.as_str()?.trim();
    let text = request.get("text")?.as_str()?.trim();
    if name.is_empty() || text.is_empty() {
        return None;
    }
    Some((name.to_string(), text.to_string()))
}

fn parse_due_item_request(
    value: &serde_json::Value,
) -> Option<(String, String, Option<String>, Option<String>)> {
    let request = value.get("create_due_item_request")?;
    if request.is_null() {
        return None;
    }
    let name = request.get("name")?.as_str()?.trim();
    let text = request.get("text")?.as_str()?.trim();
    if name.is_empty() || text.is_empty() {
        return None;
    }
    let trigger_time = request
        .get("trigger_time")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            request
                .get("trigger_datetime")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        });
    let trigger_context = request
        .get("trigger_context")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    Some((
        name.to_string(),
        text.to_string(),
        trigger_time,
        trigger_context,
    ))
}

fn create_manual_memory(
    config: &AppConfig,
    name: &str,
    text: &str,
) -> anyhow::Result<MemoryRecord> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    if find_active_memory_by_name(&connection, name)?.is_some() {
        return Err(anyhow!("Memory '{name}' already exists"));
    }
    create_memory_file(&config.memory_dir, name, text)?;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))?;
    sync_memory_context_after_mutation(config, name, Some(name))?;

    let reopened = open_sqlite_connection(&config.database_path)?;
    find_active_memory_by_name(&reopened, name)?
        .ok_or_else(|| anyhow!("created memory '{name}' not found after bootstrap"))
}

fn create_due_item_from_request(
    config: &AppConfig,
    name: &str,
    text: &str,
    trigger_time: Option<&str>,
    trigger_context: Option<&str>,
) -> anyhow::Result<AgendaItemRecord> {
    if trigger_time.is_none() && trigger_context.is_none() {
        return Err(anyhow!(
            "Either trigger_time or trigger_context must be provided for due items"
        ));
    }
    if let Some(trigger_time) = trigger_time {
        let parsed =
            parse_trigger_datetime_for_validation(trigger_time).map_err(|error| anyhow!(error))?;
        if parsed < Utc::now() {
            return Err(anyhow!(
                "Attempted to create a due item for {trigger_time}, which is in the past. The current time is {}",
                Utc::now().to_rfc3339()
            ));
        }
    }

    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    if find_active_agenda_item_by_name(&connection, name)?.is_some() {
        return Err(anyhow!("Active due item '{name}' already exists"));
    }

    let canonical_path = config
        .agenda_dir
        .join(format!("{}.md", derive_agenda_item_name(name)));
    if canonical_path.exists() {
        std::fs::remove_file(&canonical_path)?;
    }

    create_agenda_file(
        &config.agenda_dir,
        name,
        text,
        None,
        trigger_time,
        trigger_context,
    )?;
    elroy_db::bootstrap_database(&BootstrapPlan::from_config(config))?;
    sync_due_item_context_after_mutation(config, name, Some(name))?;

    let reopened = open_sqlite_connection(&config.database_path)?;
    find_active_agenda_item_by_name(&reopened, name)?
        .ok_or_else(|| anyhow!("created due item '{name}' not found after bootstrap"))
}

pub fn ingest_memo_from_config(
    config: &AppConfig,
    text: &str,
) -> anyhow::Result<Vec<IngestedMemoItem>> {
    let provider_config =
        provider_config_from_app_config(config).map_err(|error| anyhow!(error))?;
    let model = best_effort_provider_model(Some(&provider_config), &config.assistant_name)
        .ok_or_else(|| anyhow!("failed to create live provider model"))?;

    let mut previous_attempt_error: Option<String> = None;
    for _attempt in 1..=3 {
        let augmented = augment_text_from_config(config, text)?;
        let system = if let Some(previous_attempt_error) = &previous_attempt_error {
            format!(
                "Your task is to convert text into either a due item or a memory.\n\n\
                A memory is a generic note, without a specific time or context that it should be recalled.\n\n\
                A due item is similar to a memory, but it should be something the user wants or needs to surface in a specific context or time.\n\n\
                Where possible, convert any relative dates or times to ISO 8601 format. Note the local time is {}, or {} UTC.\n\n\
                If creating a due item with a trigger_time, note that due items cannot be created for time in the past.\n\n\
                You should provide EITHER a create_due_item_request OR a create_memory_request, not both. Set the field you don't need to null.\n\n\
                A previous attempt at this task failed with error: {}",
                Local::now().to_rfc3339(),
                Utc::now().to_rfc3339(),
                previous_attempt_error
            )
        } else {
            format!(
                "Your task is to convert text into either a due item or a memory.\n\n\
                A memory is a generic note, without a specific time or context that it should be recalled.\n\n\
                A due item is similar to a memory, but it should be something the user wants or needs to surface in a specific context or time.\n\n\
                Where possible, convert any relative dates or times to ISO 8601 format. Note the local time is {}, or {} UTC.\n\n\
                If creating a due item with a trigger_time, note that due items cannot be created for time in the past.\n\n\
                You should provide EITHER a create_due_item_request OR a create_memory_request, not both. Set the field you don't need to null.",
                Local::now().to_rfc3339(),
                Utc::now().to_rfc3339()
            )
        };

        let response = collect_assistant_response(model.next_events(ConversationRequest {
            user_message: &augmented,
            tools: &[],
            transcript: &[ConversationMessage::new(
                MessageRole::User,
                format!("{system}\n\n{augmented}"),
            )],
            force_tool: None,
        })?);
        let parsed = serde_json::from_str::<serde_json::Value>(response.trim())
            .map_err(|error| anyhow!("ingest_memo returned invalid JSON: {error}"))?;

        let memory_request = parse_memory_request(&parsed);
        let due_item_request = parse_due_item_request(&parsed);
        match (memory_request, due_item_request) {
            (Some((name, text)), None) => {
                let memory = create_manual_memory(config, &name, &text)?;
                return Ok(vec![IngestedMemoItem::Memory(memory)]);
            }
            (None, Some((name, text, trigger_time, trigger_context))) => {
                match create_due_item_from_request(
                    config,
                    &name,
                    &text,
                    trigger_time.as_deref(),
                    trigger_context.as_deref(),
                ) {
                    Ok(due_item) => return Ok(vec![IngestedMemoItem::DueItem(due_item)]),
                    Err(error) => {
                        previous_attempt_error = Some(error.to_string());
                        continue;
                    }
                }
            }
            (None, None) => return Ok(Vec::new()),
            (Some(_), Some(_)) => {
                return Err(anyhow!(
                    "ingest_memo returned both create_memory_request and create_due_item_request"
                ));
            }
        }
    }

    Err(anyhow!(
        "Abandoning ingest_memo after 3 attempts: {}",
        previous_attempt_error.unwrap_or_else(|| "unknown error".to_string())
    ))
}

pub fn get_memories_from_config(
    config: &AppConfig,
    memory_ids: &[i64],
) -> anyhow::Result<Vec<MemoryRecord>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    Ok(load_memories_by_ids(&connection, memory_ids)?)
}

pub fn get_memory_by_name_from_config(
    config: &AppConfig,
    memory_name: &str,
) -> anyhow::Result<Option<MemoryRecord>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    Ok(find_active_memory_by_name_in_scope(
        &connection,
        memory_name,
        &config.memory_dir,
    )?)
}

pub fn get_active_memories_from_config(config: &AppConfig) -> anyhow::Result<Vec<MemoryRecord>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    Ok(list_active_memories_in_scope(
        &connection,
        &config.memory_dir,
        10_000,
    )?)
}

pub fn get_source_list_for_memory_structured_from_config(
    config: &AppConfig,
    memory_name: &str,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let memory = find_active_memory_by_name_in_scope(&connection, memory_name, &config.memory_dir)?
        .ok_or_else(|| anyhow!("Memory '{memory_name}' not found for the current user."))?;
    let (frontmatter, _) = read_memory_parts(std::path::Path::new(&memory.file_path))?;
    Ok(list_memory_sources(frontmatter.as_deref()))
}

pub fn get_source_content_for_memory_text_from_config(
    config: &AppConfig,
    memory_name: &str,
    index: usize,
) -> anyhow::Result<String> {
    get_source_content_for_memory_from_config(config, memory_name, index)
}

pub fn get_relevant_memories_and_due_items_from_config(
    config: &AppConfig,
    query: &str,
) -> anyhow::Result<Vec<RelevantRecallItem>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;

    let provider_config = elroy_config::fast_provider_config_from_app_config(config).ok();
    let embedding_provider_config =
        elroy_config::embedding_provider_config_from_app_config(config).ok();

    let relevance_model =
        best_effort_provider_model(provider_config.as_ref(), &config.assistant_name);
    let embedding_client = best_effort_embedding_client(embedding_provider_config.as_ref());
    let query_embedding = embedding_client
        .as_ref()
        .and_then(|client| client.embed(query).ok());

    let selection_clients = RecallSelectionClients {
        limit: 2,
        embedding_distance_threshold: Some(config.l2_memory_relevance_distance_threshold as f32),
        recency_weight: config.recency_weight as f32,
        connection: Some(&connection),
        relevance_model: relevance_model
            .as_ref()
            .map(|model| model as &dyn ModelClient),
        embedding_client: embedding_client.as_ref(),
        query_embedding: query_embedding.as_deref(),
        now_iso: None,
    };

    let source_fetch_limit = semantic_recall_source_fetch_limit(
        6,
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
        list_all_active_memories_in_scope(&connection, &config.memory_dir)?
    } else {
        list_active_memories_in_scope(&connection, &config.memory_dir, 10_000)?
    };
    let due_items = elroy_db::list_active_due_items(&connection, source_fetch_limit)?;
    let agenda_items = elroy_db::list_active_plain_agenda_items(&connection, source_fetch_limit)?;

    let relevant_memories =
        select_relevant_recall_memories(query, &memories, &[], selection_clients);
    let relevant_due_items = select_relevant_recall_due_items(query, &due_items, selection_clients);
    let relevant_agenda_items =
        select_relevant_recall_agenda_items(query, &agenda_items, selection_clients);

    let mut items = relevant_memories
        .into_iter()
        .cloned()
        .map(RelevantRecallItem::Memory)
        .collect::<Vec<_>>();
    items.extend(
        relevant_due_items
            .into_iter()
            .cloned()
            .map(RelevantRecallItem::DueItem),
    );
    items.extend(
        relevant_agenda_items
            .into_iter()
            .cloned()
            .map(RelevantRecallItem::AgendaItem),
    );
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::{
        IngestedMemoItem, RelevantRecallItem, archive_memory_file, create_memory_file,
        create_memory_file_with_frontmatter, get_active_memories_from_config,
        get_memories_from_config, get_memory_by_name_from_config,
        get_relevant_memories_and_due_items_from_config,
        get_source_content_for_memory_text_from_config,
        get_source_list_for_memory_structured_from_config, ingest_memo_from_config,
        read_memory_parts, sanitize_filename, update_memory_body,
    };
    use elroy_config::AppConfig;
    use elroy_db::{
        BootstrapPlan, LOCAL_USER_TOKEN, find_active_agenda_item_by_name, load_context_messages,
        open_sqlite_connection,
    };

    #[test]
    fn sanitize_filename_compacts_words() {
        assert_eq!(sanitize_filename("Runner Notes"), "runner_notes");
        assert_eq!(sanitize_filename("!!!"), "item");
    }

    #[test]
    fn file_create_update_and_archive_work() {
        let unique = format!(
            "elroy-rs-memory-crate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let archive = root.join("archive");
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_memory_file(&root, "Runner Notes", "Remember this")
            .expect("memory file should be created");
        update_memory_body(&path, "Updated text").expect("memory should update");
        let archived = archive_memory_file(&path, &archive).expect("memory should archive");

        assert!(archived.exists());
        let content = std::fs::read_to_string(archived).expect("archived memory should read");
        assert!(content.contains("Updated text"));

        std::fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn file_create_with_frontmatter_round_trips_metadata() {
        let unique = format!(
            "elroy-rs-memory-frontmatter-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_memory_file_with_frontmatter(
            &root,
            "Runner Notes",
            "Remember this",
            Some("source_type: ContextMessageSet\nmessage_ids_json: [1,2,3]"),
        )
        .expect("memory file should be created");
        let (frontmatter, body) = read_memory_parts(&path).expect("memory parts should read");

        assert_eq!(
            frontmatter.as_deref(),
            Some("source_type: ContextMessageSet\nmessage_ids_json: [1,2,3]")
        );
        assert_eq!(body, "Remember this");

        std::fs::remove_dir_all(root).expect("root should be removed");
    }

    fn unique_home(prefix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        ))
    }

    #[test]
    fn ingest_memo_from_config_can_create_memory_and_pin_it_to_context() {
        let home = unique_home("elroy-rs-memory-ingest-memory");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        let mut server = mockito::Server::new();
        let ingest_mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::Regex(
                "convert text into either a due item or a memory".to_string(),
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
                                "create_memory_request": {
                                    "name": "Ted gift idea",
                                    "text": "Ted, your best friend, would appreciate a first edition of War and Peace."
                                },
                                "create_due_item_request": null
                            }).to_string()
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

        let result = ingest_memo_from_config(&config, "Ted bday gift: War and Peace")
            .expect("ingest should succeed");
        assert!(matches!(&result[..], [IngestedMemoItem::Memory(_)]));

        let mut connection =
            open_sqlite_connection(&database_path).expect("sqlite connection should open");
        let transcript = load_context_messages(&mut connection, LOCAL_USER_TOKEN)
            .expect("context messages should load");
        assert!(transcript.iter().any(|message| {
            message
                .tool_call_id
                .as_deref()
                .is_some_and(|id| id == "context-memory:ted gift idea")
        }));

        ingest_mock.assert();
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn ingest_memo_from_config_can_retry_invalid_due_item_and_create_contextual_due_item() {
        let home = unique_home("elroy-rs-memory-ingest-due-item");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        let mut server = mockito::Server::new();
        let invalid_due_item = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::Regex(
                "convert text into either a due item or a memory".to_string(),
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
                                "create_memory_request": null,
                                "create_due_item_request": {
                                    "name": "stress stretch",
                                    "text": "Do 20 push-ups",
                                    "trigger_time": "2000-01-01 09:00",
                                    "trigger_context": "when I mention feeling stressed"
                                }
                            }).to_string()
                        }]
                    }]
                })
                .to_string(),
            )
            .expect(1)
            .create();
        let valid_due_item = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::Regex(
                "A previous attempt at this task failed with error".to_string(),
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
                                "create_memory_request": null,
                                "create_due_item_request": {
                                    "name": "stress stretch",
                                    "text": "Do 20 push-ups",
                                    "trigger_context": "when I mention feeling stressed"
                                }
                            }).to_string()
                        }]
                    }]
                })
                .to_string(),
            )
            .expect(1)
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

        let result = ingest_memo_from_config(
            &config,
            "Create a reminder for stress stretching when I mention feeling stressed.",
        )
        .expect("ingest should succeed");
        assert!(matches!(
            &result[..],
            [IngestedMemoItem::DueItem(item)] if item.name == "stress stretch"
        ));

        let connection =
            open_sqlite_connection(&database_path).expect("sqlite connection should open");
        let due_item = find_active_agenda_item_by_name(&connection, "stress stretch")
            .expect("lookup should work")
            .expect("due item should exist");
        assert_eq!(
            due_item.trigger_context.as_deref(),
            Some("when I mention feeling stressed")
        );

        invalid_due_item.assert();
        valid_due_item.assert();
        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn get_memories_from_config_returns_requested_memory_ids() {
        let home = unique_home("elroy-rs-memory-get-memories");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        create_memory_file(
            &memory_dir,
            "Test Memory 1",
            "This is the first test memory",
        )
        .expect("first memory should be created");
        create_memory_file(
            &memory_dir,
            "Test Memory 2",
            "This is the second test memory",
        )
        .expect("second memory should be created");
        create_memory_file(
            &memory_dir,
            "Test Memory 3",
            "This is the third test memory",
        )
        .expect("third memory should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let connection =
            open_sqlite_connection(&config.database_path).expect("sqlite connection should open");
        let memory_1 = elroy_db::find_active_memory_by_name(&connection, "Test Memory 1")
            .expect("first memory lookup should succeed")
            .expect("first memory should exist");
        let memory_2 = elroy_db::find_active_memory_by_name(&connection, "Test Memory 2")
            .expect("second memory lookup should succeed")
            .expect("second memory should exist");
        let memory_3 = elroy_db::find_active_memory_by_name(&connection, "Test Memory 3")
            .expect("third memory lookup should succeed")
            .expect("third memory should exist");

        let retrieved = get_memories_from_config(&config, &[memory_1.id, memory_3.id])
            .expect("memory lookup by ids should succeed");
        let retrieved_names = retrieved
            .iter()
            .map(|memory| memory.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            retrieved_names,
            vec![memory_1.name.as_str(), memory_3.name.as_str()]
        );
        assert!(!retrieved_names.contains(&memory_2.name.as_str()));

        let empty = get_memories_from_config(&config, &[]).expect("empty lookup should succeed");
        assert!(empty.is_empty());

        let missing =
            get_memories_from_config(&config, &[99999]).expect("missing lookup should succeed");
        assert!(missing.is_empty());

        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_query_helpers_return_active_memories_in_scope() {
        let home = unique_home("elroy-rs-memory-query-helpers");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let archive_dir = memory_dir.join("archive");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        std::fs::create_dir_all(&archive_dir).expect("archive dir should be created");

        let archived_path = create_memory_file(&memory_dir, "Archived Memory", "Old archived text")
            .expect("archived memory should be created");
        archive_memory_file(&archived_path, &archive_dir).expect("memory should archive");
        create_memory_file(
            &memory_dir,
            "Test Memory 1",
            "This is the first test memory",
        )
        .expect("first memory should be created");
        create_memory_file(
            &memory_dir,
            "Test Memory 2",
            "This is the second test memory",
        )
        .expect("second memory should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let memory = get_memory_by_name_from_config(&config, "test memory 1")
            .expect("lookup should succeed")
            .expect("memory should exist");
        assert_eq!(memory.name, "test memory 1");

        let active_memories =
            get_active_memories_from_config(&config).expect("active memories should load");
        let active_names = active_memories
            .iter()
            .map(|memory| memory.name.as_str())
            .collect::<Vec<_>>();
        assert!(active_names.contains(&"test memory 1"));
        assert!(active_names.contains(&"test memory 2"));
        assert!(!active_names.contains(&"archived memory"));

        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn source_helpers_return_structured_memory_and_context_sources() {
        let home = unique_home("elroy-rs-memory-source-helpers");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        create_memory_file(&memory_dir, "Running progress", "I ran a marathon today")
            .expect("first source memory should be created");
        create_memory_file(&memory_dir, "Run today", "I ran 24 miles today")
            .expect("second source memory should be created");
        let running_progress_path = memory_dir.join("running_progress.md");
        let run_today_path = memory_dir.join("run_today.md");
        create_memory_file_with_frontmatter(
            &memory_dir,
            "Running summary",
            "The user ran a marathon and later reported running 24 miles in total.",
            elroy_recall::memory_source_frontmatter(&[
                ("Running progress", running_progress_path.as_path()),
                ("Run today", run_today_path.as_path()),
            ])
            .as_deref(),
        )
        .expect("consolidated memory should be created");
        create_memory_file_with_frontmatter(
            &memory_dir,
            "Context-backed memory",
            "I ran a marathon today",
            Some("source_type: ContextMessageSet\nmessage_ids_json: [1]"),
        )
        .expect("context-backed memory should be created");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let mut connection =
            open_sqlite_connection(&database_path).expect("sqlite connection should open");
        elroy_db::replace_context_messages(
            &mut connection,
            elroy_db::LOCAL_USER_TOKEN,
            &[elroy_llm::ConversationMessage {
                role: elroy_llm::MessageRole::User,
                content: Some("Hello, I ran a marathon today!".to_string()),
                chat_model: None,
                id: Some(1),
                created_at_unix: 1,
                tool_calls: None,
                tool_call_id: None,
            }],
        )
        .expect("context messages should persist");

        let source_list =
            get_source_list_for_memory_structured_from_config(&config, "Running summary")
                .expect("source list should load");
        assert!(source_list.contains(&("Memory".to_string(), "Running progress".to_string())));
        assert!(source_list.contains(&("Memory".to_string(), "Run today".to_string())));

        let source_index = source_list
            .iter()
            .position(|entry| entry == &("Memory".to_string(), "Running progress".to_string()))
            .expect("running progress source should exist");
        let source_content = get_source_content_for_memory_text_from_config(
            &config,
            "Running summary",
            source_index,
        )
        .expect("source content should load");
        assert!(source_content.contains("Running progress"));

        let context_source_list =
            get_source_list_for_memory_structured_from_config(&config, "Context backed memory")
                .expect("context source list should load");
        assert_eq!(
            context_source_list,
            vec![("ContextMessageSet".to_string(), "1".to_string())]
        );
        let context_source_content =
            get_source_content_for_memory_text_from_config(&config, "Context backed memory", 0)
                .expect("context source content should load");
        assert!(context_source_content.contains("Hello, I ran a marathon today"));

        std::fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn relevant_recall_helper_returns_memory_due_item_and_agenda_item_matches() {
        let home = unique_home("elroy-rs-memory-relevant-query");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        create_memory_file(
            &memory_dir,
            "Store memory",
            "Remember to buy milk at the store",
        )
        .expect("memory should be created");
        std::fs::write(
            agenda_dir.join("milk_reminder.md"),
            "---\ndate: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: when I mention shopping\n---\n\nBuy oat milk at the store\n",
        )
        .expect("due item should be written");
        std::fs::write(
            agenda_dir.join("shopping_list.md"),
            "---\ndate: 2099-01-01\ncompleted: false\nstatus: created\n---\n\nStore receipt audit\n",
        )
        .expect("agenda item should be written");

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let items = get_relevant_memories_and_due_items_from_config(&config, "store milk")
            .expect("relevant items should load");

        assert!(items.iter().any(|item| matches!(
            item,
            RelevantRecallItem::Memory(memory) if memory.name == "store memory"
        )));
        assert!(items.iter().any(|item| matches!(
            item,
            RelevantRecallItem::DueItem(item) if item.name == "milk reminder"
        )));
        assert!(items.iter().any(|item| matches!(
            item,
            RelevantRecallItem::AgendaItem(item) if item.name == "shopping list"
        )));

        std::fs::remove_dir_all(home).expect("home should be removed");
    }
}
