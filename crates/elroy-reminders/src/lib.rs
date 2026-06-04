// Due-item surfacing, reminder selection, and synthetic context message generation.

use chrono::Utc;
use elroy_config::AppConfig;
use elroy_core::{ConversationRequest, ModelClient};
use elroy_db::AgendaItemRecord;
use elroy_db::{
    MemoryEmbeddingRecord, find_active_agenda_item_by_name, list_active_due_items,
    list_inactive_due_items, load_memory_embeddings_for_paths, open_sqlite_connection,
    run_migrations, upsert_memory_embedding,
};
use elroy_llm::{ConversationMessage, MessageRole, StreamEvent};
use elroy_recall::{
    RecallSelectionClients, build_recall_query, context_due_item_tool_messages,
    parse_relevance_filter_response, parse_sidebar_trigger_datetime, select_due_items_by_overlap,
    synthetic_tool_context_messages, transcript_contains_recalled_agenda_item,
};
use std::collections::HashMap;

pub struct DueItemSurfacingContext {
    pub timed_due_item_context: Vec<ConversationMessage>,
    pub contextual_due_item_context: Vec<ConversationMessage>,
}

pub fn get_db_due_item_by_name_from_config(
    config: &AppConfig,
    name: &str,
) -> anyhow::Result<Option<AgendaItemRecord>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    Ok(find_active_agenda_item_by_name(&connection, name)?
        .filter(|item| item.trigger_datetime.is_some() || item.trigger_context.is_some()))
}

pub fn get_active_due_items_from_config(
    config: &AppConfig,
) -> anyhow::Result<Vec<AgendaItemRecord>> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    Ok(list_active_due_items(&connection, 10_000)?)
}

pub fn get_due_timed_items_from_config(
    config: &AppConfig,
) -> anyhow::Result<Vec<AgendaItemRecord>> {
    let now_iso = Utc::now().to_rfc3339();
    Ok(get_active_due_items_from_config(config)?
        .into_iter()
        .filter(|item| {
            item.trigger_datetime
                .as_deref()
                .is_some_and(|dt| dt <= now_iso.as_str())
        })
        .collect())
}

pub fn get_due_items_from_config(
    config: &AppConfig,
    include_completed: bool,
) -> anyhow::Result<Vec<AgendaItemRecord>> {
    if !include_completed {
        return get_active_due_items_from_config(config);
    }

    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let mut items = list_active_due_items(&connection, 10_000)?;
    items.extend(
        list_inactive_due_items(&connection, 10_000)?
            .into_iter()
            .filter(|item| {
                item.status
                    .as_deref()
                    .is_some_and(|status| status == "completed")
            }),
    );
    Ok(items)
}

pub fn get_active_due_item_names_from_config(config: &AppConfig) -> anyhow::Result<Vec<String>> {
    Ok(get_active_due_items_from_config(config)?
        .into_iter()
        .map(|item| item.name)
        .collect())
}

pub fn get_due_item_by_name_from_config(
    config: &AppConfig,
    name: &str,
) -> anyhow::Result<Option<String>> {
    Ok(get_db_due_item_by_name_from_config(config, name)?.map(|item| item.body))
}

pub fn get_due_item_context_messages_from_config(
    config: &AppConfig,
) -> anyhow::Result<Vec<ConversationMessage>> {
    Ok(due_item_context_messages(&get_due_timed_items_from_config(
        config,
    )?))
}

pub fn due_item_context_messages(items: &[AgendaItemRecord]) -> Vec<ConversationMessage> {
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

const SEMANTIC_REMINDER_CANDIDATE_LIMIT: usize = 100;

fn semantic_reminder_candidate_limit(selection_clients: RecallSelectionClients<'_>) -> usize {
    if selection_clients.relevance_model.is_some() || selection_clients.embedding_client.is_some() {
        selection_clients
            .limit
            .saturating_mul(3)
            .max(SEMANTIC_REMINDER_CANDIDATE_LIMIT)
    } else {
        selection_clients
            .limit
            .saturating_mul(3)
            .max(selection_clients.limit)
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

fn recency_penalty(updated_at_unix: i64, now_unix: i64, recency_weight: f32) -> f32 {
    if recency_weight <= 0.0 {
        return 0.0;
    }
    let age_seconds = now_unix.saturating_sub(updated_at_unix).max(0) as f32;
    let age_years = age_seconds / (86_400.0 * 365.0);
    recency_weight * age_years
}

fn embedding_rank_due_item_candidates<'a>(
    query: &str,
    candidates: impl IntoIterator<Item = &'a AgendaItemRecord>,
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
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
    let mut embedding_cache: HashMap<String, MemoryEmbeddingRecord> = selection_clients
        .connection
        .and_then(|connection| {
            let paths = candidates
                .iter()
                .map(|candidate| candidate.file_path.clone())
                .collect::<Vec<_>>();
            load_memory_embeddings_for_paths(connection, &paths).ok()
        })
        .unwrap_or_default();

    let mut ranked = candidates
        .into_iter()
        .filter_map(|candidate| {
            let embedding_text = agenda_item_embedding_text(candidate);
            let file_path = candidate.file_path.as_str();
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
                    candidate.updated_at_unix,
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

fn recent_contextual_due_item_candidates<'a>(
    due_items: &'a [AgendaItemRecord],
    limit: usize,
    now_iso: &str,
) -> Vec<&'a AgendaItemRecord> {
    let mut items = due_items
        .iter()
        .filter(|item| item.trigger_context.is_some())
        .filter(|item| {
            item.trigger_datetime
                .as_deref()
                .is_none_or(|trigger_datetime| trigger_datetime > now_iso)
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        right
            .updated_at_unix
            .cmp(&left.updated_at_unix)
            .then_with(|| left.name.cmp(&right.name))
    });
    items.into_iter().take(limit).collect()
}

fn filter_due_item_candidates_for_relevance<'a>(
    model: Option<&dyn ModelClient>,
    query: &str,
    candidates: Vec<&'a AgendaItemRecord>,
) -> Vec<&'a AgendaItemRecord> {
    let Some(model) = model else {
        return candidates;
    };
    if candidates.is_empty() {
        return candidates;
    }

    let responses = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| format!("{index}. {}", agenda_item_embedding_text(candidate)))
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

pub fn select_contextual_due_items_by_overlap<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    limit: usize,
    skip_time_due_before: Option<&str>,
) -> Vec<&'a AgendaItemRecord> {
    select_due_items_by_overlap(prompt, due_items, limit, skip_time_due_before)
        .into_iter()
        .filter(|item| item.trigger_context.is_some())
        .collect()
}

pub fn select_recalled_due_items<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    select_contextual_due_items_by_overlap(prompt, due_items, limit, Some(now_iso))
}

pub fn select_relevant_contextual_due_items<'a>(
    query: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = semantic_reminder_candidate_limit(selection_clients);
    let overlap_candidates =
        select_contextual_due_items_by_overlap(query, due_items, candidate_limit, Some(now_iso));
    let candidates = if selection_clients.relevance_model.is_some() {
        let mut merged_candidates = overlap_candidates;
        for candidate in embedding_rank_due_item_candidates(
            query,
            due_items
                .iter()
                .filter(|item| item.trigger_context.is_some()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
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
        let embedding_candidates = embedding_rank_due_item_candidates(
            query,
            due_items
                .iter()
                .filter(|item| item.trigger_context.is_some()),
            RecallSelectionClients {
                limit: candidate_limit,
                ..selection_clients
            },
        );
        if embedding_candidates.is_empty() {
            overlap_candidates
        } else {
            embedding_candidates
        }
    } else {
        overlap_candidates
    };
    filter_due_item_candidates_for_relevance(selection_clients.relevance_model, query, candidates)
        .into_iter()
        .take(selection_clients.limit)
        .collect()
}

pub fn build_due_item_surfacing_context(
    prompt: &str,
    existing_transcript: &[ConversationMessage],
    recall_context: &[ConversationMessage],
    timed_due_items: &[AgendaItemRecord],
    all_due_items: &[AgendaItemRecord],
    now_iso: &str,
    selection_clients: RecallSelectionClients<'_>,
) -> DueItemSurfacingContext {
    let timed_due_item_context = due_item_context_messages(timed_due_items);
    let mut due_item_dedupe_transcript = existing_transcript.to_vec();
    due_item_dedupe_transcript.extend(recall_context.iter().cloned());
    due_item_dedupe_transcript.extend(timed_due_item_context.iter().cloned());
    let contextual_due_item_context = recall_due_item_context_messages(
        prompt,
        &due_item_dedupe_transcript,
        all_due_items,
        now_iso,
        selection_clients,
    );

    DueItemSurfacingContext {
        timed_due_item_context,
        contextual_due_item_context,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use elroy_config::AppConfig;
    use elroy_core::{ConversationRequest, ModelClient, StreamingModelClient};
    use elroy_db::{AgendaItemRecord, BootstrapPlan, bootstrap_database};
    use elroy_llm::{ConversationMessage, EmbeddingProviderConfig, MessageRole, StreamEvent};
    use elroy_recall::{
        RecallSelectionClients, best_effort_embedding_client, context_due_item_tool_messages,
        synthetic_tool_context_messages,
    };

    use super::{
        due_item_context_messages, get_active_due_item_names_from_config,
        get_active_due_items_from_config, get_db_due_item_by_name_from_config,
        get_due_item_by_name_from_config, get_due_item_context_messages_from_config,
        get_due_items_from_config, get_due_timed_items_from_config,
        recall_due_item_context_messages, select_contextual_due_items_by_overlap,
        select_recalled_due_items, select_relevant_contextual_due_items,
    };

    fn due_item(
        id: i64,
        name: &str,
        body: &str,
        trigger_datetime: Option<&str>,
        trigger_context: Option<&str>,
        updated_at_unix: i64,
    ) -> AgendaItemRecord {
        AgendaItemRecord {
            id,
            legacy_frontmatter_id: None,
            name: name.to_string(),
            file_path: format!("/tmp/{}.md", name.replace(' ', "_").to_ascii_lowercase()),
            agenda_date: Some("unscheduled".to_string()),
            is_completed: false,
            status: Some("created".to_string()),
            trigger_datetime: trigger_datetime.map(str::to_string),
            trigger_context: trigger_context.map(str::to_string),
            closing_comment: None,
            checklist_total: 0,
            checklist_completed: 0,
            body: body.to_string(),
            is_active: true,
            updated_at_unix,
        }
    }

    #[derive(Default)]
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
    fn due_item_context_messages_create_synthetic_tool_context() {
        let messages = due_item_context_messages(&[due_item(
            1,
            "call mom",
            "Call mom tonight",
            Some("2000-01-01T09:00:00"),
            Some("after dinner"),
            1,
        )]);

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
        let messages = due_item_context_messages(&[due_item(
            1,
            "call mom",
            "Call mom tonight",
            None,
            Some("after dinner"),
            1,
        )]);

        assert!(messages.is_empty());
    }

    #[test]
    fn due_item_context_messages_include_multiple_due_items() {
        let messages = due_item_context_messages(&[
            due_item(
                1,
                "call mom",
                "Call mom tonight",
                Some("2000-01-01T09:00:00"),
                None,
                1,
            ),
            due_item(
                2,
                "pay rent",
                "Pay rent",
                Some("2000-01-02T09:00:00"),
                None,
                2,
            ),
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
    fn recall_due_item_context_messages_creates_contextual_tool_message() {
        let transcript = vec![ConversationMessage::new(
            MessageRole::Assistant,
            "Tell me when payroll follows up.",
        )];
        let due_items = vec![due_item(
            1,
            "Payroll Follow-up",
            "Reply to payroll",
            None,
            Some("after payroll email"),
            20,
        )];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
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
        let transcript = context_due_item_tool_messages(&due_item(
            1,
            "payroll follow up",
            "Reply to payroll",
            None,
            Some("after payroll email"),
            20,
        ));
        let due_items = vec![due_item(
            1,
            "payroll follow up",
            "Reply to payroll",
            None,
            Some("after payroll email"),
            20,
        )];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
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
        );

        assert!(messages.is_empty());
    }

    #[test]
    fn recall_due_item_context_messages_skip_due_items_already_present_in_recall_metadata() {
        let transcript = synthetic_tool_context_messages(
            "bootstrap-memory-recall",
            "get_fast_recall",
            "{}",
            r##"{
  "content": "# payroll follow up\nReply to payroll",
  "recall_metadata": [
    {
      "memory_type": "AgendaItem",
      "memory_id": 1,
      "name": "payroll follow up"
    }
  ]
}"##,
        );
        let due_items = vec![due_item(
            1,
            "payroll follow up",
            "Reply to payroll",
            None,
            Some("after payroll email"),
            20,
        )];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
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
        );

        assert!(messages.is_empty());
    }

    #[test]
    fn recall_due_item_context_messages_do_not_skip_same_name_due_item_with_different_id() {
        let transcript = synthetic_tool_context_messages(
            "bootstrap-memory-recall",
            "get_fast_recall",
            "{}",
            r##"{
  "content": "# payroll follow up\nOld payroll reminder",
  "recall_metadata": [
    {
      "memory_type": "AgendaItem",
      "memory_id": 1,
      "name": "payroll follow up"
    }
  ]
}"##,
        );
        let due_items = vec![due_item(
            2,
            "payroll follow up",
            "Reply to payroll",
            None,
            Some("after payroll email"),
            20,
        )];

        let messages = recall_due_item_context_messages(
            "I just got the payroll email",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
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
        );

        assert_eq!(messages.len(), 2);
        assert!(
            messages[1]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("Reply to payroll"))
        );
    }

    #[test]
    fn recall_due_item_context_messages_can_broaden_beyond_overlap_via_relevance_model() {
        let transcript = vec![ConversationMessage::new(
            MessageRole::Assistant,
            "Tell me what to bring to practice.",
        )];
        let due_items = vec![due_item(
            1,
            "Practice Reminder",
            "Bring the resistance bands",
            None,
            Some("before basketball practice"),
            20,
        )];
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: r#"{"answers":[true],"reasoning":"This reminder is semantically relevant."}"#
                .to_string(),
        }]]);

        let messages = recall_due_item_context_messages(
            "What gear should I bring?",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
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

        assert_eq!(messages.len(), 2);
        assert!(messages[1].content.as_deref().is_some_and(|content| {
            content.contains("Practice Reminder") && content.contains("resistance bands")
        }));
    }

    #[test]
    fn recall_due_item_context_messages_can_prefer_semantic_contextual_match_over_weaker_overlap() {
        struct SemanticReminderRelevanceModel;

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

        let transcript = vec![ConversationMessage::new(
            MessageRole::Assistant,
            "What do you need help with?",
        )];
        let due_items = vec![
            due_item(
                1,
                "Gear Inventory",
                "Review the storage locker spreadsheet.",
                None,
                Some("after equipment handoff"),
                10,
            ),
            due_item(
                2,
                "Practice Reminder",
                "Bring the resistance bands",
                None,
                Some("before basketball practice"),
                20,
            ),
        ];
        let messages = recall_due_item_context_messages(
            "What gear should I bring?",
            &transcript,
            &due_items,
            "2026-05-15T12:00:00",
            RecallSelectionClients {
                limit: 2,
                relevance_model: Some(&SemanticReminderRelevanceModel),
                embedding_client: None,
                embedding_distance_threshold: None,
                recency_weight: 0.0,
                connection: None,
                query_embedding: None,
                now_iso: None,
            },
        );

        let joined_tool_content = messages
            .iter()
            .filter(|message| message.role == MessageRole::Tool)
            .filter_map(|message| message.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined_tool_content.contains("Practice Reminder"));
        assert!(joined_tool_content.contains("Bring the resistance bands"));
        assert!(!joined_tool_content.contains("Gear Inventory"));
    }

    #[test]
    fn select_recalled_due_items_prefers_trigger_context_overlap() {
        let due_items = vec![
            due_item(
                1,
                "Payroll Follow-up",
                "Reply to payroll",
                None,
                Some("after payroll email"),
                20,
            ),
            due_item(2, "Dinner", "Call family", None, Some("after dinner"), 10),
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
    fn select_contextual_due_items_by_overlap_can_include_not_yet_due_items_for_search() {
        let due_items = vec![due_item(
            1,
            "Payroll Follow-up",
            "Reply to payroll",
            Some("2099-01-01T09:00:00"),
            Some("after payroll email"),
            20,
        )];

        let recalled = select_contextual_due_items_by_overlap(
            "I just got the payroll email",
            &due_items,
            2,
            None,
        );

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].name, "Payroll Follow-up");
    }

    #[test]
    fn select_recalled_due_items_skips_time_due_items() {
        let due_items = vec![due_item(
            1,
            "Payroll Follow-up",
            "Reply to payroll",
            Some("2000-01-01T09:00:00"),
            Some("after payroll email"),
            20,
        )];

        let recalled = select_recalled_due_items(
            "I just got the payroll email",
            &due_items,
            "2026-05-15T12:00:00",
            2,
        );

        assert!(recalled.is_empty());
    }

    #[test]
    fn select_relevant_contextual_due_items_can_surface_older_semantic_candidate_beyond_old_candidate_cap()
     {
        let mut due_items = (0..40)
            .map(|index| {
                due_item(
                    index + 1,
                    &format!("Recent Reminder {index}"),
                    &format!("Review the workout locker spreadsheet {index}."),
                    None,
                    Some("after equipment handoff"),
                    100 - index,
                )
            })
            .collect::<Vec<_>>();
        due_items.push(due_item(
            99,
            "Resistance Bands Reminder",
            "Pack resistance bands before drills.",
            None,
            Some("before basketball practice"),
            1,
        ));
        let mut answers = vec![false; 41];
        *answers.last_mut().expect("answers should exist") = true;
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: serde_json::json!({
                "answers": answers,
                "reasoning": "Only the older practice reminder is semantically relevant."
            })
            .to_string(),
        }]]);

        let relevant_due_items = select_relevant_contextual_due_items(
            "What belongs in my workout kit?",
            &due_items,
            "2026-05-15T12:00:00",
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

        assert_eq!(relevant_due_items.len(), 1);
        assert_eq!(relevant_due_items[0].name, "Resistance Bands Reminder");
        assert!(relevant_due_items[0].body.contains("resistance bands"));
    }

    #[test]
    fn select_relevant_contextual_due_items_can_surface_future_hybrid_semantic_candidate_via_embedding_without_relevance_model()
     {
        let due_items = vec![
            due_item(
                1,
                "Workout Kit Review",
                "Review the workout locker spreadsheet.",
                None,
                Some("after equipment handoff"),
                20,
            ),
            due_item(
                2,
                "Practice Reminder",
                "Pack resistance bands before drills.",
                Some("2099-05-20T09:00:00"),
                Some("before basketball practice"),
                10,
            ),
        ];

        let mut server = mockito::Server::new();
        let _query_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "input": "What belongs in my workout kit?"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [1.0, 0.0]}] }).to_string())
            .create();
        let _semantic_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(
                "Pack resistance bands before drills".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [1.0, 0.0]}] }).to_string())
            .create();
        let _overlap_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(
                "Review the workout locker spreadsheet".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [0.0, 1.0]}] }).to_string())
            .create();

        let embedding_client = best_effort_embedding_client(Some(&EmbeddingProviderConfig {
            model: "text-embedding-3-small".to_string(),
            api_key: "embedding-test-key".to_string(),
            base_url: format!("{}/embeddings", server.url()),
            timeout_seconds: 60,
        }))
        .expect("embedding client should build");

        let relevant_due_items = select_relevant_contextual_due_items(
            "What belongs in my workout kit?",
            &due_items,
            "2026-05-15T12:00:00",
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

        assert_eq!(relevant_due_items.len(), 1);
        assert_eq!(relevant_due_items[0].name, "Practice Reminder");
        assert!(relevant_due_items[0].body.contains("resistance bands"));
    }

    #[test]
    fn select_relevant_contextual_due_items_can_surface_semantic_contextual_task_via_embedding_without_relevance_model()
     {
        let due_items = vec![
            due_item(
                1,
                "Workout Kit Review",
                "Review the workout locker spreadsheet.",
                None,
                Some("after equipment handoff"),
                20,
            ),
            due_item(
                2,
                "Practice Packing Task",
                "Pack resistance bands before drills.",
                None,
                Some("before basketball practice"),
                10,
            ),
        ];

        let mut server = mockito::Server::new();
        let _query_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "input": "What belongs in my workout kit?"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [1.0, 0.0]}] }).to_string())
            .create();
        let _semantic_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(
                "Pack resistance bands before drills".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [1.0, 0.0]}] }).to_string())
            .create();
        let _overlap_embedding_mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::Regex(
                "Review the workout locker spreadsheet".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({ "data": [{"embedding": [0.0, 1.0]}] }).to_string())
            .create();

        let embedding_client = best_effort_embedding_client(Some(&EmbeddingProviderConfig {
            model: "text-embedding-3-small".to_string(),
            api_key: "embedding-test-key".to_string(),
            base_url: format!("{}/embeddings", server.url()),
            timeout_seconds: 60,
        }))
        .expect("embedding client should build");

        let relevant_due_items = select_relevant_contextual_due_items(
            "What belongs in my workout kit?",
            &due_items,
            "2026-05-15T12:00:00",
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

        assert_eq!(relevant_due_items.len(), 1);
        assert_eq!(relevant_due_items[0].name, "Practice Packing Task");
        assert!(relevant_due_items[0].body.contains("resistance bands"));
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

    fn write_due_item_file(
        agenda_dir: &std::path::Path,
        slug: &str,
        frontmatter: &str,
        body: &str,
    ) {
        std::fs::write(
            agenda_dir.join(format!("{slug}.md")),
            format!("---\n{frontmatter}\n---\n\n{body}\n"),
        )
        .expect("due item fixture should be written");
    }

    #[test]
    fn reminder_query_helpers_match_due_item_repository_cases() {
        let home = unique_home("elroy-rs-reminder-query-helpers");
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("elroy.db");
        std::fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        std::fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");

        write_due_item_file(
            &agenda_dir,
            "due_test",
            "date: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00",
            "This reminder is due",
        );
        write_due_item_file(
            &agenda_dir,
            "future_test",
            "date: unscheduled\ncompleted: false\nstatus: created\ntrigger_datetime: 2099-01-01T09:00:00",
            "This reminder is for tomorrow",
        );
        write_due_item_file(
            &agenda_dir,
            "contextual_test",
            "date: unscheduled\ncompleted: false\nstatus: created\ntrigger_context: when user mentions work",
            "Context-only reminder",
        );
        write_due_item_file(
            &agenda_dir,
            "completed_test",
            "date: unscheduled\ncompleted: true\nstatus: completed\ntrigger_datetime: 2000-01-01T10:00:00",
            "Completed reminder",
        );

        let mut config = AppConfig::defaults();
        config.home_dir = home.clone();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        bootstrap_database(&BootstrapPlan::from_config(&config)).expect("bootstrap should succeed");

        let active_due_items =
            get_active_due_items_from_config(&config).expect("active due items should load");
        assert!(active_due_items.iter().any(|item| item.name == "due test"));
        assert!(
            active_due_items
                .iter()
                .any(|item| item.name == "future test")
        );
        assert!(
            active_due_items
                .iter()
                .any(|item| item.name == "contextual test")
        );
        assert!(
            !active_due_items
                .iter()
                .any(|item| item.name == "completed test")
        );
        let active_due_item_names = get_active_due_item_names_from_config(&config)
            .expect("active due item names should load");
        assert!(active_due_item_names.contains(&"due test".to_string()));
        assert!(active_due_item_names.contains(&"future test".to_string()));
        assert!(active_due_item_names.contains(&"contextual test".to_string()));
        assert!(!active_due_item_names.contains(&"completed test".to_string()));

        let due_item = get_db_due_item_by_name_from_config(&config, "due test")
            .expect("due item query should succeed")
            .expect("due item should exist");
        assert_eq!(due_item.body, "This reminder is due");
        assert_eq!(
            get_due_item_by_name_from_config(&config, "due test").expect("due item body query"),
            Some("This reminder is due".to_string())
        );

        let due_timed_items =
            get_due_timed_items_from_config(&config).expect("due timed items should load");
        assert!(due_timed_items.iter().any(|item| item.name == "due test"));
        assert!(
            !due_timed_items
                .iter()
                .any(|item| item.name == "future test")
        );
        assert!(
            !due_timed_items
                .iter()
                .any(|item| item.name == "contextual test")
        );
        assert!(
            !due_timed_items
                .iter()
                .any(|item| item.name == "completed test")
        );

        let due_items_with_completed =
            get_due_items_from_config(&config, true).expect("due items should load");
        assert!(
            due_items_with_completed
                .iter()
                .any(|item| item.name == "due test")
        );
        assert!(
            due_items_with_completed
                .iter()
                .any(|item| item.name == "future test")
        );
        assert!(
            due_items_with_completed
                .iter()
                .any(|item| item.name == "contextual test")
        );
        assert!(
            due_items_with_completed
                .iter()
                .any(|item| item.name == "completed test")
        );

        let context_msgs = get_due_item_context_messages_from_config(&config)
            .expect("due item context messages should load");
        assert_eq!(context_msgs.len(), 2);
        assert!(
            context_msgs[1]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("This reminder is due"))
        );

        std::fs::remove_dir_all(home).expect("home should be removed");
    }
}
