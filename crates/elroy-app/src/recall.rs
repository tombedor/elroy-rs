use std::collections::HashSet;

use chrono::Utc;
use elroy_core::{ConversationRequest, ModelClient};
use elroy_db::{AgendaItemRecord, MemoryEmbeddingRecord, MemoryRecord, load_memory_embeddings_for_paths, upsert_memory_embedding};
use elroy_llm::{ConversationMessage, LiveEmbeddingClient, MessageRole, StreamEvent};
use serde_json::Value;

use crate::{
    format_agenda_item_recall_detail, format_memory_detail, excerpt,
    context_due_item_tool_messages, context_memory_tool_call_id, context_due_item_tool_call_id,
    context_task_tool_call_id, synthetic_tool_context_messages, AppError,
};

#[derive(Clone, Copy, Default)]
pub(crate) struct RecallModelClients<'a> {
    pub(crate) classifier_model: Option<&'a dyn ModelClient>,
    pub(crate) embedding_client: Option<&'a LiveEmbeddingClient>,
    pub(crate) embedding_distance_threshold: Option<f32>,
    pub(crate) recency_weight: f32,
    pub(crate) reflection_max_words: usize,
}

pub(crate) fn recall_model_clients(classifier_model: Option<&dyn ModelClient>) -> RecallModelClients<'_> {
    RecallModelClients {
        classifier_model,
        embedding_client: None,
        embedding_distance_threshold: None,
        recency_weight: 0.0,
        reflection_max_words: 100,
    }
}

pub(crate) struct RecallContext<'a> {
    pub(crate) transcript: &'a [ConversationMessage],
    pub(crate) memories: &'a [MemoryRecord],
    pub(crate) due_items: &'a [AgendaItemRecord],
    pub(crate) agenda_items: &'a [AgendaItemRecord],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryRecallDecision {
    pub(crate) needs_recall: bool,
    pub(crate) reasoning: String,
    pub(crate) used_llm: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RecalledItemRef {
    pub(crate) id: Option<i64>,
    pub(crate) name: String,
}

#[derive(Clone, Copy)]
pub(crate) struct RecallSelectionClients<'a> {
    pub(crate) limit: usize,
    pub(crate) relevance_model: Option<&'a dyn ModelClient>,
    pub(crate) embedding_client: Option<&'a LiveEmbeddingClient>,
    pub(crate) embedding_distance_threshold: Option<f32>,
    pub(crate) recency_weight: f32,
    pub(crate) connection: Option<&'a rusqlite::Connection>,
    pub(crate) query_embedding: Option<&'a [f32]>,
}

pub(crate) struct ReflectiveRecallPromptInputs<'a> {
    pub(crate) memories: &'a [&'a MemoryRecord],
    pub(crate) due_items: &'a [&'a AgendaItemRecord],
    pub(crate) agenda_items: &'a [&'a AgendaItemRecord],
    pub(crate) prompt: &'a str,
    pub(crate) recent_context: &'a [String],
    pub(crate) reflection_max_words: usize,
}

// ── transcript helpers ────────────────────────────────────────────────────────

pub(crate) fn transcript_contains_context_memory(
    transcript: &[ConversationMessage],
    memory_name: &str,
) -> bool {
    let tool_call_id = context_memory_tool_call_id(memory_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub(crate) fn transcript_contains_context_due_item(
    transcript: &[ConversationMessage],
    due_item_name: &str,
) -> bool {
    let tool_call_id = context_due_item_tool_call_id(due_item_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub(crate) fn transcript_contains_context_task(transcript: &[ConversationMessage], task_name: &str) -> bool {
    let tool_call_id = context_task_tool_call_id(task_name);
    transcript
        .iter()
        .any(|message| message_matches_tool_call_id(message, &tool_call_id))
}

pub(crate) fn transcript_contains_recalled_agenda_item(
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

pub(crate) fn message_matches_tool_call_id(message: &ConversationMessage, tool_call_id: &str) -> bool {
    message.tool_call_id.as_deref() == Some(tool_call_id)
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| tool_calls.iter().any(|call| call.id == tool_call_id))
}

// ── main recall entry points ──────────────────────────────────────────────────

pub(crate) fn recall_memory_context_messages_with_decision(
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

#[cfg(test)]
pub(crate) fn recall_memory_context_messages(
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
        },
        context,
    )
}

pub(crate) fn recall_due_item_context_messages(
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

pub(crate) fn memory_recall_status_updates_with_decision(
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
pub(crate) fn memory_recall_status_updates(
    memory_recall_classifier_enabled: bool,
    prompt: &str,
    fetched_memories: bool,
) -> Vec<StreamEvent> {
    let used_llm_classifier =
        memory_recall_classifier_enabled && !should_skip_memory_recall(prompt);
    memory_recall_status_updates_with_decision(used_llm_classifier, fetched_memories)
}

pub(crate) fn prompt_prelude_status_updates_with_decision(
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
pub(crate) fn prompt_prelude_status_updates(
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

pub(crate) fn should_skip_memory_recall(prompt: &str) -> bool {
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

pub(crate) fn parse_memory_recall_decision(response: &str) -> Option<(bool, String)> {
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

pub(crate) fn parse_relevance_filter_response(response: &str) -> Option<Vec<bool>> {
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

pub(crate) fn semantic_recall_enabled(
    relevance_model: Option<&dyn ModelClient>,
    embedding_client: Option<&LiveEmbeddingClient>,
) -> bool {
    relevance_model.is_some() || embedding_client.is_some()
}

pub(crate) fn semantic_recall_source_fetch_limit(
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
            let distance = l2_distance(&query_embedding, &embedding)?;
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

fn memory_embedding_text(memory: &MemoryRecord) -> String {
    format!("# {}\n{}", memory.name, memory.body.trim())
}

pub(crate) fn select_relevant_recall_memories<'a>(
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

pub(crate) fn select_relevant_recall_due_items<'a>(
    query: &str,
    due_items: &'a [AgendaItemRecord],
    selection_clients: RecallSelectionClients<'_>,
) -> Vec<&'a AgendaItemRecord> {
    let candidate_limit = semantic_recall_candidate_limit(
        selection_clients.limit,
        selection_clients.relevance_model,
        selection_clients.embedding_client,
    );
    let overlap_candidates = select_due_items_by_overlap(query, due_items, candidate_limit, None);
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

pub(crate) fn select_relevant_contextual_due_items<'a>(
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

pub(crate) fn select_relevant_recall_agenda_items<'a>(
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

pub(crate) fn classify_memory_recall_with_model(
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

pub(crate) fn determine_memory_recall_decision(
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

pub(crate) fn build_recall_query(prompt: &str, transcript: &[ConversationMessage], window: usize) -> String {
    let mut parts = recent_recall_context(transcript, window);
    parts.push(prompt.trim().to_string());
    parts.retain(|part| !part.is_empty());
    parts.join("\n")
}

pub(crate) fn recent_recall_context(transcript: &[ConversationMessage], window: usize) -> Vec<String> {
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

pub(crate) fn parse_reflective_recall_model_response(response: &str) -> Option<(bool, Option<String>)> {
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

#[cfg(test)]
pub(crate) fn recalled_memory_names(transcript: &[ConversationMessage]) -> HashSet<String> {
    recalled_item_refs_by_type(transcript, "Memory")
        .into_iter()
        .map(|item| item.name)
        .collect()
}

pub(crate) fn recalled_item_refs_by_type(
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

pub(crate) fn select_due_items_by_overlap<'a>(
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

#[cfg(test)]
pub(crate) fn select_recalled_due_items<'a>(
    prompt: &str,
    due_items: &'a [AgendaItemRecord],
    now_iso: &str,
    limit: usize,
) -> Vec<&'a AgendaItemRecord> {
    select_due_items_by_overlap(prompt, due_items, limit, Some(now_iso))
}

pub(crate) fn parse_recalled_item_refs(content: &str, desired_memory_type: &str) -> Vec<RecalledItemRef> {
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

pub(crate) fn recalled_item_matches(
    recalled_items: &[RecalledItemRef],
    item_id: i64,
    item_name: &str,
) -> bool {
    let normalized_name = item_name.to_ascii_lowercase();
    recalled_items.iter().any(|recalled| {
        recalled.id == Some(item_id) || (recalled.id.is_none() && recalled.name == normalized_name)
    })
}

pub(crate) fn select_recalled_memories<'a>(
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

pub(crate) fn significant_tokens(text: &str) -> HashSet<String> {
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
