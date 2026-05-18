use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use chrono::{Local, Utc};
use elroy_core::{ConversationRequest, ModelClient};
use elroy_llm::StreamEvent;
use elroy_db::{
    BootstrapPlan, MemoryRecord, get_or_create_memory_operation_tracker,
    load_context_messages, load_memory_embeddings_for_paths, open_sqlite_connection,
    run_migrations, save_memory_operation_tracker, upsert_memory_embedding,
};
use elroy_llm::{
    ConversationMessage, EmbeddingProviderConfig, LiveEmbeddingClient, MessageRole, ProviderConfig,
};
use elroy_memory::create_memory_file_with_frontmatter;
use serde_json::Value;

use crate::{
    AppError,
    best_effort_embedding_client, best_effort_provider_model,
    create_consolidated_memories_from_records,
    formulate_memory_from_transcript,
    list_all_active_memories_in_scope,
    CONTEXT_MESSAGE_SOURCE_TYPE,
    LOCAL_USER_TOKEN,
    MEMORY_CONSOLIDATION_CLUSTER_LIMIT,
    MEMORY_SOURCE_TYPE,
    MEMORY_WORD_COUNT_LIMIT,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MemoryConsolidationSettings {
    pub(crate) memory_cluster_similarity_threshold: f64,
    pub(crate) max_memory_cluster_size: usize,
    pub(crate) min_memory_cluster_size: usize,
    pub(crate) fast_provider_config: Option<ProviderConfig>,
    pub(crate) embedding_provider_config: Option<EmbeddingProviderConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConsolidatedMemoryOutput {
    pub(crate) name: String,
    pub(crate) text: String,
}

pub(crate) fn run_auto_memory_if_needed(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
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
        memory_consolidation_settings,
    )?;
    Ok(())
}

pub(crate) fn record_memory_creation_and_maybe_consolidate(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memories_between_consolidation: usize,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
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
        consolidate_memories(connection, bootstrap_plan, memory_consolidation_settings)?;
        tracker.memories_since_consolidation = 0;
        tracker.updated_at_unix = Utc::now().timestamp();
        save_memory_operation_tracker(connection, &tracker)?;
        Ok(())
    })()
}

fn consolidate_memories(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    memory_consolidation_settings: Option<&MemoryConsolidationSettings>,
) -> Result<(), AppError> {
    consolidate_exact_duplicate_memories(connection, bootstrap_plan)?;
    if let Some(settings) = memory_consolidation_settings {
        consolidate_semantic_memory_clusters(connection, bootstrap_plan, settings)?;
    }
    Ok(())
}

pub(crate) fn consolidate_exact_duplicate_memories(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
) -> Result<(), AppError> {
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
        let canonical = group.first().ok_or_else(|| {
            AppError::Runtime("duplicate memory group was unexpectedly empty".to_string())
        })?;
        create_consolidated_memories_from_records(
            bootstrap_plan,
            &[ConsolidatedMemoryOutput {
                name: canonical.name.clone(),
                text: canonical.body.clone(),
            }],
            &group,
        )
        .map_err(AppError::Io)?;
        *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
        run_migrations(connection)?;
    }

    Ok(())
}

pub(crate) fn consolidate_semantic_memory_clusters(
    connection: &mut rusqlite::Connection,
    bootstrap_plan: &BootstrapPlan,
    settings: &MemoryConsolidationSettings,
) -> Result<(), AppError> {
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
        create_consolidated_memories_from_records(bootstrap_plan, &outputs, &cluster_memories)
            .map_err(AppError::Io)?;
        *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
        run_migrations(connection)?;
    }

    Ok(())
}

fn load_or_create_cached_memory_embeddings(
    connection: &rusqlite::Connection,
    memories: &[MemoryRecord],
    embedding_client: &LiveEmbeddingClient,
) -> Result<Vec<(MemoryRecord, Vec<f32>)>, AppError> {
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

pub(crate) fn memory_embedding_text(memory: &MemoryRecord) -> String {
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

pub(crate) fn consolidate_memory_cluster_outputs(
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

pub(crate) fn parse_consolidated_memory_response(response: &str) -> Option<Vec<ConsolidatedMemoryOutput>> {
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

pub(crate) fn create_memory_file_from_context_messages(
    memory_dir: &Path,
    name: &str,
    text: &str,
    context_messages: &[ConversationMessage],
) -> std::io::Result<PathBuf> {
    let frontmatter = context_message_source_frontmatter(context_messages);
    create_memory_file_with_frontmatter(memory_dir, name, text, frontmatter.as_deref())
}

pub(crate) fn context_message_source_frontmatter(
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

pub(crate) fn parse_context_message_source_ids(frontmatter: Option<&str>) -> Option<Vec<i64>> {
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

pub(crate) fn memory_source_frontmatter(memory_sources: &[(&str, &Path)]) -> Option<String> {
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

pub(crate) fn parse_memory_sources(frontmatter: Option<&str>) -> Option<Vec<(String, String)>> {
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

pub(crate) fn list_memory_sources(frontmatter: Option<&str>) -> Vec<(String, String)> {
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

pub(crate) fn format_context_message_source_content(messages: &[ConversationMessage]) -> String {
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

pub(crate) fn format_memory_file_source_content(source_name: &str, source_body: &str) -> String {
    format!("#{source_name}\n{source_body}")
}
