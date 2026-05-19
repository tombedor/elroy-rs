use chrono::{Local, TimeZone, Utc};
use elroy_config::{AppConfig, fast_provider_config_from_app_config};
use elroy_core::{
    ConversationRequest, LiveProviderModel, ModelClient, excerpt, validated_transcript,
};
use elroy_db::{
    LOCAL_USER_TOKEN, SYNTHETIC_FIRST_USER_MESSAGE, UserPreferenceRecord, load_context_messages,
    load_user_preferences, replace_context_messages,
};
use elroy_llm::{ConversationMessage, LiveModelClient, MessageRole, StreamEvent};
use elroy_user::{effective_persona, effective_user_preferred_name};

const CONTEXT_REFRESH_SUMMARY_WORD_LIMIT: usize = 300;

pub fn drop_old_context_messages(
    connection: &mut rusqlite::Connection,
    max_context_age_minutes: f64,
) -> anyhow::Result<()> {
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

pub fn current_system_message(
    default_assistant_name: &str,
    preferences: Option<&UserPreferenceRecord>,
) -> ConversationMessage {
    ConversationMessage::new(
        MessageRole::System,
        effective_persona(preferences, default_assistant_name),
    )
}

pub fn repair_system_message_placement(
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

pub fn repair_first_user_precedes_first_assistant(
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

pub fn load_validated_runtime_transcript(
    connection: &mut rusqlite::Connection,
    default_assistant_name: &str,
    ensure_alternating_roles: bool,
) -> anyhow::Result<Vec<ConversationMessage>> {
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
        return load_context_messages(connection, LOCAL_USER_TOKEN).map_err(anyhow::Error::from);
    }

    Ok(validated)
}

pub fn refreshed_context_messages_with_system(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> anyhow::Result<Vec<ConversationMessage>> {
    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    let system_message = current_system_message(&config.assistant_name, preferences.as_ref());
    let mut refreshed = load_context_messages(connection, LOCAL_USER_TOKEN)?
        .into_iter()
        .filter(|message| message.role != MessageRole::System)
        .collect::<Vec<_>>();
    refreshed.insert(0, system_message);
    Ok(refreshed)
}

pub fn refresh_persisted_system_instructions(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let refreshed = refreshed_context_messages_with_system(connection, config)?;
    replace_context_messages(connection, LOCAL_USER_TOKEN, &refreshed)?;
    Ok(())
}

pub fn reset_persisted_context(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    let system_message = current_system_message(&config.assistant_name, preferences.as_ref());
    replace_context_messages(connection, LOCAL_USER_TOKEN, &[system_message])?;
    Ok(())
}

pub fn context_refresh_summary_system_prompt(assistant_name: &str) -> String {
    format!(
        "Your job is to summarize a history of previous messages in a conversation between an AI persona and a human.\nThe conversation you are given is from a fixed context window and may not be complete.\nMessages sent by the AI are marked with the 'assistant' role.\nSummarize what happened in the conversation from the perspective of {} (use the first person).\nNote not only the content of the messages but also the context and relationship between the entities mentioned.\nAlso take note of the overall tone of the conversation.\nOnly output the summary, and keep it concise.",
        assistant_name
    )
}

pub fn format_context_messages_for_summary(
    messages: &[ConversationMessage],
    user_name: &str,
    assistant_name: &str,
) -> String {
    let conversation_range = messages
        .iter()
        .filter(|message| message.role == MessageRole::User)
        .map(|message| message.created_at_unix)
        .min()
        .zip(
            messages
                .iter()
                .filter(|message| message.role == MessageRole::User)
                .map(|message| message.created_at_unix)
                .max(),
        )
        .map(|(min, max)| {
            format!(
                "Messages from {} to {}",
                format_context_summary_timestamp(min),
                format_context_summary_timestamp(max)
            )
        })
        .unwrap_or_else(|| "No messages in context".to_string());

    let lines = messages
        .iter()
        .filter_map(|message| match message.role {
            MessageRole::System => None,
            MessageRole::User => message
                .content
                .as_deref()
                .map(str::trim)
                .filter(|content| !content.is_empty())
                .map(|content| {
                    format!(
                        "{user_name} ({}): {}",
                        format_context_summary_timestamp(message.created_at_unix),
                        excerpt(content, 400)
                    )
                }),
            MessageRole::Assistant => {
                let mut lines = Vec::new();
                if let Some(content) = message.content.as_deref().map(str::trim)
                    && !content.is_empty()
                {
                    lines.push(format!(
                        "{assistant_name} ({}): {}",
                        format_context_summary_timestamp(message.created_at_unix),
                        excerpt(content, 400)
                    ));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    lines.extend(tool_calls.iter().map(|call| {
                        format!(
                            "{assistant_name} TOOL CALL REQUEST ({}): function name: {}, arguments: {}",
                            format_context_summary_timestamp(message.created_at_unix),
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
                .map(|content| {
                    format!(
                        "TOOL CALL RESULT ({}): {}",
                        format_context_summary_timestamp(message.created_at_unix),
                        excerpt(content, 400)
                    )
                }),
        })
        .collect::<Vec<_>>();

    ["Conversation Summary".to_string()]
        .into_iter()
        .chain(lines)
        .chain(std::iter::once(conversation_range))
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub fn summarize_context_messages_with_model(
    model: &dyn ModelClient,
    assistant_name: &str,
    user_name: &str,
    messages: &[ConversationMessage],
) -> anyhow::Result<String> {
    let base_prompt = format_context_messages_for_summary(messages, user_name, assistant_name);
    if base_prompt.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "cannot summarize empty context-refresh transcript"
        ));
    }

    let prompt = format!(
        "{}\n\nYour word limit is {}. DO NOT EXCEED IT.",
        base_prompt, CONTEXT_REFRESH_SUMMARY_WORD_LIMIT
    );

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
        return Err(anyhow::anyhow!("summary model returned no assistant text"));
    }

    Ok(format!("Recent conversation summary: {summary}"))
}

pub fn build_context_summary_message(
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
    let Ok(provider_config) = fast_provider_config_from_app_config(config) else {
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

pub fn format_context_summary_message(messages: &[ConversationMessage]) -> String {
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

pub fn format_context_summary_timestamp(unix_seconds: i64) -> String {
    Local
        .timestamp_opt(unix_seconds, 0)
        .single()
        .unwrap_or_else(Local::now)
        .format("%A, %B %d, %Y %I:%M %p %Z")
        .to_string()
}

pub fn approximate_message_token_count(message: &ConversationMessage) -> usize {
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

pub fn count_context_tokens(context_messages: &[ConversationMessage]) -> usize {
    context_messages
        .iter()
        .map(approximate_message_token_count)
        .sum()
}

pub fn is_context_refresh_needed(
    context_messages: &[ConversationMessage],
    max_tokens: usize,
) -> bool {
    if !context_messages
        .iter()
        .any(|message| message.role == MessageRole::User)
    {
        return false;
    }

    count_context_tokens(context_messages) > max_tokens
}

pub fn compress_context_messages(
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

pub fn strip_transient_context_messages(
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

pub fn strip_input_message_for_persistence(
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
