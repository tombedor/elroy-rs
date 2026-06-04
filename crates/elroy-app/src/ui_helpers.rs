use std::path::Path;

use chrono::{Local, NaiveDateTime, TimeZone, Utc};
use elroy_db::{
    AgendaItemRecord, LOCAL_USER_TOKEN, SYNTHETIC_FIRST_USER_MESSAGE, UserPreferenceRecord,
    find_active_agenda_item_by_name, list_active_plain_agenda_items, load_context_messages,
    load_user_preferences,
};
use elroy_feature_requests::{
    FeatureRequestRecord, get_feature_request, list_feature_requests,
    list_self_reflection_feature_requests,
};
use elroy_llm::{ConversationMessage, MessageRole};
use elroy_tui::{
    SidebarSection, TuiCommandParameter, TuiSnapshot, format_assistant_transcript_lines,
};
use elroy_user::effective_user_preferred_name;
use serde_json::{Map, Value};

use crate::AppError;
use crate::list_active_memories_in_scope;
use crate::{list_active_tasks, list_recent_codex_sessions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandResultTarget {
    History,
    Toast,
}

pub(crate) fn command_result_target(command_name: &str) -> CommandResultTarget {
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

pub(crate) fn is_short_single_line_result(content: &str) -> bool {
    !content.is_empty() && !content.contains('\n') && content.chars().count() <= 180
}

pub(crate) fn ordered_command_parameters(
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

pub(crate) fn display_command_name(name: &str) -> &str {
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

pub(crate) fn load_snapshot_from_connection(
    connection: &mut rusqlite::Connection,
    home_dir: &Path,
    memory_dir: &Path,
    show_internal_thought: bool,
) -> Result<TuiSnapshot, AppError> {
    let conversation_lines = load_context_messages(connection, LOCAL_USER_TOKEN)?
        .into_iter()
        .filter(|message| !is_bootstrap_or_synthetic_startup_message(message))
        .flat_map(|message| format_persisted_conversation_message(message, show_internal_thought))
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
        show_internal_thought,
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

fn is_bootstrap_or_synthetic_startup_message(message: &ConversationMessage) -> bool {
    is_bootstrap_session_context_message(message)
        || (message.role == MessageRole::User
            && message.content.as_deref() == Some(SYNTHETIC_FIRST_USER_MESSAGE))
}

fn format_persisted_conversation_message(
    message: ConversationMessage,
    show_internal_thought: bool,
) -> Vec<String> {
    let content = message.content.unwrap_or_default();
    if content.is_empty() {
        return Vec::new();
    }

    match message.role {
        MessageRole::System => Vec::new(),
        MessageRole::User => vec![format!("user: {content}")],
        MessageRole::Assistant => {
            format_assistant_transcript_lines(&content, show_internal_thought)
        }
        MessageRole::Tool => vec![format!("tool result: {content}")],
    }
}

pub(crate) fn is_bootstrap_session_context_message(message: &ConversationMessage) -> bool {
    message.role == MessageRole::Assistant
        && message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| calls.iter().any(|call| call.name == "get_session_context"))
        || (message.role == MessageRole::Tool
            && message
                .tool_call_id
                .as_deref()
                .is_some_and(|id| id.starts_with("bootstrap-session-context:")))
}

pub(crate) fn build_session_context_message(
    connection: &mut rusqlite::Connection,
) -> Result<String, AppError> {
    let preferences = load_user_preferences(connection, LOCAL_USER_TOKEN)?;
    Ok(session_context_message_from_messages(
        preferences.as_ref(),
        &load_context_messages(connection, LOCAL_USER_TOKEN)?,
    ))
}

fn session_context_message_from_messages(
    preferences: Option<&UserPreferenceRecord>,
    messages: &[ConversationMessage],
) -> String {
    let preferred_name = effective_user_preferred_name(preferences);
    let now = Local::now();
    let current_datetime = now.format("%A, %B %d, %Y %I:%M %p %Z").to_string();
    let today = now.date_naive();
    let earliest_today_user = messages
        .iter()
        .filter(|message| {
            message.role == MessageRole::User
                && message.content.as_deref() != Some(SYNTHETIC_FIRST_USER_MESSAGE)
        })
        .filter_map(|message| {
            Local
                .timestamp_opt(message.created_at_unix, 0)
                .single()
                .map(|timestamp| (timestamp.date_naive(), timestamp))
        })
        .filter(|(date, _)| *date == today)
        .map(|(_, timestamp)| timestamp)
        .min();

    if let Some(first_chat) = earliest_today_user {
        return format!(
            "Current date/time: {current_datetime}. {preferred_name} has logged in. I first started chatting with {preferred_name} today at {}.",
            first_chat.format("%I:%M %p")
        );
    }

    format!(
        "Current date/time: {current_datetime}. {preferred_name} has logged in. I haven't chatted with {preferred_name} yet today. I should offer a brief greeting (less than 50 words)."
    )
}

pub(crate) fn format_agenda_sidebar_title(item: &AgendaItemRecord, now: NaiveDateTime) -> String {
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

pub(crate) fn format_codex_session_title(session: &elroy_codex::CodexSessionRecord) -> String {
    let repo_name = Path::new(&session.repo_path)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(&session.repo_path);
    format!("{repo_name} ({}) {}", session.status, session.thread_id)
}

pub(crate) fn parse_sidebar_trigger_datetime(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M").ok())
}

pub(crate) fn resolve_agenda_sidebar_item(
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

pub(crate) fn format_feature_request_sidebar_title(record: FeatureRequestRecord) -> String {
    format!("{} ({})", record.title, record.status)
}

pub(crate) fn feature_request_detail_content(record: &FeatureRequestRecord) -> String {
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

fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

pub(crate) fn resolve_feature_request_sidebar_item(
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

pub(crate) fn should_offer_greeting(
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
