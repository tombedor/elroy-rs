// Due-item surfacing, reminder selection, and synthetic context message generation.

use elroy_db::AgendaItemRecord;
use elroy_llm::ConversationMessage;
use elroy_recall::{parse_sidebar_trigger_datetime, synthetic_tool_context_messages};

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
