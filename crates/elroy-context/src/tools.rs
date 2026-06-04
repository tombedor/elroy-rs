use elroy_config::AppConfig;
use elroy_db::{LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection, run_migrations};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};
use serde_json::{Value, json};

use crate::{refresh_persisted_system_instructions, reset_persisted_context};

pub fn context_tools(config: &AppConfig) -> Vec<ExecutableTool> {
    let config_for_show = config.clone();
    let show_context_messages = ExecutableTool::new(
        ToolSpec::new(
            "show_context_messages",
            "Show persisted transcript context messages for the local user.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = arguments.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
            let mut connection = match open_sqlite_connection(&config_for_show.database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            match load_context_messages(&mut connection, LOCAL_USER_TOKEN) {
                Ok(messages) => {
                    let start = messages.len().saturating_sub(limit);
                    match serde_json::to_string_pretty(&messages[start..]) {
                        Ok(content) => ToolExecutionResult::success(content),
                        Err(error) => ToolExecutionResult::error(format!(
                            "failed to serialize context messages: {error}"
                        )),
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to load context messages: {error}"))
                }
            }
        },
    );

    let config_for_reset = config.clone();
    let reset_messages = ExecutableTool::new(
        ToolSpec::new(
            "reset_messages",
            "Reset persisted transcript context to the current system message only.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut connection = match open_sqlite_connection(&config_for_reset.database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            match reset_persisted_context(&mut connection, &config_for_reset) {
                Ok(()) => ToolExecutionResult::success("Context reset complete".to_string()),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to reset context: {error}"))
                }
            }
        },
    );

    let config_for_refresh = config.clone();
    let refresh_system_instructions = ExecutableTool::new(
        ToolSpec::new(
            "refresh_system_instructions",
            "Refresh the effective system instructions for the local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            let mut connection = match open_sqlite_connection(&config_for_refresh.database_path) {
                Ok(connection) => connection,
                Err(error) => {
                    return ToolExecutionResult::error(format!("failed to open database: {error}"));
                }
            };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            match refresh_persisted_system_instructions(&mut connection, &config_for_refresh) {
                Ok(()) => {
                    ToolExecutionResult::success("System instruction refresh complete".to_string())
                }
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to refresh system instructions: {error}"
                )),
            }
        },
    );

    vec![
        show_context_messages,
        reset_messages,
        refresh_system_instructions,
    ]
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::context_tools;
    use elroy_config::AppConfig;
    use elroy_db::{
        LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection, replace_context_messages,
        run_migrations,
    };
    use elroy_llm::{ConversationMessage, MessageRole};
    use elroy_tools::ExecutableToolRegistry;

    fn test_config(name: &str) -> (AppConfig, std::path::PathBuf) {
        let unique = format!(
            "elroy-rs-context-tools-{name}-{}",
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
        (config, home)
    }

    #[test]
    fn show_context_messages_respects_limit() {
        let (config, home) = test_config("show");
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::System, "system"),
                ConversationMessage::new(MessageRole::User, "one"),
                ConversationMessage::new(MessageRole::Assistant, "two"),
            ],
        )
        .expect("messages should persist");

        let registry = ExecutableToolRegistry::new(context_tools(&config));
        let shown = registry.invoke("show_context_messages", "{\"limit\":2}");
        assert!(!shown.is_error);
        let messages: Vec<ConversationMessage> =
            serde_json::from_str(&shown.content).expect("context payload should deserialize");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("one"));
        assert_eq!(messages[1].content.as_deref(), Some("two"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn reset_and_refresh_context_tools_update_persisted_context() {
        let (config, home) = test_config("reset");
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[
                ConversationMessage::new(MessageRole::User, "hello"),
                ConversationMessage::new(MessageRole::Assistant, "hi"),
            ],
        )
        .expect("messages should persist");

        let registry = ExecutableToolRegistry::new(context_tools(&config));
        let reset = registry.invoke("reset_messages", "{}");
        assert!(!reset.is_error);
        assert_eq!(reset.content, "Context reset complete");

        let stored = load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].role, MessageRole::System);
        assert!(
            stored[0]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("I am Elroy"))
        );

        let refresh = registry.invoke("refresh_system_instructions", "{}");
        assert!(!refresh.is_error);
        assert_eq!(refresh.content, "System instruction refresh complete");

        let refreshed = load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(refreshed.len(), 1);
        assert_eq!(refreshed[0].role, MessageRole::System);

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
