use std::path::Path;
use std::sync::Arc;

use elroy_config::AppConfig;
use elroy_db::{
    LOCAL_USER_TOKEN, UserPreferenceRecord, load_user_preferences, open_sqlite_connection,
    run_migrations, save_user_preferences,
};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};
use serde_json::{Value, json};

use super::{
    DEFAULT_USER_PREFERRED_NAME, UNKNOWN_FULL_NAME, effective_user_full_name,
    effective_user_preferred_name,
};

/// Callback invoked after user preferences are saved.
/// Receives a mutable connection and the app config; returns an error string on failure.
pub type PostSaveCallback =
    Arc<dyn Fn(&mut rusqlite::Connection, &AppConfig) -> anyhow::Result<()> + Send + Sync>;

pub fn user_tools(config: AppConfig, post_save: PostSaveCallback) -> Vec<ExecutableTool> {
    let config_for_assistant_name = config.clone();
    let post_save_for_assistant_name = post_save.clone();
    let set_assistant_name = ExecutableTool::new(
        ToolSpec::new(
            "set_assistant_name",
            "Set the assistant name for this local user.",
            JsonSchema::object(
                [("assistant_name", json!({"type": "string"}))],
                ["assistant_name"],
            ),
        ),
        move |arguments| {
            let Some(assistant_name) = arguments.get("assistant_name").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error(
                    "set_assistant_name requires string assistant_name",
                );
            };
            mutate_user_preferences_in_config(
                &config_for_assistant_name,
                &post_save_for_assistant_name,
                |record| {
                    record.assistant_name = Some(assistant_name.to_string());
                    Ok(format!("Assistant name updated to {assistant_name}."))
                },
            )
        },
    );

    let config_for_persona = config.clone();
    let post_save_for_persona = post_save.clone();
    let set_persona = ExecutableTool::new(
        ToolSpec::new(
            "set_persona",
            "Set the system persona template for this local user.",
            JsonSchema::object(
                [("system_persona", json!({"type": "string"}))],
                ["system_persona"],
            ),
        ),
        move |arguments| {
            let Some(system_persona) = arguments.get("system_persona").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error("set_persona requires string system_persona");
            };
            let system_persona = system_persona.trim();
            if system_persona.is_empty() {
                return ToolExecutionResult::error("System persona cannot be blank.");
            }
            mutate_user_preferences_in_config(
                &config_for_persona,
                &post_save_for_persona,
                |record| {
                    if record.system_persona.as_deref() == Some(system_persona) {
                        return Ok("New system persona and old system persona are identical".into());
                    }
                    record.system_persona = Some(system_persona.to_string());
                    Ok("System persona updated.".into())
                },
            )
        },
    );

    let config_for_reset_system_persona = config.clone();
    let post_save_for_reset = post_save.clone();
    let reset_system_persona = ExecutableTool::new(
        ToolSpec::new(
            "reset_system_persona",
            "Clear the persisted system persona for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            mutate_user_preferences_in_config(
                &config_for_reset_system_persona,
                &post_save_for_reset,
                |record| {
                    record.system_persona = None;
                    Ok("System persona cleared, will now use default persona.".into())
                },
            )
        },
    );

    let config_for_preferred_name = config.clone();
    let post_save_for_preferred_name = post_save.clone();
    let set_user_preferred_name = ExecutableTool::new(
        ToolSpec::new(
            "set_user_preferred_name",
            "Set the preferred name for this local user.",
            JsonSchema::object(
                [
                    ("preferred_name", json!({"type": "string"})),
                    ("override_existing", json!({"type": "boolean"})),
                ],
                ["preferred_name"],
            ),
        ),
        move |arguments| {
            let Some(preferred_name) = arguments.get("preferred_name").and_then(Value::as_str)
            else {
                return ToolExecutionResult::error(
                    "set_user_preferred_name requires string preferred_name",
                );
            };
            let override_existing = arguments
                .get("override_existing")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            mutate_user_preferences_in_config(
                &config_for_preferred_name,
                &post_save_for_preferred_name,
                |record| {
                    let existing = effective_user_preferred_name(Some(record));
                    if existing != DEFAULT_USER_PREFERRED_NAME && !override_existing {
                        return Ok(format!(
                            "Preferred name already set to {}. If this should be changed, use override_existing=True.",
                            existing
                        ));
                    }
                    record.preferred_name = Some(preferred_name.to_string());
                    Ok(format!(
                        "Set user preferred name to {}. Was {}.",
                        preferred_name, existing
                    ))
                },
            )
        },
    );

    let database_path = config.database_path.clone();
    let get_user_preferred_name = ExecutableTool::new(
        ToolSpec::new(
            "get_user_preferred_name",
            "Return the preferred name for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            with_user_preferences_at_path(&database_path, |record| {
                Ok(ToolExecutionResult::success(effective_user_preferred_name(
                    record.as_ref(),
                )))
            })
        },
    );

    let config_for_full_name = config.clone();
    let post_save_for_full_name = post_save.clone();
    let set_user_full_name = ExecutableTool::new(
        ToolSpec::new(
            "set_user_full_name",
            "Set the full name for this local user.",
            JsonSchema::object(
                [
                    ("full_name", json!({"type": "string"})),
                    ("override_existing", json!({"type": "boolean"})),
                ],
                ["full_name"],
            ),
        ),
        move |arguments| {
            let Some(full_name) = arguments.get("full_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("set_user_full_name requires string full_name");
            };
            let override_existing = arguments
                .get("override_existing")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            mutate_user_preferences_in_config(
                &config_for_full_name,
                &post_save_for_full_name,
                |record| {
                    let existing = effective_user_full_name(Some(record));
                    if existing != UNKNOWN_FULL_NAME && !override_existing {
                        return Ok(format!(
                            "Full name already set to {}. If this should be changed, set override_existing=True.",
                            existing
                        ));
                    }
                    record.full_name = Some(full_name.to_string());
                    Ok(format!(
                        "Full name set to {}. Previous value was {}.",
                        full_name, existing
                    ))
                },
            )
        },
    );

    let database_path = config.database_path.clone();
    let get_user_full_name = ExecutableTool::new(
        ToolSpec::new(
            "get_user_full_name",
            "Return the full name for this local user.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| {
            with_user_preferences_at_path(&database_path, |record| {
                Ok(ToolExecutionResult::success(effective_user_full_name(
                    record.as_ref(),
                )))
            })
        },
    );

    vec![
        set_assistant_name,
        set_persona,
        reset_system_persona,
        set_user_preferred_name,
        get_user_preferred_name,
        set_user_full_name,
        get_user_full_name,
    ]
}

fn with_user_preferences_at_path(
    database_path: &Path,
    operation: impl FnOnce(Option<UserPreferenceRecord>) -> rusqlite::Result<ToolExecutionResult>,
) -> ToolExecutionResult {
    let mut connection = match open_sqlite_connection(database_path) {
        Ok(connection) => connection,
        Err(error) => {
            return ToolExecutionResult::error(format!("failed to open database: {error}"));
        }
    };
    if let Err(error) = run_migrations(&mut connection) {
        return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
    }
    match load_user_preferences(&connection, LOCAL_USER_TOKEN).and_then(operation) {
        Ok(result) => result,
        Err(error) => ToolExecutionResult::error(format!("database query failed: {error}")),
    }
}

fn mutate_user_preferences_in_config(
    config: &AppConfig,
    post_save: &PostSaveCallback,
    operation: impl FnOnce(&mut UserPreferenceRecord) -> rusqlite::Result<String>,
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
    let mut record = load_user_preferences(&connection, LOCAL_USER_TOKEN)
        .ok()
        .flatten()
        .unwrap_or(UserPreferenceRecord {
            user_token: LOCAL_USER_TOKEN.to_string(),
            assistant_name: None,
            preferred_name: None,
            full_name: None,
            system_persona: None,
            created_at_unix: 0,
            updated_at_unix: 0,
        });

    match operation(&mut record).and_then(|message| {
        save_user_preferences(&mut connection, &record)?;
        Ok(message)
    }) {
        Ok(message) => {
            if let Err(error) = post_save(&mut connection, config) {
                return ToolExecutionResult::error(format!(
                    "user preference update failed: {error}"
                ));
            }
            ToolExecutionResult::success(message)
        }
        Err(error) => ToolExecutionResult::error(format!("user preference update failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use super::user_tools;
    use crate::tools::PostSaveCallback;
    use elroy_config::AppConfig;
    use elroy_context::refresh_persisted_system_instructions;
    use elroy_db::{
        LOCAL_USER_TOKEN, load_context_messages, load_user_preferences, open_sqlite_connection,
    };
    use elroy_llm::MessageRole;
    use elroy_tools::ExecutableToolRegistry;

    fn refresh_callback() -> PostSaveCallback {
        Arc::new(refresh_persisted_system_instructions)
    }

    #[test]
    fn user_tools_can_manage_preferences_and_refresh_system_context() {
        let unique = format!(
            "elroy-rs-user-tools-{}",
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

        let registry = ExecutableToolRegistry::new(user_tools(config.clone(), refresh_callback()));
        let default_name = registry.invoke("get_user_preferred_name", "{}");
        assert!(!default_name.is_error);
        assert_eq!(default_name.content, "User");

        let preferred =
            registry.invoke("set_user_preferred_name", "{\"preferred_name\":\"Jimmy\"}");
        assert!(!preferred.is_error);
        assert!(preferred.content.contains("Jimmy"));

        let duplicate =
            registry.invoke("set_user_preferred_name", "{\"preferred_name\":\"James\"}");
        assert!(!duplicate.is_error);
        assert_eq!(
            duplicate.content,
            "Preferred name already set to Jimmy. If this should be changed, use override_existing=True."
        );

        let preferred_override = registry.invoke(
            "set_user_preferred_name",
            "{\"preferred_name\":\"James\",\"override_existing\":true}",
        );
        assert!(!preferred_override.is_error);
        assert_eq!(
            preferred_override.content,
            "Set user preferred name to James. Was Jimmy."
        );

        let assistant = registry.invoke("set_assistant_name", "{\"assistant_name\":\"Nova\"}");
        assert!(!assistant.is_error);
        assert!(assistant.content.contains("Nova"));

        let full_name = registry.invoke("set_user_full_name", "{\"full_name\":\"James Smith\"}");
        assert!(!full_name.is_error);
        assert_eq!(
            full_name.content,
            "Full name set to James Smith. Previous value was Unknown name."
        );

        let full_name_duplicate =
            registry.invoke("set_user_full_name", "{\"full_name\":\"James T. Smith\"}");
        assert!(!full_name_duplicate.is_error);
        assert_eq!(
            full_name_duplicate.content,
            "Full name already set to James Smith. If this should be changed, set override_existing=True."
        );

        let full_name_override = registry.invoke(
            "set_user_full_name",
            "{\"full_name\":\"James T. Smith\",\"override_existing\":true}",
        );
        assert!(!full_name_override.is_error);
        assert_eq!(
            full_name_override.content,
            "Full name set to James T. Smith. Previous value was James Smith."
        );

        let get_full_name = registry.invoke("get_user_full_name", "{}");
        assert!(!get_full_name.is_error);
        assert_eq!(get_full_name.content, "James T. Smith");

        let persona = registry.invoke(
            "set_persona",
            "{\"system_persona\":\"You are $ASSISTANT_ALIAS helping $USER_ALIAS.\"}",
        );
        assert!(!persona.is_error);
        assert_eq!(persona.content, "System persona updated.");

        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        let context = load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(context[0].role, MessageRole::System);
        assert_eq!(
            context[0].content.as_deref(),
            Some("You are Nova helping James.")
        );

        let persisted = load_user_preferences(&connection, LOCAL_USER_TOKEN)
            .expect("preferences should load")
            .expect("preferences should exist");
        assert_eq!(
            persisted.system_persona.as_deref(),
            Some("You are $ASSISTANT_ALIAS helping $USER_ALIAS.")
        );

        let reset = registry.invoke("reset_system_persona", "{}");
        assert!(!reset.is_error);
        assert_eq!(
            reset.content,
            "System persona cleared, will now use default persona."
        );

        let cleared = load_user_preferences(&connection, LOCAL_USER_TOKEN)
            .expect("preferences should load")
            .expect("preferences should exist");
        assert_eq!(cleared.system_persona, None);
        let context = load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("load ok");
        assert_eq!(context[0].role, MessageRole::System);
        assert!(
            context[0]
                .content
                .as_deref()
                .is_some_and(|content| content.contains("I am Nova"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
