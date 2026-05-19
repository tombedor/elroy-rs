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
