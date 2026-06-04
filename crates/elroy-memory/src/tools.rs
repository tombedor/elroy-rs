use elroy_config::AppConfig;
use elroy_context::load_validated_runtime_transcript;
use elroy_core::memory_store::{archive_memory_file, update_memory_body};
use elroy_db::{
    BootstrapPlan, LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection,
    replace_context_messages, run_migrations,
};
use elroy_recall::{
    archive_memory_file_from_config, context_memory_tool_call_id, context_memory_tool_messages,
    create_consolidated_memory_from_config, create_memory_file_from_context_messages,
    examine_memories_from_config, find_active_memory_by_name_in_scope, format_memory_detail,
    format_memory_listing, get_source_content_for_memory_from_config,
    get_source_list_for_memory_from_config, list_active_memories_in_scope,
    memory_consolidation_settings_from_app_config, message_matches_tool_call_id,
    mutate_memory_file_from_config, record_memory_creation_and_maybe_consolidate,
    search_memories_from_config, update_outdated_or_incorrect_memory_from_config,
};
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};
use serde_json::{Value, json};
use std::path::PathBuf;

pub fn memory_tools(config: &AppConfig) -> Vec<ExecutableTool> {
    let config_for_memory_write = config.clone();
    let create_memory = ExecutableTool::new(
        ToolSpec::new(
            "create_memory",
            "Create a new file-backed memory from current context and rebuild derived state.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                ],
                ["name", "text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_memory requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("create_memory requires string text");
            };
            let created = (|| -> Result<PathBuf, std::io::Error> {
                let mut connection = open_sqlite_connection(&config_for_memory_write.database_path)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                run_migrations(&mut connection)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let context_messages = load_context_messages(&mut connection, LOCAL_USER_TOKEN)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let path = create_memory_file_from_context_messages(
                    &config_for_memory_write.memory_dir,
                    name,
                    text,
                    &context_messages,
                )?;
                elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config_for_memory_write))
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                record_memory_creation_and_maybe_consolidate(
                    &mut connection,
                    &BootstrapPlan::from_config(&config_for_memory_write),
                    config_for_memory_write.memories_between_consolidation,
                    Some(&memory_consolidation_settings_from_app_config(
                        &config_for_memory_write,
                    )),
                )
                .map_err(|error| std::io::Error::other(error.to_string()))?;
                Ok(path)
            })();
            match created {
                Ok(_path) => ToolExecutionResult::success(format!("New memory created: {name}")),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to create memory: {error}"))
                }
            }
        },
    );

    let config_for_memory_update = config.clone();
    let update_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_memory",
            "Update the content of an existing active memory and rebuild derived state.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                ],
                ["text"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("update_memory requires a string name");
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error("update_memory requires string text");
            };
            let text = text.to_string();
            mutate_memory_file_from_config(&config_for_memory_update, name, move |path| {
                update_memory_body(path, &text)
            })
        },
    );

    let config_for_outdated_memory_update = config.clone();
    let update_outdated_or_incorrect_memory = ExecutableTool::new(
        ToolSpec::new(
            "update_outdated_or_incorrect_memory",
            "Replace one active memory with an updated version while preserving the old content as a source.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("update_text", json!({"type": "string"})),
                ],
                ["memory_name", "update_text"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "update_outdated_or_incorrect_memory requires string memory_name",
                );
            };
            let Some(update_text) = arguments.get("update_text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "update_outdated_or_incorrect_memory requires string update_text",
                );
            };
            update_outdated_or_incorrect_memory_from_config(
                &config_for_outdated_memory_update,
                memory_name,
                update_text,
            )
        },
    );

    let config_for_memory_archive = config.clone();
    let archive_memory = ExecutableTool::new(
        ToolSpec::new(
            "archive_memory",
            "Archive an active memory to the archive directory and rebuild derived state.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("archive_memory requires a string name");
            };
            let archive_dir = config_for_memory_archive.memory_dir.join("archive");
            archive_memory_file_from_config(&config_for_memory_archive, name, move |path| {
                archive_memory_file(path, &archive_dir)
            })
        },
    );

    let config_for_consolidated_memory_write = config.clone();
    let create_consolidated_memory = ExecutableTool::new(
        ToolSpec::new(
            "create_consolidated_memory",
            "Create a consolidated memory from one or more existing active memories.",
            JsonSchema::object(
                [
                    ("name", json!({"type": "string"})),
                    ("text", json!({"type": "string"})),
                    (
                        "source_names",
                        json!({
                            "type": "array",
                            "items": {"type": "string"}
                        }),
                    ),
                ],
                ["name", "text", "source_names"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires a string name",
                );
            };
            let Some(text) = arguments.get("text").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires string text",
                );
            };
            let Some(source_names) = arguments.get("source_names").and_then(Value::as_array) else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires array source_names",
                );
            };
            let source_names = source_names
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>();
            let Some(source_names) = source_names else {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires string source_names entries",
                );
            };
            if source_names.is_empty() {
                return ToolExecutionResult::error(
                    "create_consolidated_memory requires at least one source memory",
                );
            }
            match create_consolidated_memory_from_config(
                &config_for_consolidated_memory_write,
                name,
                text,
                &source_names,
            ) {
                Ok(path) => ToolExecutionResult::success(
                    json!({
                        "created": true,
                        "file_path": path.display().to_string(),
                    })
                    .to_string(),
                ),
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to create consolidated memory: {error}"
                )),
            }
        },
    );

    let config_for_memory_show = config.clone();
    let show_memory = ExecutableTool::new(
        ToolSpec::new(
            "show_memory",
            "Show one active memory by exact name.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("name", json!({"type": "string"})),
                ],
                [] as [&str; 0],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments
                .get("memory_name")
                .and_then(Value::as_str)
                .or_else(|| arguments.get("name").and_then(Value::as_str))
            else {
                return ToolExecutionResult::error("show_memory requires a string name");
            };
            match open_sqlite_connection(&config_for_memory_show.database_path) {
                Ok(connection) => {
                    match find_active_memory_by_name_in_scope(
                        &connection,
                        name,
                        &config_for_memory_show.memory_dir,
                    ) {
                        Ok(Some(memory)) => ToolExecutionResult::success(
                            json!({
                                "name": memory.name,
                                "file_path": memory.file_path,
                                "body": memory.body,
                                "updated_at_unix": memory.updated_at_unix,
                            })
                            .to_string(),
                        ),
                        Ok(None) => ToolExecutionResult::error(format!(
                            "Memory '{name}' not found for the current user."
                        )),
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let config_for_memory_print = config.clone();
    let print_memory = ExecutableTool::new(
        ToolSpec::new(
            "print_memory",
            "Print the detailed content of an active memory in fact format.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error("print_memory requires a string memory_name");
            };
            match open_sqlite_connection(&config_for_memory_print.database_path) {
                Ok(connection) => {
                    match find_active_memory_by_name_in_scope(
                        &connection,
                        name,
                        &config_for_memory_print.memory_dir,
                    ) {
                        Ok(Some(memory)) => {
                            ToolExecutionResult::success(format_memory_detail(&memory))
                        }
                        Ok(None) => ToolExecutionResult::success(format!(
                            "Memory '{name}' not found for the current user."
                        )),
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let config_for_memory_list = config.clone();
    let list_memories = ExecutableTool::new(
        ToolSpec::new(
            "list_memories",
            "List active memories available to Elroy.",
            JsonSchema::object([("limit", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = arguments.get("limit").and_then(Value::as_i64).unwrap_or(10) as usize;
            match open_sqlite_connection(&config_for_memory_list.database_path) {
                Ok(connection) => {
                    match list_active_memories_in_scope(
                        &connection,
                        &config_for_memory_list.memory_dir,
                        limit,
                    ) {
                        Ok(memories) => {
                            let payload = memories
                                .into_iter()
                                .map(|memory| {
                                    json!({
                                        "name": memory.name,
                                        "file_path": memory.file_path,
                                        "excerpt": elroy_core::excerpt(&memory.body, 180),
                                        "updated_at_unix": memory.updated_at_unix,
                                    })
                                })
                                .collect::<Vec<_>>();
                            ToolExecutionResult::success(
                                serde_json::to_string_pretty(&payload)
                                    .expect("memory payload should serialize"),
                            )
                        }
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let config_for_memory_print_list = config.clone();
    let print_memories = ExecutableTool::new(
        ToolSpec::new(
            "print_memories",
            "List active memories available to Elroy.",
            JsonSchema::object([("n", json!({"type": "integer"}))], [] as [&str; 0]),
        ),
        move |arguments| {
            let limit = elroy_tools::argument_limit(&arguments, 10);
            match open_sqlite_connection(&config_for_memory_print_list.database_path) {
                Ok(connection) => {
                    match list_active_memories_in_scope(
                        &connection,
                        &config_for_memory_print_list.memory_dir,
                        limit,
                    ) {
                        Ok(memories) => {
                            ToolExecutionResult::success(format_memory_listing(&memories))
                        }
                        Err(error) => {
                            ToolExecutionResult::error(format!("database query failed: {error}"))
                        }
                    }
                }
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to open database: {error}"))
                }
            }
        },
    );

    let config_for_memory_search = config.clone();
    let search_memories = ExecutableTool::new(
        ToolSpec::new(
            "search_memories",
            "Search active memories by keyword.",
            JsonSchema::object([("query", json!({"type": "string"}))], ["query"]),
        ),
        move |arguments| {
            let Some(query) = arguments.get("query").and_then(Value::as_str) else {
                return ToolExecutionResult::error("search_memories requires a string query");
            };
            let limit = elroy_tools::argument_limit(&arguments, 10);
            search_memories_from_config(&config_for_memory_search, query, limit)
        },
    );

    let config_for_memory_examine = config.clone();
    let examine_memories = ExecutableTool::new(
        ToolSpec::new(
            "examine_memories",
            "Deeply examine active memories and items for relevance to a complex question.",
            JsonSchema::object([("question", json!({"type": "string"}))], ["question"]),
        ),
        move |arguments| {
            let Some(question) = arguments.get("question").and_then(Value::as_str) else {
                return ToolExecutionResult::error("examine_memories requires a string question");
            };
            examine_memories_from_config(&config_for_memory_examine, question)
        },
    );

    let config_for_source_list = config.clone();
    let get_source_list_for_memory = ExecutableTool::new(
        ToolSpec::new(
            "get_source_list_for_memory",
            "List available sources for one active memory.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "get_source_list_for_memory requires a string memory_name",
                );
            };
            match get_source_list_for_memory_from_config(&config_for_source_list, name) {
                Ok(result) => ToolExecutionResult::success(result),
                Err(error) => ToolExecutionResult::error(error.to_string()),
            }
        },
    );

    let config_for_source_content = config.clone();
    let get_source_content_for_memory = ExecutableTool::new(
        ToolSpec::new(
            "get_source_content_for_memory",
            "Show the file-backed source content for one active memory.",
            JsonSchema::object(
                [
                    ("memory_name", json!({"type": "string"})),
                    ("index", json!({"type": "integer"})),
                ],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "get_source_content_for_memory requires a string memory_name",
                );
            };
            let index = arguments.get("index").and_then(Value::as_i64).unwrap_or(0) as usize;
            match get_source_content_for_memory_from_config(&config_for_source_content, name, index)
            {
                Ok(result) => ToolExecutionResult::success(result),
                Err(error) => ToolExecutionResult::error(error.to_string()),
            }
        },
    );

    let get_fast_recall = ExecutableTool::new(
        ToolSpec::new(
            "get_fast_recall",
            "No-op tool used to acknowledge synthetic recall context.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| ToolExecutionResult::success("OK".to_string()),
    );

    let get_reflective_recall = ExecutableTool::new(
        ToolSpec::new(
            "get_reflective_recall",
            "No-op tool used to acknowledge synthetic reflective recall context.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| ToolExecutionResult::success("OK".to_string()),
    );

    let config_for_add_to_context = config.clone();
    let add_memory_to_current_context = ExecutableTool::new(
        ToolSpec::new(
            "add_memory_to_current_context",
            "Pin one active memory into the persisted transcript context.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "add_memory_to_current_context requires a string memory_name",
                );
            };
            let mut connection =
                match open_sqlite_connection(&config_for_add_to_context.database_path) {
                    Ok(connection) => connection,
                    Err(error) => {
                        return ToolExecutionResult::error(format!(
                            "failed to open database: {error}"
                        ));
                    }
                };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            let transcript = match load_validated_runtime_transcript(
                &mut connection,
                &config_for_add_to_context.assistant_name,
                config_for_add_to_context.llm_provider() == elroy_config::LlmProvider::Anthropic,
            ) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load validated transcript: {error}"
                    ));
                }
            };
            let memory = match find_active_memory_by_name_in_scope(
                &connection,
                memory_name,
                &config_for_add_to_context.memory_dir,
            ) {
                Ok(Some(memory)) => memory,
                Ok(None) => {
                    return ToolExecutionResult::error(format!(
                        "Memory '{memory_name}' not found for the current user."
                    ));
                }
                Err(error) => {
                    return ToolExecutionResult::error(format!("database query failed: {error}"));
                }
            };
            let tool_call_id = context_memory_tool_call_id(&memory.name);
            if transcript
                .iter()
                .any(|message| message_matches_tool_call_id(message, &tool_call_id))
            {
                return ToolExecutionResult::success(format!(
                    "Memory '{}' added to context.",
                    memory.name
                ));
            }
            let mut updated_transcript = transcript;
            updated_transcript.extend(context_memory_tool_messages(&memory));
            match replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript) {
                Ok(()) => ToolExecutionResult::success(format!(
                    "Memory '{}' added to context.",
                    memory.name
                )),
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to persist context messages: {error}"
                )),
            }
        },
    );

    let config_for_drop_from_context = config.clone();
    let drop_memory_from_current_context = ExecutableTool::new(
        ToolSpec::new(
            "drop_memory_from_current_context",
            "Remove one pinned memory from the persisted transcript context.",
            JsonSchema::object(
                [("memory_name", json!({"type": "string"}))],
                ["memory_name"],
            ),
        ),
        move |arguments| {
            let Some(memory_name) = arguments.get("memory_name").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "drop_memory_from_current_context requires a string memory_name",
                );
            };
            let mut connection =
                match open_sqlite_connection(&config_for_drop_from_context.database_path) {
                    Ok(connection) => connection,
                    Err(error) => {
                        return ToolExecutionResult::error(format!(
                            "failed to open database: {error}"
                        ));
                    }
                };
            if let Err(error) = run_migrations(&mut connection) {
                return ToolExecutionResult::error(format!("failed to run migrations: {error}"));
            }
            let transcript = match load_validated_runtime_transcript(
                &mut connection,
                &config_for_drop_from_context.assistant_name,
                config_for_drop_from_context.llm_provider() == elroy_config::LlmProvider::Anthropic,
            ) {
                Ok(transcript) => transcript,
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load validated transcript: {error}"
                    ));
                }
            };
            let tool_call_id = context_memory_tool_call_id(memory_name);
            let updated_transcript = transcript
                .into_iter()
                .filter(|message| !message_matches_tool_call_id(message, &tool_call_id))
                .collect::<Vec<_>>();
            match replace_context_messages(&mut connection, LOCAL_USER_TOKEN, &updated_transcript) {
                Ok(()) => ToolExecutionResult::success(format!(
                    "Memory '{}' dropped from context.",
                    memory_name.to_ascii_lowercase()
                )),
                Err(error) => ToolExecutionResult::error(format!(
                    "failed to persist context messages: {error}"
                )),
            }
        },
    );

    vec![
        create_memory,
        update_memory,
        update_outdated_or_incorrect_memory,
        archive_memory,
        create_consolidated_memory,
        show_memory,
        print_memory,
        list_memories,
        print_memories,
        search_memories,
        examine_memories,
        get_source_list_for_memory,
        get_source_content_for_memory,
        get_fast_recall,
        get_reflective_recall,
        add_memory_to_current_context,
        drop_memory_from_current_context,
    ]
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use super::memory_tools;
    use elroy_config::AppConfig;
    use elroy_db::{
        BootstrapPlan, LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection,
        replace_context_messages, run_migrations,
    };
    use elroy_llm::{ConversationMessage, MessageRole};
    use elroy_recall::message_matches_tool_call_id;
    use elroy_tools::ExecutableToolRegistry;

    fn persisted_context_messages(config: &AppConfig) -> Vec<ConversationMessage> {
        let mut connection =
            open_sqlite_connection(&config.database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        load_context_messages(&mut connection, LOCAL_USER_TOKEN).expect("context should load")
    }

    #[test]
    fn memory_tools_can_show_and_print_memories() {
        let unique = format!(
            "elroy-rs-memory-tools-show-{}",
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
        fs::write(
            memory_dir.join("runner_notes.md"),
            "remember the hill workout\n",
        )
        .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let memory = registry.invoke("show_memory", "{\"memory_name\":\"runner notes\"}");
        let printed_memory = registry.invoke("print_memory", "{\"memory_name\":\"runner notes\"}");
        let missing_printed_memory =
            registry.invoke("print_memory", "{\"memory_name\":\"missing\"}");
        let printed_memories = registry.invoke("print_memories", "{\"n\":10}");

        assert!(!memory.is_error);
        assert!(memory.content.contains("remember the hill workout"));
        assert!(!printed_memory.is_error);
        assert!(printed_memory.content.starts_with("#runner notes\n"));
        assert!(printed_memory.content.contains("remember the hill workout"));
        assert!(!missing_printed_memory.is_error);
        assert_eq!(
            missing_printed_memory.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(!printed_memories.is_error);
        assert!(printed_memories.content.contains("Memories"));
        assert!(printed_memories.content.contains("runner notes"));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn print_memories_lists_oldest_visible_first() {
        let unique = format!(
            "elroy-rs-memory-tools-order-{}",
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
        fs::write(memory_dir.join("older.md"), "remember the earlier note\n")
            .expect("older memory should be written");
        std::thread::sleep(Duration::from_secs(1));
        fs::write(memory_dir.join("newer.md"), "remember the later note\n")
            .expect("newer memory should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let printed = registry.invoke("print_memories", "{\"n\":10}");

        assert!(!printed.is_error);
        let older_index = printed
            .content
            .find("- older | Text: remember the earlier note")
            .expect("older memory should appear");
        let newer_index = printed
            .content
            .find("- newer | Text: remember the later note")
            .expect("newer memory should appear");
        assert!(older_index < newer_index);

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_tools_can_list_source_metadata_and_content() {
        let unique = format!(
            "elroy-rs-memory-tools-sources-{}",
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
        fs::write(
            memory_dir.join("runner_notes.md"),
            "remember the hill workout\nwith the harder second interval\n",
        )
        .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path.clone();
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        let missing_source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"missing\"}",
        );
        let missing_source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"missing\"}",
        );
        let out_of_range = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\",\"index\":1}",
        );

        assert!(!source.is_error);
        assert_eq!(source.content, "No sources found for memory 'runner notes'");
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[]");
        assert!(missing_source_list.is_error);
        assert_eq!(
            missing_source_list.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(missing_source.is_error);
        assert_eq!(
            missing_source.content,
            "Memory 'missing' not found for the current user."
        );
        assert!(!out_of_range.is_error);
        assert_eq!(
            out_of_range.content,
            "No sources found for memory 'runner notes'"
        );

        let mut connection = open_sqlite_connection(&database_path).expect("database should open");
        run_migrations(&mut connection).expect("migrations should run");
        replace_context_messages(
            &mut connection,
            LOCAL_USER_TOKEN,
            &[ConversationMessage::new(
                MessageRole::User,
                "Hello, I ran a marathon today!",
            )],
        )
        .expect("context messages should persist");

        let created = registry.invoke(
            "create_memory",
            "{\"name\":\"Running progress\",\"text\":\"I ran a marathon today\"}",
        );
        let created_source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"running progress\"}",
        );
        let created_source = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"running progress\"}",
        );
        let created_out_of_range = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"running progress\",\"index\":1}",
        );

        assert!(!created.is_error);
        assert_eq!(created.content, "New memory created: Running progress");
        assert!(!created_source_list.is_error);
        assert_eq!(
            created_source_list.content,
            "[[\"ContextMessageSet\",\"1\"]]"
        );
        assert!(!created_source.is_error);
        assert!(
            created_source
                .content
                .contains("user: Hello, I ran a marathon today!")
        );
        assert!(created_out_of_range.is_error);
        assert_eq!(
            created_out_of_range.content,
            "Index 1 out of range. Available indices: [0]"
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_tools_can_add_and_drop_memory_from_current_context() {
        let unique = format!(
            "elroy-rs-memory-tools-context-memory-{}",
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
        fs::write(
            memory_dir.join("travel_preference.md"),
            "User likes window seats on long flights.\n",
        )
        .expect("memory should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir;
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let add = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!add.is_error);
        assert_eq!(add.content, "Memory 'travel preference' added to context.");

        let stored = persisted_context_messages(&config);
        assert!(stored.iter().any(|message| {
            message.tool_calls.as_ref().is_some_and(|calls| {
                calls.iter().any(|call| {
                    call.id == "context-memory:travel preference" && call.name == "get_fast_recall"
                })
            })
        }));
        assert!(stored.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:travel preference")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("travel preference"))
        }));

        let add_again = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!add_again.is_error);
        let stored_again = persisted_context_messages(&config);
        assert_eq!(
            stored_again
                .iter()
                .filter(|message| {
                    message_matches_tool_call_id(message, "context-memory:travel preference")
                })
                .count(),
            2
        );

        let dropped = registry.invoke(
            "drop_memory_from_current_context",
            "{\"memory_name\":\"travel preference\"}",
        );
        assert!(!dropped.is_error);
        assert_eq!(
            dropped.content,
            "Memory 'travel preference' dropped from context."
        );
        let stripped = persisted_context_messages(&config);
        assert!(!stripped.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:travel preference")
        }));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_tools_can_update_and_archive_memories() {
        let unique = format!(
            "elroy-rs-memory-tools-mutations-{}",
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
        fs::write(memory_dir.join("runner_notes.md"), "old text\n")
            .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let update = registry.invoke(
            "update_memory",
            "{\"memory_name\":\"runner notes\",\"text\":\"new text\"}",
        );
        assert!(!update.is_error);
        assert!(
            fs::read_to_string(memory_dir.join("runner_notes.md"))
                .expect("updated memory should be readable")
                .contains("new text")
        );

        let archive = registry.invoke("archive_memory", "{\"memory_name\":\"runner notes\"}");
        assert!(!archive.is_error);
        assert!(memory_dir.join("archive").join("runner_notes.md").exists());

        let connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert!(active_memories.is_empty());

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_tools_refresh_pinned_current_context_after_updates() {
        let unique = format!(
            "elroy-rs-memory-tools-refresh-context-{}",
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
        fs::write(memory_dir.join("runner_notes.md"), "old text\n")
            .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let add = registry.invoke(
            "add_memory_to_current_context",
            "{\"memory_name\":\"runner notes\"}",
        );
        assert!(!add.is_error);

        let update = registry.invoke(
            "update_memory",
            "{\"memory_name\":\"runner notes\",\"text\":\"new text\"}",
        );
        assert!(!update.is_error);

        let updated_context = persisted_context_messages(&config);
        assert!(updated_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:runner notes")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("new text"))
        }));
        assert!(!updated_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:runner notes")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("old text"))
        }));
        assert_eq!(
            updated_context
                .iter()
                .filter(|message| message_matches_tool_call_id(
                    message,
                    "context-memory:runner notes"
                ))
                .count(),
            2
        );

        let outdated = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"runner notes\",\"update_text\":\"second correction\"}",
        );
        assert!(!outdated.is_error);

        let refreshed_context = persisted_context_messages(&config);
        assert!(refreshed_context.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:runner notes")
                && message
                    .content
                    .as_deref()
                    .is_some_and(|content| content.contains("second correction"))
        }));
        assert_eq!(
            refreshed_context
                .iter()
                .filter(|message| message_matches_tool_call_id(
                    message,
                    "context-memory:runner notes"
                ))
                .count(),
            2
        );

        let archived = registry.invoke("archive_memory", "{\"memory_name\":\"runner notes\"}");
        assert!(!archived.is_error);
        let stripped = persisted_context_messages(&config);
        assert!(!stripped.iter().any(|message| {
            message_matches_tool_call_id(message, "context-memory:runner notes")
        }));

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn memory_tools_can_append_outdated_memory_update() {
        let unique = format!(
            "elroy-rs-memory-tools-outdated-update-{}",
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
        fs::write(memory_dir.join("runner_notes.md"), "old text\n")
            .expect("memory file should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir;
        config.database_path = database_path;
        elroy_db::bootstrap_database(&BootstrapPlan::from_config(&config))
            .expect("bootstrap should succeed");

        let registry = ExecutableToolRegistry::new(memory_tools(&config));
        let update = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"runner notes\",\"update_text\":\"new correction\"}",
        );
        assert!(!update.is_error);
        assert_eq!(update.content, "Memory 'runner notes' has been updated");

        let missing_update = registry.invoke(
            "update_outdated_or_incorrect_memory",
            "{\"memory_name\":\"missing\",\"update_text\":\"unused\"}",
        );
        assert!(!missing_update.is_error);
        assert_eq!(missing_update.content, "Memory 'missing' not found");

        let file_text =
            fs::read_to_string(memory_dir.join("runner_notes.md")).expect("memory should read");
        assert!(file_text.contains("old text"));
        assert!(file_text.contains("Update ("));
        assert!(file_text.contains("new correction"));
        assert!(memory_dir.join("archive").join("runner_notes.md").exists());

        let source_list = registry.invoke(
            "get_source_list_for_memory",
            "{\"memory_name\":\"runner notes\"}",
        );
        assert!(!source_list.is_error);
        assert_eq!(source_list.content, "[[\"Memory\",\"runner notes\"]]");

        let source_content = registry.invoke(
            "get_source_content_for_memory",
            "{\"memory_name\":\"runner notes\",\"index\":0}",
        );
        assert!(!source_content.is_error);
        assert!(source_content.content.contains("#runner notes"));
        assert!(source_content.content.contains("old text"));
        assert!(!source_content.content.contains("new correction"));

        let connection =
            open_sqlite_connection(&config.database_path).expect("database should reopen");
        let active_memories =
            elroy_db::list_active_memories(&connection, 10).expect("active memories should list");
        assert_eq!(active_memories.len(), 1);
        assert!(active_memories[0].body.contains("new correction"));
        let historical_rows: Vec<(String, bool, String)> = {
            let mut statement = connection
                .prepare(
                    "SELECT file_path, is_active, body
                     FROM memories
                     WHERE LOWER(name) = LOWER(?1)
                     ORDER BY is_active DESC, updated_at_unix DESC, file_path ASC",
                )
                .expect("memory history query should prepare");
            let rows = statement
                .query_map(["runner notes"], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)? != 0,
                        row.get::<_, String>(2)?,
                    ))
                })
                .expect("memory history rows should map");
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .expect("memory history rows should collect")
        };
        assert_eq!(historical_rows.len(), 2);
        assert_eq!(
            historical_rows[0].0,
            memory_dir.join("runner_notes.md").display().to_string()
        );
        assert!(historical_rows[0].1);
        assert!(historical_rows[0].2.contains("old text"));
        assert!(historical_rows[0].2.contains("new correction"));
        assert_eq!(
            historical_rows[1].0,
            memory_dir
                .join("archive")
                .join("runner_notes.md")
                .display()
                .to_string()
        );
        assert!(!historical_rows[1].1);
        assert!(historical_rows[1].2.contains("old text"));
        assert!(!historical_rows[1].2.contains("new correction"));

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
