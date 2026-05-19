use elroy_config::AppConfig;
use elroy_core::memory_store::{archive_memory_file, update_memory_body};
use elroy_db::{
    BootstrapPlan, LOCAL_USER_TOKEN, load_context_messages, open_sqlite_connection, run_migrations,
};
use elroy_recall::{
    archive_memory_file_from_config, create_consolidated_memory_from_config,
    create_memory_file_from_context_messages, examine_memories_from_config,
    find_active_memory_by_name_in_scope, format_memory_detail, format_memory_listing,
    get_source_content_for_memory_from_config, get_source_list_for_memory_from_config,
    list_active_memories_in_scope, memory_consolidation_settings_from_app_config,
 mutate_memory_file_from_config,
    record_memory_creation_and_maybe_consolidate, search_memories_from_config,
    update_outdated_or_incorrect_memory_from_config,
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
    ]
}
