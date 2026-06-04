use std::path::PathBuf;

use chrono::Utc;
use elroy_db::LOCAL_USER_TOKEN;
use elroy_tools::{ExecutableTool, JsonSchema, ToolExecutionResult, ToolSpec};
use serde_json::{Value, json};

use crate::{
    FeatureRequestRecord, find_best_feature_request_match, get_feature_request,
    list_feature_requests, update_feature_request, write_new_feature_request,
};

pub fn feature_request_tools(home_dir: PathBuf) -> Vec<ExecutableTool> {
    let home_for_list = home_dir.clone();
    let list_feature_requests_tool = ExecutableTool::new(
        ToolSpec::new(
            "list_feature_requests",
            "List the current markdown-backed feature requests.",
            JsonSchema::object(Vec::<(String, Value)>::new(), [] as [&str; 0]),
        ),
        move |_| match list_feature_requests(&home_for_list) {
            Ok(records) => ToolExecutionResult::success(feature_request_listing_content(&records)),
            Err(error) => {
                ToolExecutionResult::error(format!("failed to list feature requests: {error}"))
            }
        },
    );

    let home_for_make = home_dir.clone();
    let make_feature_request = ExecutableTool::new(
        ToolSpec::new(
            "make_feature_request",
            "Create or merge a markdown feature request for future product work.",
            JsonSchema::object(
                [
                    ("title", json!({"type": "string"})),
                    ("description", json!({"type": "string"})),
                    ("rationale", json!({"type": "string"})),
                ],
                ["title", "description"],
            ),
        ),
        move |arguments| {
            let Some(title) = arguments.get("title").and_then(Value::as_str) else {
                return ToolExecutionResult::error("make_feature_request requires a string title");
            };
            let Some(description) = arguments.get("description").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "make_feature_request requires a string description",
                );
            };
            let title = title.trim();
            let description = description.trim();
            let rationale =
                normalize_optional_tool_string(arguments.get("rationale").and_then(Value::as_str));
            let supporting_context = build_feature_request_supporting_context(
                LOCAL_USER_TOKEN,
                title,
                description,
                rationale,
            );

            match find_best_feature_request_match(&home_for_make, title, description) {
                Ok(Some(matched)) => {
                    let mut aliases = matched.record.aliases.clone();
                    if title != matched.record.title && !aliases.iter().any(|alias| alias == title)
                    {
                        aliases.push(title.to_string());
                        aliases.sort();
                    }
                    match update_feature_request(
                        &matched.record,
                        None,
                        None,
                        Some(&aliases),
                        None,
                        None,
                        Some(Some(&merge_feature_request_supporting_context(
                            matched.record.supporting_context.as_deref(),
                            &supporting_context,
                        ))),
                    ) {
                        Ok(updated) => ToolExecutionResult::success(format!(
                            "Merged into existing feature request: {} ({}; match reason: {}).",
                            updated.title,
                            updated
                                .path
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or_default(),
                            matched.reason
                        )),
                        Err(error) => ToolExecutionResult::error(format!(
                            "failed to update feature request: {error}"
                        )),
                    }
                }
                Ok(None) => match write_new_feature_request(
                    &home_for_make,
                    title,
                    description,
                    rationale,
                    Some(&supporting_context),
                    "user_request",
                ) {
                    Ok(created) => ToolExecutionResult::success(format!(
                        "Created feature request: {} ({}).",
                        created.title,
                        created
                            .path
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or_default()
                    )),
                    Err(error) => ToolExecutionResult::error(format!(
                        "failed to create feature request: {error}"
                    )),
                },
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to match feature request: {error}"))
                }
            }
        },
    );

    let home_for_edit = home_dir.clone();
    let edit_feature_request = ExecutableTool::new(
        ToolSpec::new(
            "edit_feature_request",
            "Edit an existing markdown feature request.",
            JsonSchema::object(
                [
                    ("identifier", json!({"type": "string"})),
                    ("title", json!({"type": "string"})),
                    ("description", json!({"type": "string"})),
                    ("rationale", json!({"type": "string"})),
                    ("status", json!({"type": "string"})),
                ],
                ["identifier"],
            ),
        ),
        move |arguments| {
            let Some(identifier) = arguments.get("identifier").and_then(Value::as_str) else {
                return ToolExecutionResult::error(
                    "edit_feature_request requires a string identifier",
                );
            };
            let title =
                normalize_optional_tool_string(arguments.get("title").and_then(Value::as_str));
            let description = normalize_optional_tool_string(
                arguments.get("description").and_then(Value::as_str),
            );
            let rationale =
                normalize_optional_tool_string(arguments.get("rationale").and_then(Value::as_str));
            let status =
                normalize_optional_tool_string(arguments.get("status").and_then(Value::as_str));

            let record = match get_feature_request(&home_for_edit, identifier.trim()) {
                Ok(Some(record)) => record,
                Ok(None) => {
                    return ToolExecutionResult::success(format!(
                        "Feature request '{}' not found.",
                        identifier
                    ));
                }
                Err(error) => {
                    return ToolExecutionResult::error(format!(
                        "failed to load feature request: {error}"
                    ));
                }
            };
            let edit_context = format!(
                "- Edited at: {}\n- Edited by user token: {}",
                Utc::now().to_rfc3339(),
                LOCAL_USER_TOKEN
            );
            match update_feature_request(
                &record,
                title,
                status,
                None,
                description,
                Some(rationale),
                Some(Some(&merge_feature_request_supporting_context(
                    record.supporting_context.as_deref(),
                    &edit_context,
                ))),
            ) {
                Ok(updated) => ToolExecutionResult::success(format!(
                    "Updated feature request: {} ({}).",
                    updated.title,
                    updated
                        .path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or_default()
                )),
                Err(error) => {
                    ToolExecutionResult::error(format!("failed to update feature request: {error}"))
                }
            }
        },
    );

    vec![
        list_feature_requests_tool,
        make_feature_request,
        edit_feature_request,
    ]
}

fn feature_request_listing_content(records: &[FeatureRequestRecord]) -> String {
    if records.is_empty() {
        return "No feature requests found.".to_string();
    }

    let mut lines = vec![
        format!("Feature requests ({}):", records.len()),
        String::new(),
    ];
    for record in records {
        let aliases = if record.aliases.is_empty() {
            String::new()
        } else {
            format!(" | aliases: {}", record.aliases.join(", "))
        };
        lines.push(format!(
            "- {} [{}] ({}){}",
            record.title,
            record.status,
            record
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
            aliases
        ));
        lines.push(format!("  Summary: {}", record.summary));
    }
    lines.join("\n")
}

fn normalize_optional_tool_string(value: Option<&str>) -> Option<&str> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

fn build_feature_request_supporting_context(
    user_token: &str,
    title: &str,
    description: &str,
    rationale: Option<&str>,
) -> String {
    let mut lines = vec![
        format!("- Captured at: {}", Utc::now().to_rfc3339()),
        format!("- Requested title: {title}"),
        format!("- Description: {}", description.trim()),
    ];
    if let Some(rationale) = rationale {
        lines.push(format!("- Rationale: {}", rationale.trim()));
    }
    lines.push(format!("- User token: {user_token}"));
    lines.join("\n")
}

fn merge_feature_request_supporting_context(existing: Option<&str>, new_context: &str) -> String {
    match normalize_optional_tool_string(existing) {
        None => new_context.to_string(),
        Some(existing) if existing.contains(new_context) => existing.to_string(),
        Some(existing) => format!("{}\n\n{}", existing.trim_end(), new_context),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::feature_request_tools;
    use crate::list_feature_requests;
    use elroy_tools::ExecutableToolRegistry;

    #[test]
    fn feature_request_tools_can_list_make_merge_and_edit_requests() {
        let unique = format!(
            "elroy-rs-feature-requests-tools-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        fs::create_dir_all(&home).expect("home should be created");

        let registry = ExecutableToolRegistry::new(feature_request_tools(home.clone()));

        let empty = registry.invoke("list_feature_requests", "{}");
        assert!(!empty.is_error);
        assert_eq!(empty.content, "No feature requests found.");

        let created = registry.invoke(
            "make_feature_request",
            "{\"title\":\"Add calendar sync\",\"description\":\"Sync Elroy tasks to a calendar provider.\",\"rationale\":\"Users want a unified schedule.\"}",
        );
        assert!(!created.is_error);
        assert!(
            created
                .content
                .contains("Created feature request: Add calendar sync")
        );

        let merged = registry.invoke(
            "make_feature_request",
            "{\"title\":\"Add calendar synchronization\",\"description\":\"Sync tasks to an external calendar.\",\"rationale\":\"Users want calendar parity.\"}",
        );
        assert!(!merged.is_error);
        assert!(
            merged
                .content
                .contains("Merged into existing feature request: Add calendar sync")
        );

        let listed = registry.invoke("list_feature_requests", "{}");
        assert!(!listed.is_error);
        assert!(listed.content.contains("Feature requests (1):"));
        assert!(
            listed
                .content
                .contains("aliases: Add calendar synchronization")
        );

        let edited = registry.invoke(
            "edit_feature_request",
            "{\"identifier\":\"add calendar sync\",\"status\":\"closed\",\"description\":\"Sync Elroy tasks to a calendar provider with account selection.\"}",
        );
        assert!(!edited.is_error);
        assert!(
            edited
                .content
                .contains("Updated feature request: Add calendar sync")
        );

        let records = list_feature_requests(&home).expect("feature requests should list");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, "closed");
        assert_eq!(
            records[0].summary,
            "Sync Elroy tasks to a calendar provider with account selection."
        );
        assert!(
            records[0]
                .supporting_context
                .as_deref()
                .is_some_and(|content| content.contains("Edited by user token: local-user"))
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }
}
