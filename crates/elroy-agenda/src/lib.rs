use std::path::{Path, PathBuf};

use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecklistItem {
    pub id: i64,
    pub text: String,
    pub completed: bool,
    pub due_date: Option<String>,
}

pub fn create_agenda_file(
    agenda_dir: &Path,
    name: &str,
    text: &str,
    date: Option<&str>,
    trigger_datetime: Option<&str>,
    trigger_context: Option<&str>,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(agenda_dir)?;
    let path = unique_markdown_path(agenda_dir, name);
    let mut frontmatter = vec![format!("date: {}", date.unwrap_or("unscheduled"))];
    frontmatter.push("completed: false".to_string());
    if let Some(trigger_datetime) = trigger_datetime {
        frontmatter.push(format!("trigger_datetime: {trigger_datetime}"));
    }
    if let Some(trigger_context) = trigger_context {
        frontmatter.push(format!("trigger_context: {trigger_context}"));
    }
    let content = format!("---\n{}\n---\n\n{text}\n", frontmatter.join("\n"));
    std::fs::write(&path, content)?;
    Ok(path)
}

pub fn append_agenda_update(path: &Path, note: &str) -> std::io::Result<()> {
    let (frontmatter, body) = read_markdown_parts(path)?;
    let timestamp = local_timestamp_string();
    let update_line = format!("- **{timestamp}**: {note}");
    let new_body = if let Some((main, updates)) = split_updates_section(&body) {
        if updates.is_empty() {
            format!("{}\n\n## Updates\n\n{}", main.trim(), update_line)
                .trim()
                .to_string()
        } else {
            format!(
                "{}\n\n## Updates\n\n{}\n{}",
                main.trim(),
                updates.trim(),
                update_line
            )
            .trim()
            .to_string()
        }
    } else if body.trim().is_empty() {
        format!("## Updates\n\n{update_line}")
    } else {
        format!("{}\n\n## Updates\n\n{update_line}", body.trim())
    };
    write_markdown_parts(path, frontmatter.as_ref(), &new_body)
}

pub fn update_agenda_body(path: &Path, text: &str) -> std::io::Result<()> {
    let (frontmatter, _) = read_markdown_parts(path)?;
    write_markdown_parts(path, frontmatter.as_ref(), text)
}

pub fn mark_agenda_item_completed(
    path: &Path,
    closing_comment: Option<&str>,
) -> std::io::Result<()> {
    update_frontmatter_fields(path, |frontmatter| {
        frontmatter.insert(
            YamlValue::String("completed".to_string()),
            YamlValue::Bool(true),
        );
        frontmatter.insert(
            YamlValue::String("status".to_string()),
            YamlValue::String("completed".to_string()),
        );
        if let Some(closing_comment) = closing_comment {
            frontmatter.insert(
                YamlValue::String("closing_comment".to_string()),
                YamlValue::String(closing_comment.to_string()),
            );
        }
    })
}

pub fn mark_agenda_item_deleted(path: &Path, closing_comment: Option<&str>) -> std::io::Result<()> {
    update_frontmatter_fields(path, |frontmatter| {
        frontmatter.insert(
            YamlValue::String("status".to_string()),
            YamlValue::String("deleted".to_string()),
        );
        if let Some(closing_comment) = closing_comment {
            frontmatter.insert(
                YamlValue::String("closing_comment".to_string()),
                YamlValue::String(closing_comment.to_string()),
            );
        }
    })
}

pub fn rename_agenda_file(path: &Path, new_name: &str) -> std::io::Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("agenda path has no parent directory"))?;
    let candidate = unique_markdown_path(parent, new_name);
    std::fs::rename(path, &candidate)?;
    Ok(candidate)
}

pub fn get_checklist(path: &Path) -> std::io::Result<Vec<ChecklistItem>> {
    let (frontmatter, _) = read_markdown_parts(path)?;
    Ok(frontmatter
        .and_then(|mapping| {
            mapping
                .get(YamlValue::String("checklist".to_string()))
                .cloned()
        })
        .map(|value| normalize_checklist(&value))
        .unwrap_or_default())
}

pub fn add_checklist_item(path: &Path, text: &str, due_date: Option<&str>) -> std::io::Result<i64> {
    let mut checklist = get_checklist(path)?;
    let next_id = checklist.iter().map(|item| item.id).max().unwrap_or(0) + 1;
    checklist.push(ChecklistItem {
        id: next_id,
        text: text.to_string(),
        completed: false,
        due_date: due_date.map(ToString::to_string),
    });
    write_checklist(path, &checklist)?;
    Ok(next_id)
}

pub fn update_checklist_item(
    path: &Path,
    item_id: i64,
    text: Option<&str>,
    completed: Option<bool>,
) -> std::io::Result<ChecklistItem> {
    let mut checklist = get_checklist(path)?;
    let Some(item) = checklist.iter_mut().find(|item| item.id == item_id) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no checklist item with id {item_id} found"),
        ));
    };
    if let Some(text) = text {
        item.text = text.to_string();
    }
    if let Some(completed) = completed {
        item.completed = completed;
    }
    let updated = item.clone();
    write_checklist(path, &checklist)?;
    Ok(updated)
}

fn sanitize_filename(name: &str) -> String {
    let cleaned = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = cleaned.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "item".to_string()
    } else {
        let mut compact = String::new();
        let mut previous_underscore = false;
        for ch in trimmed.chars() {
            if ch == '_' {
                if !previous_underscore {
                    compact.push(ch);
                }
                previous_underscore = true;
            } else {
                compact.push(ch);
                previous_underscore = false;
            }
        }
        compact
    }
}

fn unique_markdown_path(root: &Path, name: &str) -> PathBuf {
    let base = sanitize_filename(name);
    let first = root.join(format!("{base}.md"));
    if !first.exists() {
        return first;
    }

    let mut counter = 2;
    loop {
        let candidate = root.join(format!("{base}-{counter}.md"));
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

fn split_updates_section(body: &str) -> Option<(String, String)> {
    let marker = "\n## Updates\n\n";
    if let Some((main, updates)) = body.split_once(marker) {
        return Some((main.trim_end().to_string(), updates.trim().to_string()));
    }
    let stripped = body.trim();
    stripped
        .strip_prefix("## Updates\n")
        .map(|updates| (String::new(), updates.trim().to_string()))
}

fn update_frontmatter_fields(
    path: &Path,
    update: impl FnOnce(&mut YamlMapping),
) -> std::io::Result<()> {
    let (frontmatter, body) = read_markdown_parts(path)?;
    let mut mapping = frontmatter.unwrap_or_default();
    update(&mut mapping);
    write_markdown_parts(path, Some(&mapping), &body)
}

fn read_markdown_parts(path: &Path) -> std::io::Result<(Option<YamlMapping>, String)> {
    let raw = std::fs::read_to_string(path)?;
    let Some(stripped) = raw.strip_prefix("---\n") else {
        return Ok((None, raw.trim().to_string()));
    };
    let Some((frontmatter_raw, body)) = stripped.split_once("\n---\n") else {
        return Ok((None, raw.trim().to_string()));
    };
    let frontmatter = serde_yaml::from_str::<YamlMapping>(frontmatter_raw).unwrap_or_default();
    Ok((Some(frontmatter), body.trim().to_string()))
}

fn write_markdown_parts(
    path: &Path,
    frontmatter: Option<&YamlMapping>,
    body: &str,
) -> std::io::Result<()> {
    let content = match frontmatter {
        Some(frontmatter) if !frontmatter.is_empty() => {
            let yaml = serde_yaml::to_string(frontmatter)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            format!("---\n{}---\n\n{}\n", yaml, body.trim())
        }
        _ => format!("{}\n", body.trim()),
    };
    std::fs::write(path, content)
}

fn write_checklist(path: &Path, checklist: &[ChecklistItem]) -> std::io::Result<()> {
    update_frontmatter_fields(path, |frontmatter| {
        let list = checklist
            .iter()
            .map(|item| {
                let mut mapping = YamlMapping::new();
                mapping.insert(
                    YamlValue::String("id".to_string()),
                    YamlValue::Number(item.id.into()),
                );
                mapping.insert(
                    YamlValue::String("text".to_string()),
                    YamlValue::String(item.text.clone()),
                );
                mapping.insert(
                    YamlValue::String("completed".to_string()),
                    YamlValue::Bool(item.completed),
                );
                if let Some(due_date) = &item.due_date {
                    mapping.insert(
                        YamlValue::String("due_date".to_string()),
                        YamlValue::String(due_date.clone()),
                    );
                }
                YamlValue::Mapping(mapping)
            })
            .collect::<Vec<_>>();
        frontmatter.insert(
            YamlValue::String("checklist".to_string()),
            YamlValue::Sequence(list),
        );
    })
}

fn normalize_checklist(value: &YamlValue) -> Vec<ChecklistItem> {
    let Some(items) = value.as_sequence() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let mapping = item.as_mapping()?;
            let id = mapping
                .get(YamlValue::String("id".to_string()))
                .and_then(YamlValue::as_i64)?;
            let text = mapping
                .get(YamlValue::String("text".to_string()))
                .and_then(YamlValue::as_str)?
                .to_string();
            let completed = mapping
                .get(YamlValue::String("completed".to_string()))
                .and_then(YamlValue::as_bool)
                .unwrap_or(false);
            let due_date = mapping
                .get(YamlValue::String("due_date".to_string()))
                .and_then(YamlValue::as_str)
                .map(ToString::to_string);
            Some(ChecklistItem {
                id,
                text,
                completed,
                due_date,
            })
        })
        .collect()
}

fn local_timestamp_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("unix-{seconds}")
}

#[cfg(test)]
mod tests {
    use super::{
        add_checklist_item, append_agenda_update, create_agenda_file, get_checklist,
        mark_agenda_item_completed, mark_agenda_item_deleted, rename_agenda_file,
        update_agenda_body, update_checklist_item,
    };

    #[test]
    fn create_update_complete_and_delete_work() {
        let unique = format!(
            "elroy-rs-agenda-crate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_agenda_file(
            &root,
            "Doctor Visit",
            "Bring forms",
            Some("2026-05-15"),
            None,
            None,
        )
        .expect("agenda file should be created");
        append_agenda_update(&path, "called ahead").expect("agenda update should append");
        mark_agenda_item_completed(&path, Some("done")).expect("agenda should complete");
        mark_agenda_item_deleted(&path, Some("no longer needed")).expect("agenda should delete");

        let content = std::fs::read_to_string(path).expect("agenda should read");
        assert!(content.contains("## Updates"));
        assert!(content.contains("called ahead"));
        assert!(content.contains("completed: true"));
        assert!(content.contains("status: deleted"));
        assert!(content.contains("closing_comment: no longer needed"));

        std::fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn checklist_items_can_be_added_and_updated() {
        let unique = format!(
            "elroy-rs-agenda-checklist-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_agenda_file(&root, "Trip", "Pack bags", Some("2026-05-15"), None, None)
            .expect("agenda file should be created");
        let item_id = add_checklist_item(&path, "passport", Some("2026-05-14"))
            .expect("checklist item should be added");
        let updated = update_checklist_item(&path, item_id, Some("passport + visa"), Some(true))
            .expect("checklist item should update");
        let checklist = get_checklist(&path).expect("checklist should load");

        assert_eq!(updated.id, item_id);
        assert!(updated.completed);
        assert_eq!(checklist.len(), 1);
        assert_eq!(checklist[0].text, "passport + visa");
        assert!(checklist[0].completed);

        std::fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn agenda_body_can_update_and_file_can_rename() {
        let unique = format!(
            "elroy-rs-agenda-rename-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_agenda_file(
            &root,
            "Doctor Visit",
            "Bring forms",
            Some("2026-05-15"),
            None,
            None,
        )
        .expect("agenda file should be created");
        update_agenda_body(&path, "Bring forms and insurance").expect("agenda body should update");
        let renamed =
            rename_agenda_file(&path, "Doctor Appointment").expect("agenda file should rename");

        assert!(renamed.ends_with("doctor_appointment.md"));
        let content = std::fs::read_to_string(renamed).expect("agenda should read");
        assert!(content.contains("Bring forms and insurance"));

        std::fs::remove_dir_all(root).expect("root should be removed");
    }
}
