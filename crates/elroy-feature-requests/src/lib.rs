use std::path::{Path, PathBuf};

use elroy_memory::sanitize_filename;
use serde_yaml::Value as YamlValue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureRequestRecord {
    pub path: PathBuf,
    pub request_id: String,
    pub title: String,
    pub status: String,
    pub source: String,
    pub created_at: String,
    pub updated_at: String,
    pub aliases: Vec<String>,
    pub summary: String,
    pub rationale: Option<String>,
    pub supporting_context: Option<String>,
}

pub fn feature_requests_dir(home_dir: &Path) -> PathBuf {
    home_dir.join("feature-requests")
}

pub fn slugify_feature_request_title(title: &str) -> String {
    sanitize_filename(&title.trim().to_lowercase().replace(['_', ' '], "-"))
}

pub fn write_new_feature_request(
    home_dir: &Path,
    title: &str,
    summary: &str,
    rationale: Option<&str>,
    supporting_context: Option<&str>,
    source: &str,
) -> std::io::Result<FeatureRequestRecord> {
    let dir = feature_requests_dir(home_dir);
    std::fs::create_dir_all(&dir)?;
    let path = feature_request_path(&dir, title);
    let now = iso_timestamp_now();
    let request_id = slugify_feature_request_title(title);
    let content = build_feature_request_content(&FeatureRequestContentSpec {
        request_id: &request_id,
        title,
        status: "open",
        source,
        created_at: &now,
        updated_at: &now,
        aliases: &[],
        summary,
        rationale,
        supporting_context,
    });
    std::fs::write(&path, content)?;
    load_feature_request(&path)
}

pub fn load_feature_request(path: &Path) -> std::io::Result<FeatureRequestRecord> {
    let raw = std::fs::read_to_string(path)?;
    let (frontmatter, body) = parse_frontmatter_and_body(&raw);
    let summary = section_value(&body, "Summary").unwrap_or_else(|| body.trim().to_string());
    let rationale = section_value(&body, "Why It Matters");
    let supporting_context = section_value(&body, "Supporting Context");
    let aliases = frontmatter
        .get("aliases")
        .and_then(YamlValue::as_sequence)
        .map(|items| {
            items
                .iter()
                .filter_map(yaml_value_to_string)
                .filter(|value| !value.trim().is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(FeatureRequestRecord {
        path: path.to_path_buf(),
        request_id: frontmatter
            .get("id")
            .and_then(yaml_value_to_string)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_string()
            }),
        title: frontmatter
            .get("title")
            .and_then(yaml_value_to_string)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default()
                    .to_string()
            }),
        status: frontmatter
            .get("status")
            .and_then(yaml_value_to_string)
            .unwrap_or_else(|| "open".to_string()),
        source: infer_feature_request_source(
            frontmatter
                .get("source")
                .and_then(yaml_value_to_string)
                .as_deref(),
            supporting_context.as_deref(),
        ),
        created_at: frontmatter
            .get("created_at")
            .and_then(yaml_value_to_string)
            .unwrap_or_default(),
        updated_at: frontmatter
            .get("updated_at")
            .and_then(yaml_value_to_string)
            .unwrap_or_default(),
        aliases,
        summary,
        rationale,
        supporting_context,
    })
}

pub fn update_feature_request(
    record: &FeatureRequestRecord,
    title: Option<&str>,
    status: Option<&str>,
    aliases: Option<&[String]>,
    summary: Option<&str>,
    rationale: Option<Option<&str>>,
    supporting_context: Option<Option<&str>>,
) -> std::io::Result<FeatureRequestRecord> {
    let updated_title = title.unwrap_or(&record.title);
    let updated_status = status.unwrap_or(&record.status);
    let updated_summary = summary.unwrap_or(&record.summary);
    let updated_rationale = rationale.unwrap_or(record.rationale.as_deref());
    let updated_supporting_context =
        supporting_context.unwrap_or(record.supporting_context.as_deref());
    let updated_aliases = aliases
        .map(|items| items.to_vec())
        .unwrap_or_else(|| record.aliases.clone());
    let updated_at = iso_timestamp_now();

    let content = build_feature_request_content(&FeatureRequestContentSpec {
        request_id: &record.request_id,
        title: updated_title,
        status: updated_status,
        source: &record.source,
        created_at: if record.created_at.is_empty() {
            &updated_at
        } else {
            &record.created_at
        },
        updated_at: &updated_at,
        aliases: &updated_aliases,
        summary: updated_summary,
        rationale: updated_rationale,
        supporting_context: updated_supporting_context,
    });
    std::fs::write(&record.path, content)?;
    load_feature_request(&record.path)
}

pub fn list_feature_requests(home_dir: &Path) -> std::io::Result<Vec<FeatureRequestRecord>> {
    let dir = feature_requests_dir(home_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("md")
            )
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
        .into_iter()
        .map(|path| load_feature_request(&path))
        .collect()
}

pub fn list_self_reflection_feature_requests(
    home_dir: &Path,
    active_only: bool,
) -> std::io::Result<Vec<FeatureRequestRecord>> {
    let mut records = list_feature_requests(home_dir)?
        .into_iter()
        .filter(|record| record.source == "self_reflection")
        .collect::<Vec<_>>();
    if active_only {
        records.retain(is_active_feature_request);
    }
    Ok(records)
}

pub fn get_feature_request(
    home_dir: &Path,
    identifier: &str,
) -> std::io::Result<Option<FeatureRequestRecord>> {
    let normalized_identifier = normalize(identifier);
    for record in list_feature_requests(home_dir)? {
        let mut candidates = vec![
            record.request_id.clone(),
            record.title.clone(),
            record
                .path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string(),
        ];
        candidates.extend(record.aliases.clone());
        if candidates
            .into_iter()
            .any(|candidate| normalize(&candidate) == normalized_identifier)
        {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

pub fn is_active_feature_request(record: &FeatureRequestRecord) -> bool {
    !matches!(
        record.status.trim().to_lowercase().as_str(),
        "closed" | "completed" | "done" | "cancelled" | "rejected"
    )
}

fn feature_request_path(root: &Path, title: &str) -> PathBuf {
    let base = slugify_feature_request_title(title);
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

struct FeatureRequestContentSpec<'a> {
    request_id: &'a str,
    title: &'a str,
    status: &'a str,
    source: &'a str,
    created_at: &'a str,
    updated_at: &'a str,
    aliases: &'a [String],
    summary: &'a str,
    rationale: Option<&'a str>,
    supporting_context: Option<&'a str>,
}

fn build_feature_request_content(spec: &FeatureRequestContentSpec<'_>) -> String {
    let frontmatter = serde_yaml::to_string(&serde_yaml::Mapping::from_iter([
        (
            YamlValue::String("id".to_string()),
            YamlValue::String(spec.request_id.to_string()),
        ),
        (
            YamlValue::String("title".to_string()),
            YamlValue::String(spec.title.to_string()),
        ),
        (
            YamlValue::String("status".to_string()),
            YamlValue::String(spec.status.to_string()),
        ),
        (
            YamlValue::String("source".to_string()),
            YamlValue::String(spec.source.to_string()),
        ),
        (
            YamlValue::String("created_at".to_string()),
            YamlValue::String(spec.created_at.to_string()),
        ),
        (
            YamlValue::String("updated_at".to_string()),
            YamlValue::String(spec.updated_at.to_string()),
        ),
        (
            YamlValue::String("aliases".to_string()),
            YamlValue::Sequence(
                spec.aliases
                    .iter()
                    .cloned()
                    .map(YamlValue::String)
                    .collect::<Vec<_>>(),
            ),
        ),
    ]))
    .expect("feature request frontmatter should serialize");

    let mut body = vec!["## Summary".to_string(), spec.summary.trim().to_string()];
    if let Some(rationale) = spec.rationale.filter(|value| !value.trim().is_empty()) {
        body.push(String::new());
        body.push("## Why It Matters".to_string());
        body.push(rationale.trim().to_string());
    }
    if let Some(supporting_context) = spec
        .supporting_context
        .filter(|value| !value.trim().is_empty())
    {
        body.push(String::new());
        body.push("## Supporting Context".to_string());
        body.push(supporting_context.trim().to_string());
    }

    format!("---\n{}---\n\n{}\n", frontmatter, body.join("\n"))
}

fn parse_frontmatter_and_body(raw: &str) -> (serde_yaml::Mapping, String) {
    let Some(stripped) = raw.strip_prefix("---\n") else {
        return (serde_yaml::Mapping::new(), raw.trim().to_string());
    };
    let Some((frontmatter_raw, body)) = stripped.split_once("\n---\n") else {
        return (serde_yaml::Mapping::new(), raw.trim().to_string());
    };
    let frontmatter = serde_yaml::from_str::<YamlValue>(frontmatter_raw)
        .ok()
        .and_then(|value| value.as_mapping().cloned())
        .unwrap_or_default();
    (frontmatter, body.trim().to_string())
}

fn section_value(body: &str, section_name: &str) -> Option<String> {
    let mut sections = std::collections::BTreeMap::<String, Vec<String>>::new();
    let mut current_section = None::<String>;
    for line in body.lines() {
        if let Some(section) = line.strip_prefix("## ") {
            current_section = Some(section.trim().to_string());
            sections.entry(section.trim().to_string()).or_default();
            continue;
        }
        if let Some(section) = &current_section {
            sections
                .entry(section.clone())
                .or_default()
                .push(line.to_string());
        }
    }
    let value = sections.remove(section_name)?;
    let cleaned = value.join("\n").trim().to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

fn infer_feature_request_source(source: Option<&str>, supporting_context: Option<&str>) -> String {
    if let Some(source) = source {
        let normalized = source.trim().to_lowercase();
        if !normalized.is_empty() {
            return normalized;
        }
    }

    let context = supporting_context.unwrap_or_default();
    let reflection_markers = [
        "- Reflected at:",
        "- Trigger phrase:",
        "- Recent user feedback:",
    ];
    if reflection_markers
        .iter()
        .all(|marker| context.contains(marker))
    {
        "self_reflection".to_string()
    } else {
        "user_request".to_string()
    }
}

fn yaml_value_to_string(value: &YamlValue) -> Option<String> {
    match value {
        YamlValue::String(value) => Some(value.clone()),
        YamlValue::Number(value) => Some(value.to_string()),
        YamlValue::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn normalize(text: &str) -> String {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens.join(" ")
}

fn iso_timestamp_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| format!("{}", duration.as_secs()))
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        get_feature_request, is_active_feature_request, list_feature_requests,
        list_self_reflection_feature_requests, load_feature_request, update_feature_request,
        write_new_feature_request,
    };

    #[test]
    fn feature_requests_can_be_created_listed_and_closed() {
        let unique = format!(
            "elroy-rs-feature-requests-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&home).expect("home dir should be created");

        let improvement = write_new_feature_request(
            &home,
            "Improve correction handling",
            "Recover more directly after user corrections.",
            Some("Reflection found a correction handling gap."),
            Some(
                "- Reflected at: 2026-05-12T00:00:00+00:00\n- Trigger phrase: correction\n- Recent user feedback: please fix corrections",
            ),
            "self_reflection",
        )
        .expect("feature request should be created");
        write_new_feature_request(
            &home,
            "General export feature",
            "Export notes to markdown.",
            None,
            None,
            "user_request",
        )
        .expect("feature request should be created");

        let improvement_exact =
            get_feature_request(&home, &improvement.request_id).expect("query should succeed");
        let listed = list_feature_requests(&home).expect("list should succeed");
        let self_reflection =
            list_self_reflection_feature_requests(&home, true).expect("list should succeed");

        assert_eq!(listed.len(), 2);
        assert_eq!(self_reflection.len(), 1);
        assert_eq!(
            improvement_exact
                .as_ref()
                .map(|record| record.title.as_str()),
            Some("Improve correction handling")
        );

        let closed =
            update_feature_request(&improvement, None, Some("closed"), None, None, None, None)
                .expect("feature request should update");
        assert!(!is_active_feature_request(&closed));

        let reloaded = load_feature_request(&closed.path).expect("feature request should reload");
        assert_eq!(reloaded.status, "closed");

        std::fs::remove_dir_all(home).expect("home dir should be removed");
    }
}
