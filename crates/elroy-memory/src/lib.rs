use std::path::{Path, PathBuf};

pub fn sanitize_filename(name: &str) -> String {
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

pub fn create_memory_file(memory_dir: &Path, name: &str, text: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(memory_dir)?;
    let path = unique_markdown_path(memory_dir, name);
    std::fs::write(&path, format!("{text}\n"))?;
    Ok(path)
}

pub fn update_memory_body(path: &Path, text: &str) -> std::io::Result<()> {
    let (frontmatter, _) = read_markdown_parts(path)?;
    write_markdown_parts(path, frontmatter.as_deref(), text)
}

pub fn archive_memory_file(path: &Path, archive_dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(archive_dir)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::other("file path has no file name"))?;
    let mut destination = archive_dir.join(file_name);
    if destination.exists() {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("item");
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("md");
        let mut counter = 2;
        while destination.exists() {
            destination = archive_dir.join(format!("{stem}-{counter}.{extension}"));
            counter += 1;
        }
    }
    std::fs::rename(path, &destination)?;
    Ok(destination)
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

fn read_markdown_parts(path: &Path) -> std::io::Result<(Option<String>, String)> {
    let raw = std::fs::read_to_string(path)?;
    let Some(stripped) = raw.strip_prefix("---\n") else {
        return Ok((None, raw.trim().to_string()));
    };
    let Some((frontmatter_raw, body)) = stripped.split_once("\n---\n") else {
        return Ok((None, raw.trim().to_string()));
    };
    Ok((Some(frontmatter_raw.to_string()), body.trim().to_string()))
}

fn write_markdown_parts(path: &Path, frontmatter: Option<&str>, body: &str) -> std::io::Result<()> {
    let content = match frontmatter {
        Some(frontmatter) if !frontmatter.trim().is_empty() => {
            format!("---\n{}\n---\n\n{}\n", frontmatter.trim(), body.trim())
        }
        _ => format!("{}\n", body.trim()),
    };
    std::fs::write(path, content)
}

#[cfg(test)]
mod tests {
    use super::{archive_memory_file, create_memory_file, sanitize_filename, update_memory_body};

    #[test]
    fn sanitize_filename_compacts_words() {
        assert_eq!(sanitize_filename("Runner Notes"), "runner_notes");
        assert_eq!(sanitize_filename("!!!"), "item");
    }

    #[test]
    fn file_create_update_and_archive_work() {
        let unique = format!(
            "elroy-rs-memory-crate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let archive = root.join("archive");
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_memory_file(&root, "Runner Notes", "Remember this")
            .expect("memory file should be created");
        update_memory_body(&path, "Updated text").expect("memory should update");
        let archived = archive_memory_file(&path, &archive).expect("memory should archive");

        assert!(archived.exists());
        let content = std::fs::read_to_string(archived).expect("archived memory should read");
        assert!(content.contains("Updated text"));

        std::fs::remove_dir_all(root).expect("root should be removed");
    }
}
