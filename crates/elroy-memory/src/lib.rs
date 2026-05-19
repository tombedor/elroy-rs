pub use elroy_core::memory_store::*;

pub mod tools;

#[cfg(test)]
mod tests {
    use super::{
        archive_memory_file, create_memory_file, create_memory_file_with_frontmatter,
        read_memory_parts, sanitize_filename, update_memory_body,
    };

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

    #[test]
    fn file_create_with_frontmatter_round_trips_metadata() {
        let unique = format!(
            "elroy-rs-memory-frontmatter-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).expect("root should be created");

        let path = create_memory_file_with_frontmatter(
            &root,
            "Runner Notes",
            "Remember this",
            Some("source_type: ContextMessageSet\nmessage_ids_json: [1,2,3]"),
        )
        .expect("memory file should be created");
        let (frontmatter, body) = read_memory_parts(&path).expect("memory parts should read");

        assert_eq!(
            frontmatter.as_deref(),
            Some("source_type: ContextMessageSet\nmessage_ids_json: [1,2,3]")
        );
        assert_eq!(body, "Remember this");

        std::fs::remove_dir_all(root).expect("root should be removed");
    }
}
