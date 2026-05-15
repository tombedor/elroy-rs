use std::path::{Path, PathBuf};

use chrono::Utc;
use elroy_agenda::{
    create_agenda_file, mark_agenda_item_completed, mark_agenda_item_deleted, rename_agenda_file,
    update_agenda_body,
};
use elroy_db::{
    AgendaItemRecord, find_active_agenda_item_by_name, list_active_agenda_items,
    list_active_due_items,
};
use rusqlite::Connection;

pub fn create_task_file(agenda_dir: &Path, name: &str, text: &str) -> std::io::Result<PathBuf> {
    create_task_file_with_schedule(agenda_dir, name, text, None, None, None)
}

pub fn create_task_file_with_schedule(
    agenda_dir: &Path,
    name: &str,
    text: &str,
    item_date: Option<&str>,
    trigger_datetime: Option<&str>,
    trigger_context: Option<&str>,
) -> std::io::Result<PathBuf> {
    let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
    create_agenda_file(
        agenda_dir,
        name,
        text,
        item_date.or(Some(today.as_str())),
        trigger_datetime,
        trigger_context,
    )
}

pub fn complete_task_file(path: &Path, closing_comment: Option<&str>) -> std::io::Result<()> {
    mark_agenda_item_completed(path, closing_comment)
}

pub fn delete_task_file(path: &Path, closing_comment: Option<&str>) -> std::io::Result<()> {
    let _ = closing_comment;
    mark_agenda_item_deleted(path)
}

pub fn rename_task_file(path: &Path, new_name: &str) -> std::io::Result<PathBuf> {
    rename_agenda_file(path, new_name)
}

pub fn update_task_text_file(path: &Path, new_text: &str) -> std::io::Result<()> {
    update_agenda_body(path, new_text)
}

pub fn find_task_by_name(
    connection: &Connection,
    name: &str,
) -> rusqlite::Result<Option<AgendaItemRecord>> {
    find_active_agenda_item_by_name(connection, name)
}

pub fn list_active_tasks(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    list_active_agenda_items(connection, limit)
}

pub fn list_triggered_tasks(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    list_active_due_items(connection, limit)
}

pub fn list_due_tasks(
    connection: &Connection,
    limit: usize,
    now_iso: &str,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    let tasks = list_active_due_items(connection, limit)?;
    Ok(tasks
        .into_iter()
        .filter(|task| {
            task.trigger_datetime
                .as_deref()
                .is_some_and(|trigger_datetime| trigger_datetime <= now_iso)
        })
        .collect())
}

pub fn list_today_tasks(
    connection: &Connection,
    limit: usize,
    today_iso: &str,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    let tasks = list_active_agenda_items(connection, limit)?;
    Ok(tasks
        .into_iter()
        .filter(|task| task.agenda_date.as_deref() == Some(today_iso))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use chrono::Utc;
    use elroy_db::{
        BootstrapInventory, bootstrap_documents, persist_bootstrap_documents, run_migrations,
        sync_derived_domain_tables,
    };
    use rusqlite::Connection;

    use super::{
        complete_task_file, create_task_file, create_task_file_with_schedule, delete_task_file,
        find_task_by_name, list_active_tasks, list_due_tasks, list_today_tasks,
        list_triggered_tasks, rename_task_file, update_task_text_file,
    };

    #[test]
    fn create_update_rename_complete_and_delete_task_file_work() {
        let unique = format!(
            "elroy-rs-tasks-crate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("root should be created");

        let path =
            create_task_file(&root, "Job Search", "Reach out to three contacts").expect("task");
        update_task_text_file(&path, "Reach out to four contacts").expect("task should update");
        let renamed = rename_task_file(&path, "Career Search").expect("task should rename");
        complete_task_file(&renamed, Some("done")).expect("task should complete");
        delete_task_file(&renamed, Some("remove")).expect("task should delete");

        let content = fs::read_to_string(renamed).expect("task file should read");
        assert!(content.contains("Reach out to four contacts"));
        assert!(content.contains("status: deleted"));

        fs::remove_dir_all(root).expect("root should be removed");
    }

    #[test]
    fn task_queries_filter_active_triggered_due_and_today_tasks() {
        let unique = format!(
            "elroy-rs-tasks-query-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let agenda_dir = home.join("agenda");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
        fs::write(
            agenda_dir.join("job_search.md"),
            format!("---\ndate: {today}\ncompleted: false\nstatus: created\n---\n\nReach out\n"),
        )
        .expect("agenda fixture should be written");
        fs::write(
            agenda_dir.join("doctor_visit.md"),
            format!(
                "---\ndate: {today}\ncompleted: false\nstatus: created\ntrigger_datetime: 2000-01-01T09:00:00\n---\n\nBring forms\n"
            ),
        )
        .expect("due fixture should be written");

        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_migrations(&mut connection).expect("migrations should run");
        let inventory = BootstrapInventory {
            memory_files: vec![],
            agenda_files: vec![
                agenda_dir.join("doctor_visit.md"),
                agenda_dir.join("job_search.md"),
            ],
        };
        let documents = bootstrap_documents(&inventory).expect("documents should parse");
        persist_bootstrap_documents(&mut connection, &documents).expect("persist should succeed");
        sync_derived_domain_tables(&mut connection, &documents).expect("sync should succeed");

        let active = list_active_tasks(&connection, 10).expect("active tasks should query");
        let triggered =
            list_triggered_tasks(&connection, 10).expect("triggered tasks should query");
        let due =
            list_due_tasks(&connection, 10, "2026-01-01T00:00:00").expect("due tasks should query");
        let today_tasks =
            list_today_tasks(&connection, 10, &today).expect("today tasks should query");
        let exact = find_task_by_name(&connection, "job search").expect("task should query");

        assert_eq!(active.len(), 2);
        assert_eq!(triggered.len(), 1);
        assert_eq!(due.len(), 1);
        assert_eq!(today_tasks.len(), 2);
        assert_eq!(
            exact.as_ref().map(|task| task.name.as_str()),
            Some("job search")
        );

        fs::remove_dir_all(home).expect("home should be removed");
    }

    #[test]
    fn create_task_file_can_persist_optional_schedule_and_trigger_fields() {
        let unique = format!(
            "elroy-rs-tasks-scheduled-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        fs::create_dir_all(&root).expect("root should be created");

        let path = create_task_file_with_schedule(
            &root,
            "File Taxes",
            "Collect forms",
            Some("2026-04-15"),
            Some("2026-04-14T09:00:00"),
            Some("after payroll email"),
        )
        .expect("scheduled task should be created");

        let content = fs::read_to_string(path).expect("task file should read");
        assert!(content.contains("date: 2026-04-15"));
        assert!(content.contains("trigger_datetime: 2026-04-14T09:00:00"));
        assert!(content.contains("trigger_context: after payroll email"));

        fs::remove_dir_all(root).expect("root should be removed");
    }
}
