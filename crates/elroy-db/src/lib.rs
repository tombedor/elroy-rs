use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use elroy_config::AppConfig;
use elroy_llm::{ConversationMessage, MessageRole, ToolCall};
use rusqlite::Connection;
use rusqlite::params;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

mod embedded {
    use refinery::embed_migrations;

    embed_migrations!("migrations");
}

const MEMORY_KIND: &str = "memory";
const AGENDA_KIND: &str = "agenda";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapPlan {
    pub memory_dir: PathBuf,
    pub agenda_dir: PathBuf,
    pub database_path: PathBuf,
}

impl BootstrapPlan {
    pub fn from_config(config: &AppConfig) -> Self {
        Self {
            memory_dir: config.memory_dir.clone(),
            agenda_dir: config.agenda_dir.clone(),
            database_path: config.database_path.clone(),
        }
    }

    pub fn can_rebuild_from_files(&self) -> bool {
        !self.memory_dir.as_os_str().is_empty() && !self.database_path.as_os_str().is_empty()
    }

    pub fn persistence_stack(&self) -> PersistenceStack {
        PersistenceStack::sqlite_first()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistenceStack {
    pub driver: &'static str,
    pub migration_tool: &'static str,
    pub async_boundary_adapter: Option<&'static str>,
}

impl PersistenceStack {
    pub fn sqlite_first() -> Self {
        Self {
            driver: "rusqlite",
            migration_tool: "refinery",
            async_boundary_adapter: Some("tokio-rusqlite"),
        }
    }
}

pub fn open_sqlite_connection(path: &Path) -> rusqlite::Result<Connection> {
    Connection::open(path)
}

pub fn run_migrations(connection: &mut Connection) -> Result<(), refinery::Error> {
    embedded::migrations::runner().run(connection)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapInventory {
    pub memory_files: Vec<PathBuf>,
    pub agenda_files: Vec<PathBuf>,
}

impl BootstrapInventory {
    pub fn discover(plan: &BootstrapPlan) -> Self {
        Self {
            memory_files: markdown_files(&plan.memory_dir),
            agenda_files: markdown_files(&plan.agenda_dir),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frontmatter {
    pub id: Option<i64>,
    pub agenda_date: Option<String>,
    pub completed: bool,
    pub status: Option<String>,
    pub trigger_datetime: Option<String>,
    pub trigger_context: Option<String>,
    pub closing_comment: Option<String>,
    pub checklist_total: i64,
    pub checklist_completed: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapDocument {
    pub kind: String,
    pub path: PathBuf,
    pub stem: String,
    pub frontmatter_id: Option<i64>,
    pub agenda_date: Option<String>,
    pub completed: bool,
    pub status: Option<String>,
    pub trigger_datetime: Option<String>,
    pub trigger_context: Option<String>,
    pub closing_comment: Option<String>,
    pub checklist_total: i64,
    pub checklist_completed: i64,
    pub body: String,
    pub updated_at_unix: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapResult {
    pub memory_files: usize,
    pub agenda_files: usize,
    pub persisted_documents: usize,
    pub synced_memories: usize,
    pub synced_agenda_items: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedCounts {
    pub memories: usize,
    pub agenda_items: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: i64,
    pub legacy_frontmatter_id: Option<i64>,
    pub name: String,
    pub file_path: String,
    pub body: String,
    pub is_active: bool,
    pub updated_at_unix: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgendaItemRecord {
    pub id: i64,
    pub legacy_frontmatter_id: Option<i64>,
    pub name: String,
    pub file_path: String,
    pub agenda_date: Option<String>,
    pub is_completed: bool,
    pub status: Option<String>,
    pub trigger_datetime: Option<String>,
    pub trigger_context: Option<String>,
    pub closing_comment: Option<String>,
    pub checklist_total: i64,
    pub checklist_completed: i64,
    pub body: String,
    pub is_active: bool,
    pub updated_at_unix: i64,
}

#[derive(Debug)]
pub enum BootstrapError {
    Io(std::io::Error),
    Sql(rusqlite::Error),
    Migration(refinery::Error),
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "bootstrap IO error: {error}"),
            Self::Sql(error) => write!(f, "bootstrap SQLite error: {error}"),
            Self::Migration(error) => write!(f, "bootstrap migration error: {error}"),
        }
    }
}

impl std::error::Error for BootstrapError {}

impl From<std::io::Error> for BootstrapError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<rusqlite::Error> for BootstrapError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sql(value)
    }
}

impl From<refinery::Error> for BootstrapError {
    fn from(value: refinery::Error) -> Self {
        Self::Migration(value)
    }
}

pub fn bootstrap_database(plan: &BootstrapPlan) -> Result<BootstrapResult, BootstrapError> {
    let inventory = BootstrapInventory::discover(plan);
    let documents = bootstrap_documents(&inventory)?;

    if let Some(parent) = plan.database_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut connection = open_sqlite_connection(&plan.database_path)?;
    run_migrations(&mut connection)?;
    persist_bootstrap_documents(&mut connection, &documents)?;
    sync_derived_domain_tables(&mut connection, &documents)?;
    let derived_counts = derived_counts(&connection)?;

    Ok(BootstrapResult {
        memory_files: inventory.memory_files.len(),
        agenda_files: inventory.agenda_files.len(),
        persisted_documents: documents.len(),
        synced_memories: derived_counts.memories,
        synced_agenda_items: derived_counts.agenda_items,
    })
}

pub fn bootstrap_documents(
    inventory: &BootstrapInventory,
) -> std::io::Result<Vec<BootstrapDocument>> {
    let mut documents = Vec::new();

    for path in &inventory.memory_files {
        documents.push(build_document(MEMORY_KIND, path)?);
    }
    for path in &inventory.agenda_files {
        documents.push(build_document(AGENDA_KIND, path)?);
    }

    documents.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(documents)
}

pub fn persist_bootstrap_documents(
    connection: &mut Connection,
    documents: &[BootstrapDocument],
) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute("DELETE FROM bootstrap_documents", [])?;

    for document in documents {
        transaction.execute(
            "INSERT INTO bootstrap_documents (
                kind,
                path,
                stem,
                frontmatter_id,
                agenda_date,
                is_completed,
                status,
                trigger_datetime,
                trigger_context,
                closing_comment,
                checklist_total,
                checklist_completed,
                body,
                updated_at_unix
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                document.kind,
                document.path.display().to_string(),
                document.stem,
                document.frontmatter_id,
                document.agenda_date,
                if document.completed { 1_i64 } else { 0_i64 },
                document.status,
                document.trigger_datetime,
                document.trigger_context,
                document.closing_comment,
                document.checklist_total,
                document.checklist_completed,
                document.body,
                document.updated_at_unix,
            ],
        )?;
    }

    transaction.commit()
}

pub fn sync_derived_domain_tables(
    connection: &mut Connection,
    documents: &[BootstrapDocument],
) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute("DELETE FROM memories", [])?;
    transaction.execute("DELETE FROM agenda_items", [])?;

    for document in documents {
        let path = document.path.display().to_string();
        let bootstrap_document_id: i64 = transaction.query_row(
            "SELECT id FROM bootstrap_documents WHERE path = ?1",
            [&path],
            |row| row.get(0),
        )?;

        if document.kind == MEMORY_KIND {
            transaction.execute(
                "INSERT INTO memories (
                    bootstrap_document_id,
                    legacy_frontmatter_id,
                    name,
                    file_path,
                    body,
                    is_active,
                    updated_at_unix
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    bootstrap_document_id,
                    document.frontmatter_id,
                    document.stem.replace('_', " "),
                    path,
                    document.body,
                    1_i64,
                    document.updated_at_unix,
                ],
            )?;
            continue;
        }

        if document.kind == AGENDA_KIND {
            let status = document.status.clone().unwrap_or_else(|| {
                if document.completed {
                    "completed".to_string()
                } else {
                    "created".to_string()
                }
            });
            let is_active = if document.completed || status == "deleted" {
                0_i64
            } else {
                1_i64
            };

            transaction.execute(
                "INSERT INTO agenda_items (
                    bootstrap_document_id,
                    legacy_frontmatter_id,
                    name,
                    file_path,
                    agenda_date,
                    is_completed,
                    status,
                    trigger_datetime,
                    trigger_context,
                    closing_comment,
                    checklist_total,
                    checklist_completed,
                    body,
                    is_active,
                    updated_at_unix
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
                params![
                    bootstrap_document_id,
                    document.frontmatter_id,
                    document.stem.replace('_', " "),
                    path,
                    document.agenda_date,
                    if document.completed { 1_i64 } else { 0_i64 },
                    status,
                    document.trigger_datetime,
                    document.trigger_context,
                    document.closing_comment,
                    document.checklist_total,
                    document.checklist_completed,
                    document.body,
                    is_active,
                    document.updated_at_unix,
                ],
            )?;
        }
    }

    transaction.commit()
}

pub fn derived_counts(connection: &Connection) -> rusqlite::Result<DerivedCounts> {
    let memories = connection.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
    let agenda_items =
        connection.query_row("SELECT COUNT(*) FROM agenda_items", [], |row| row.get(0))?;

    Ok(DerivedCounts {
        memories,
        agenda_items,
    })
}

pub fn list_active_memories(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<MemoryRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            body,
            is_active,
            updated_at_unix
        FROM memories
        WHERE is_active = 1
        ORDER BY updated_at_unix DESC, name ASC
        LIMIT ?1",
    )?;
    let rows = statement.query_map([limit as i64], |row| {
        Ok(MemoryRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            body: row.get(4)?,
            is_active: row.get::<_, i64>(5)? != 0,
            updated_at_unix: row.get(6)?,
        })
    })?;
    rows.collect()
}

pub fn search_active_memories(
    connection: &Connection,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<MemoryRecord>> {
    let needle = format!("%{}%", query.to_ascii_lowercase());
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            body,
            is_active,
            updated_at_unix
        FROM memories
        WHERE is_active = 1
          AND (
            LOWER(name) LIKE ?1
            OR LOWER(body) LIKE ?1
          )
        ORDER BY updated_at_unix DESC, name ASC
        LIMIT ?2",
    )?;
    let rows = statement.query_map(params![needle, limit as i64], |row| {
        Ok(MemoryRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            body: row.get(4)?,
            is_active: row.get::<_, i64>(5)? != 0,
            updated_at_unix: row.get(6)?,
        })
    })?;
    rows.collect()
}

pub fn find_active_memory_by_name(
    connection: &Connection,
    name: &str,
) -> rusqlite::Result<Option<MemoryRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            body,
            is_active,
            updated_at_unix
        FROM memories
        WHERE is_active = 1
          AND LOWER(name) = LOWER(?1)
        ORDER BY updated_at_unix DESC
        LIMIT 1",
    )?;
    let result = statement.query_row([name], |row| {
        Ok(MemoryRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            body: row.get(4)?,
            is_active: row.get::<_, i64>(5)? != 0,
            updated_at_unix: row.get(6)?,
        })
    });
    match result {
        Ok(record) => Ok(Some(record)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn list_active_agenda_items(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            agenda_date,
            is_completed,
            status,
            trigger_datetime,
            trigger_context,
            closing_comment,
            checklist_total,
            checklist_completed,
            body,
            is_active,
            updated_at_unix
        FROM agenda_items
        WHERE is_active = 1
        ORDER BY
            CASE WHEN trigger_datetime IS NULL THEN 1 ELSE 0 END,
            trigger_datetime ASC,
            agenda_date ASC,
            name ASC
        LIMIT ?1",
    )?;
    let rows = statement.query_map([limit as i64], |row| {
        Ok(AgendaItemRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            agenda_date: row.get(4)?,
            is_completed: row.get::<_, i64>(5)? != 0,
            status: row.get(6)?,
            trigger_datetime: row.get(7)?,
            trigger_context: row.get(8)?,
            closing_comment: row.get(9)?,
            checklist_total: row.get(10)?,
            checklist_completed: row.get(11)?,
            body: row.get(12)?,
            is_active: row.get::<_, i64>(13)? != 0,
            updated_at_unix: row.get(14)?,
        })
    })?;
    rows.collect()
}

pub fn list_active_plain_agenda_items(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            agenda_date,
            is_completed,
            status,
            trigger_datetime,
            trigger_context,
            closing_comment,
            checklist_total,
            checklist_completed,
            body,
            is_active,
            updated_at_unix
        FROM agenda_items
        WHERE is_active = 1
          AND trigger_datetime IS NULL
          AND trigger_context IS NULL
        ORDER BY
            agenda_date ASC,
            name ASC
        LIMIT ?1",
    )?;
    let rows = statement.query_map([limit as i64], |row| {
        Ok(AgendaItemRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            agenda_date: row.get(4)?,
            is_completed: row.get::<_, i64>(5)? != 0,
            status: row.get(6)?,
            trigger_datetime: row.get(7)?,
            trigger_context: row.get(8)?,
            closing_comment: row.get(9)?,
            checklist_total: row.get(10)?,
            checklist_completed: row.get(11)?,
            body: row.get(12)?,
            is_active: row.get::<_, i64>(13)? != 0,
            updated_at_unix: row.get(14)?,
        })
    })?;
    rows.collect()
}

pub fn list_active_due_items(
    connection: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<AgendaItemRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            agenda_date,
            is_completed,
            status,
            trigger_datetime,
            trigger_context,
            closing_comment,
            checklist_total,
            checklist_completed,
            body,
            is_active,
            updated_at_unix
        FROM agenda_items
        WHERE is_active = 1
          AND (
            trigger_datetime IS NOT NULL
            OR trigger_context IS NOT NULL
          )
        ORDER BY
            CASE WHEN trigger_datetime IS NULL THEN 1 ELSE 0 END,
            trigger_datetime ASC,
            agenda_date ASC,
            name ASC
        LIMIT ?1",
    )?;
    let rows = statement.query_map([limit as i64], |row| {
        Ok(AgendaItemRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            agenda_date: row.get(4)?,
            is_completed: row.get::<_, i64>(5)? != 0,
            status: row.get(6)?,
            trigger_datetime: row.get(7)?,
            trigger_context: row.get(8)?,
            closing_comment: row.get(9)?,
            checklist_total: row.get(10)?,
            checklist_completed: row.get(11)?,
            body: row.get(12)?,
            is_active: row.get::<_, i64>(13)? != 0,
            updated_at_unix: row.get(14)?,
        })
    })?;
    rows.collect()
}

pub fn find_active_agenda_item_by_name(
    connection: &Connection,
    name: &str,
) -> rusqlite::Result<Option<AgendaItemRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            id,
            legacy_frontmatter_id,
            name,
            file_path,
            agenda_date,
            is_completed,
            status,
            trigger_datetime,
            trigger_context,
            closing_comment,
            checklist_total,
            checklist_completed,
            body,
            is_active,
            updated_at_unix
        FROM agenda_items
        WHERE is_active = 1
          AND LOWER(name) = LOWER(?1)
        ORDER BY
            CASE WHEN trigger_datetime IS NULL THEN 1 ELSE 0 END,
            trigger_datetime ASC,
            agenda_date ASC,
            updated_at_unix DESC
        LIMIT 1",
    )?;
    let result = statement.query_row([name], |row| {
        Ok(AgendaItemRecord {
            id: row.get(0)?,
            legacy_frontmatter_id: row.get(1)?,
            name: row.get(2)?,
            file_path: row.get(3)?,
            agenda_date: row.get(4)?,
            is_completed: row.get::<_, i64>(5)? != 0,
            status: row.get(6)?,
            trigger_datetime: row.get(7)?,
            trigger_context: row.get(8)?,
            closing_comment: row.get(9)?,
            checklist_total: row.get(10)?,
            checklist_completed: row.get(11)?,
            body: row.get(12)?,
            is_active: row.get::<_, i64>(13)? != 0,
            updated_at_unix: row.get(14)?,
        })
    });
    match result {
        Ok(record) => Ok(Some(record)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn load_context_messages(
    connection: &mut Connection,
    user_token: &str,
) -> rusqlite::Result<Vec<ConversationMessage>> {
    let set_id = get_or_create_context_message_set(connection, user_token)?;
    let mut statement = connection.prepare(
        "SELECT
            id,
            role,
            content,
            chat_model,
            created_at_unix,
            tool_calls_json,
            tool_call_id
        FROM context_messages
        WHERE set_id = ?1
        ORDER BY position ASC",
    )?;
    let rows = statement.query_map([set_id], |row| {
        let role = parse_message_role(&row.get::<_, String>(1)?);
        let tool_calls_json: Option<String> = row.get(5)?;
        let tool_calls = parse_tool_calls_json(tool_calls_json.as_deref());
        Ok(ConversationMessage {
            role,
            content: row.get(2)?,
            chat_model: row.get(3)?,
            id: row.get(0)?,
            created_at_unix: row.get(4)?,
            tool_calls,
            tool_call_id: row.get(6)?,
        })
    })?;

    rows.collect()
}

pub fn replace_context_messages(
    connection: &mut Connection,
    user_token: &str,
    messages: &[ConversationMessage],
) -> rusqlite::Result<()> {
    let set_id = get_or_create_context_message_set(connection, user_token)?;
    let updated_at = unix_timestamp_now();
    let transaction = connection.transaction()?;
    transaction.execute("DELETE FROM context_messages WHERE set_id = ?1", [set_id])?;

    for (position, message) in messages.iter().enumerate() {
        let tool_calls_json = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(json_to_sql_error)?;
        transaction.execute(
            "INSERT INTO context_messages (
                set_id,
                position,
                role,
                content,
                chat_model,
                created_at_unix,
                tool_calls_json,
                tool_call_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                set_id,
                position as i64,
                message_role_name(&message.role),
                message.content,
                message.chat_model,
                message.created_at_unix,
                tool_calls_json,
                message.tool_call_id,
            ],
        )?;
    }

    transaction.execute(
        "UPDATE context_message_sets SET updated_at_unix = ?2 WHERE id = ?1",
        params![set_id, updated_at],
    )?;
    transaction.commit()
}

fn get_or_create_context_message_set(
    connection: &mut Connection,
    user_token: &str,
) -> rusqlite::Result<i64> {
    let existing = connection.query_row(
        "SELECT id FROM context_message_sets WHERE user_token = ?1 AND is_active = 1",
        [user_token],
        |row| row.get(0),
    );
    match existing {
        Ok(id) => Ok(id),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            let now = unix_timestamp_now();
            connection.execute(
                "INSERT INTO context_message_sets (
                    user_token,
                    is_active,
                    created_at_unix,
                    updated_at_unix
                ) VALUES (?1, 1, ?2, ?2)",
                params![user_token, now],
            )?;
            Ok(connection.last_insert_rowid())
        }
        Err(error) => Err(error),
    }
}

fn parse_message_role(value: &str) -> MessageRole {
    match value {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

fn message_role_name(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn parse_tool_calls_json(value: Option<&str>) -> Option<Vec<ToolCall>> {
    let raw = value?;
    let parsed = serde_json::from_str::<Vec<JsonValue>>(raw).ok()?;
    let tool_calls = parsed
        .into_iter()
        .filter_map(|item| serde_json::from_value::<ToolCall>(item).ok())
        .collect::<Vec<_>>();
    if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    }
}

fn json_to_sql_error(error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(error))
}

fn unix_timestamp_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

pub fn markdown_files(root: &Path) -> Vec<PathBuf> {
    if !root.exists() {
        return Vec::new();
    }

    let mut files = Vec::new();
    visit_memory_files(root, &mut files);
    files.sort();
    files
}

fn visit_memory_files(root: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_memory_files(&path, files);
            continue;
        }

        if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("md" | "markdown")
        ) {
            files.push(path);
        }
    }
}

fn build_document(kind: &str, path: &Path) -> std::io::Result<BootstrapDocument> {
    let raw = std::fs::read_to_string(path)?;
    let (frontmatter, body) = parse_frontmatter_and_body(&raw);
    let metadata = std::fs::metadata(path)?;
    let updated_at_unix = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);

    Ok(BootstrapDocument {
        kind: kind.to_string(),
        path: path.to_path_buf(),
        stem: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_string(),
        frontmatter_id: frontmatter.id,
        agenda_date: frontmatter.agenda_date,
        completed: frontmatter.completed,
        status: frontmatter.status,
        trigger_datetime: frontmatter.trigger_datetime,
        trigger_context: frontmatter.trigger_context,
        closing_comment: frontmatter.closing_comment,
        checklist_total: frontmatter.checklist_total,
        checklist_completed: frontmatter.checklist_completed,
        body,
        updated_at_unix,
    })
}

fn parse_frontmatter_and_body(raw: &str) -> (Frontmatter, String) {
    let Some(stripped) = raw.strip_prefix("---\n") else {
        return (
            Frontmatter {
                id: None,
                agenda_date: None,
                completed: false,
                status: None,
                trigger_datetime: None,
                trigger_context: None,
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
            },
            raw.trim().to_string(),
        );
    };

    let Some((frontmatter_raw, body)) = stripped.split_once("\n---\n") else {
        return (
            Frontmatter {
                id: None,
                agenda_date: None,
                completed: false,
                status: None,
                trigger_datetime: None,
                trigger_context: None,
                closing_comment: None,
                checklist_total: 0,
                checklist_completed: 0,
            },
            raw.trim().to_string(),
        );
    };

    let yaml = serde_yaml::from_str::<YamlValue>(frontmatter_raw).ok();
    let id = yaml
        .as_ref()
        .and_then(|value| value.get("id"))
        .and_then(YamlValue::as_i64);
    let agenda_date = yaml
        .as_ref()
        .and_then(|value| value.get("date"))
        .and_then(yaml_value_to_string);
    let completed = yaml
        .as_ref()
        .and_then(|value| value.get("completed"))
        .and_then(YamlValue::as_bool)
        .unwrap_or(false);
    let status = yaml
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(yaml_value_to_string);
    let trigger_datetime = yaml
        .as_ref()
        .and_then(|value| value.get("trigger_datetime"))
        .and_then(yaml_value_to_string);
    let trigger_context = yaml
        .as_ref()
        .and_then(|value| {
            value
                .get("trigger_context")
                .or_else(|| value.get("reminder_context"))
        })
        .and_then(yaml_value_to_string);
    let closing_comment = yaml
        .as_ref()
        .and_then(|value| value.get("closing_comment"))
        .and_then(yaml_value_to_string);
    let (checklist_total, checklist_completed) = yaml
        .as_ref()
        .and_then(|value| value.get("checklist"))
        .map(checklist_counts)
        .unwrap_or((0, 0));

    (
        Frontmatter {
            id,
            agenda_date,
            completed,
            status,
            trigger_datetime,
            trigger_context,
            closing_comment,
            checklist_total,
            checklist_completed,
        },
        body.trim().to_string(),
    )
}

fn checklist_counts(value: &YamlValue) -> (i64, i64) {
    let Some(items) = value.as_sequence() else {
        return (0, 0);
    };
    let total = items.len() as i64;
    let completed = items
        .iter()
        .filter(|item| {
            item.as_mapping()
                .and_then(|mapping| mapping.get(YamlValue::String("completed".to_string())))
                .and_then(YamlValue::as_bool)
                .unwrap_or(false)
        })
        .count() as i64;
    (total, completed)
}

fn yaml_value_to_string(value: &YamlValue) -> Option<String> {
    value
        .as_str()
        .map(ToString::to_string)
        .or_else(|| value.as_i64().map(|v| v.to_string()))
        .or_else(|| value.as_bool().map(|v| v.to_string()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        BootstrapInventory, BootstrapPlan, bootstrap_database, bootstrap_documents, derived_counts,
        find_active_agenda_item_by_name, find_active_memory_by_name, list_active_agenda_items,
        list_active_due_items, list_active_memories, list_active_plain_agenda_items,
        load_context_messages, markdown_files, open_sqlite_connection, persist_bootstrap_documents,
        replace_context_messages, run_migrations, search_active_memories,
        sync_derived_domain_tables,
    };
    use elroy_config::AppConfig;
    use elroy_llm::{ConversationMessage, MessageRole, ToolCall};
    use rusqlite::Connection;

    #[test]
    fn bootstrap_plan_uses_config_paths() {
        let config = AppConfig::defaults();
        let plan = BootstrapPlan::from_config(&config);

        assert_eq!(plan.memory_dir, config.memory_dir);
        assert_eq!(plan.agenda_dir, config.agenda_dir);
        assert_eq!(plan.database_path, config.database_path);
        assert!(plan.can_rebuild_from_files());
        assert_eq!(plan.persistence_stack().driver, "rusqlite");
        assert_eq!(plan.persistence_stack().migration_tool, "refinery");
    }

    #[test]
    fn markdown_files_returns_markdown_files_recursively() {
        let unique = format!(
            "elroy-rs-memory-files-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let nested = root.join("nested");

        fs::create_dir_all(&nested).expect("temp directories should be created");
        fs::write(root.join("a.md"), "# a").expect("markdown fixture should be written");
        fs::write(nested.join("b.markdown"), "# b").expect("markdown fixture should be written");
        fs::write(root.join("skip.txt"), "x").expect("non-markdown fixture should be written");

        let files = markdown_files(&root);

        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|path| path.ends_with("a.md")));
        assert!(files.iter().any(|path| path.ends_with("b.markdown")));

        fs::remove_dir_all(root).expect("temp directories should be removed");
    }

    #[test]
    fn bootstrap_inventory_discovers_memory_and_agenda_files() {
        let unique = format!(
            "elroy-rs-bootstrap-inventory-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");

        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::write(memory_dir.join("memory.md"), "# memory")
            .expect("memory fixture should be written");
        fs::write(agenda_dir.join("agenda.md"), "# agenda")
            .expect("agenda fixture should be written");

        let mut config = AppConfig::defaults();
        config.memory_dir = memory_dir.clone();
        config.agenda_dir = agenda_dir.clone();
        config.database_path = home.join("elroy.db");
        let plan = BootstrapPlan::from_config(&config);
        let inventory = BootstrapInventory::discover(&plan);

        assert_eq!(inventory.memory_files, vec![memory_dir.join("memory.md")]);
        assert_eq!(inventory.agenda_files, vec![agenda_dir.join("agenda.md")]);

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }

    #[test]
    fn sqlite_connection_can_be_opened_for_bootstrap_target() {
        let unique = format!(
            "elroy-rs-db-{}.sqlite",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);

        let connection = open_sqlite_connection(&path).expect("sqlite db should open");
        connection
            .execute("CREATE TABLE bootstrap_probe (id INTEGER PRIMARY KEY)", [])
            .expect("sqlite should accept schema statements");
        drop(connection);

        fs::remove_file(path).expect("temp sqlite file should be removed");
    }

    #[test]
    fn migrations_create_bootstrap_documents_table() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");

        run_migrations(&mut connection).expect("migrations should run");

        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='bootstrap_documents'",
                [],
                |row| row.get(0),
            )
            .expect("table existence query should succeed");
        assert_eq!(count, 1);
        let memory_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )
            .expect("memories table existence query should succeed");
        let agenda_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='agenda_items'",
                [],
                |row| row.get(0),
            )
            .expect("agenda table existence query should succeed");
        assert_eq!(memory_count, 1);
        assert_eq!(agenda_count, 1);
    }

    #[test]
    fn bootstrap_documents_parse_memory_and_agenda_frontmatter() {
        let unique = format!(
            "elroy-rs-bootstrap-docs-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");

        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::write(
            memory_dir.join("memory.md"),
            "---\nid: 7\n---\n\nremember this",
        )
        .expect("memory fixture should be written");
        fs::write(
            agenda_dir.join("agenda.md"),
            "---\ndate: 2026-05-13\ncompleted: true\nstatus: completed\ntrigger_datetime: 2026-05-14T09:00:00\ntrigger_context: on standup\nclosing_comment: done\n---\n\nship it",
        )
        .expect("agenda fixture should be written");

        let inventory = BootstrapInventory {
            memory_files: vec![memory_dir.join("memory.md")],
            agenda_files: vec![agenda_dir.join("agenda.md")],
        };
        let documents = bootstrap_documents(&inventory).expect("documents should parse");

        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].kind, "agenda");
        assert_eq!(documents[0].agenda_date.as_deref(), Some("2026-05-13"));
        assert!(documents[0].completed);
        assert_eq!(
            documents[0].trigger_datetime.as_deref(),
            Some("2026-05-14T09:00:00")
        );
        assert_eq!(documents[0].trigger_context.as_deref(), Some("on standup"));
        assert_eq!(documents[0].closing_comment.as_deref(), Some("done"));
        assert_eq!(documents[1].kind, "memory");
        assert_eq!(documents[1].frontmatter_id, Some(7));
        assert_eq!(documents[1].body, "remember this");

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }

    #[test]
    fn bootstrap_documents_can_be_persisted_after_migration() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_migrations(&mut connection).expect("migrations should run");

        let unique = format!(
            "elroy-rs-bootstrap-persist-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::write(
            memory_dir.join("memory.md"),
            "---\nid: 9\n---\n\npersist me",
        )
        .expect("memory fixture should be written");

        let inventory = BootstrapInventory {
            memory_files: vec![memory_dir.join("memory.md")],
            agenda_files: Vec::new(),
        };
        let documents = bootstrap_documents(&inventory).expect("documents should parse");
        persist_bootstrap_documents(&mut connection, &documents)
            .expect("bootstrap documents should persist");

        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM bootstrap_documents", [], |row| {
                row.get(0)
            })
            .expect("document count query should succeed");
        assert_eq!(count, 1);

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }

    #[test]
    fn derived_domain_tables_can_be_rebuilt_from_bootstrap_documents() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_migrations(&mut connection).expect("migrations should run");

        let unique = format!(
            "elroy-rs-derived-sync-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::write(
            memory_dir.join("runner_notes.md"),
            "---\nid: 9\n---\n\ntrain for the race",
        )
        .expect("memory fixture should be written");
        fs::write(
            agenda_dir.join("doctor_visit.md"),
            "---\nid: 5\ndate: 2026-05-15\ncompleted: false\ntrigger_datetime: 2026-05-15T15:00:00\nreminder_context: after lunch\nstatus: created\n---\n\nbring forms",
        )
        .expect("agenda fixture should be written");

        let inventory = BootstrapInventory {
            memory_files: vec![memory_dir.join("runner_notes.md")],
            agenda_files: vec![agenda_dir.join("doctor_visit.md")],
        };
        let documents = bootstrap_documents(&inventory).expect("documents should parse");
        persist_bootstrap_documents(&mut connection, &documents)
            .expect("bootstrap documents should persist");
        sync_derived_domain_tables(&mut connection, &documents)
            .expect("derived tables should sync");

        let counts = derived_counts(&connection).expect("counts should query");
        assert_eq!(counts.memories, 1);
        assert_eq!(counts.agenda_items, 1);

        let memory_name: String = connection
            .query_row("SELECT name FROM memories", [], |row| row.get(0))
            .expect("memory query should succeed");
        let trigger_context: String = connection
            .query_row("SELECT trigger_context FROM agenda_items", [], |row| {
                row.get(0)
            })
            .expect("agenda query should succeed");
        assert_eq!(memory_name, "runner notes");
        assert_eq!(trigger_context, "after lunch");

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }

    #[test]
    fn bootstrap_database_runs_end_to_end() {
        let unique = format!(
            "elroy-rs-bootstrap-end-to-end-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        let database_path = home.join("state").join("elroy.db");

        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::write(memory_dir.join("memory.md"), "---\nid: 3\n---\n\nremember")
            .expect("memory fixture should be written");
        fs::write(
            agenda_dir.join("agenda.md"),
            "---\ndate: 2026-05-13\ncompleted: false\n---\n\nagenda",
        )
        .expect("agenda fixture should be written");

        let plan = BootstrapPlan {
            memory_dir,
            agenda_dir,
            database_path: database_path.clone(),
        };
        let result = bootstrap_database(&plan).expect("bootstrap should succeed");

        assert_eq!(result.memory_files, 1);
        assert_eq!(result.agenda_files, 1);
        assert_eq!(result.persisted_documents, 2);
        assert_eq!(result.synced_memories, 1);
        assert_eq!(result.synced_agenda_items, 1);
        assert!(database_path.exists());

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }

    #[test]
    fn context_messages_round_trip_for_user_token() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_migrations(&mut connection).expect("migrations should run");

        let messages = vec![
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }],
            ),
            ConversationMessage::tool_result("call-1", "{\"temp\":25}"),
        ];

        replace_context_messages(&mut connection, "local-user", &messages)
            .expect("messages should persist");
        let loaded =
            load_context_messages(&mut connection, "local-user").expect("messages should load");

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].role, MessageRole::User);
        assert_eq!(loaded[0].content.as_deref(), Some("hello"));
        assert_eq!(loaded[1].role, MessageRole::Assistant);
        assert_eq!(loaded[1].tool_calls.as_ref().map(Vec::len), Some(1));
        assert_eq!(loaded[2].role, MessageRole::Tool);
        assert_eq!(loaded[2].tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn memory_and_agenda_queries_return_active_derived_records() {
        let mut connection = Connection::open_in_memory().expect("sqlite should open");
        run_migrations(&mut connection).expect("migrations should run");

        let unique = format!(
            "elroy-rs-derived-query-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );
        let home = std::env::temp_dir().join(unique);
        let memory_dir = home.join("memories");
        let agenda_dir = home.join("agenda");
        fs::create_dir_all(&memory_dir).expect("memory dir should be created");
        fs::create_dir_all(&agenda_dir).expect("agenda dir should be created");
        fs::write(
            memory_dir.join("runner_notes.md"),
            "---\nid: 9\n---\n\ntrain for the race",
        )
        .expect("memory fixture should be written");
        fs::write(
            agenda_dir.join("doctor_visit.md"),
            "---\ndate: 2026-05-15\ncompleted: false\ntrigger_datetime: 2026-05-15T15:00:00\nstatus: created\n---\n\nbring forms",
        )
        .expect("agenda fixture should be written");

        let inventory = BootstrapInventory {
            memory_files: vec![memory_dir.join("runner_notes.md")],
            agenda_files: vec![agenda_dir.join("doctor_visit.md")],
        };
        let documents = bootstrap_documents(&inventory).expect("documents should parse");
        persist_bootstrap_documents(&mut connection, &documents)
            .expect("bootstrap documents should persist");
        sync_derived_domain_tables(&mut connection, &documents)
            .expect("derived tables should sync");

        let memories = list_active_memories(&connection, 10).expect("memories should query");
        let search =
            search_active_memories(&connection, "race", 10).expect("memory search should query");
        let agenda = list_active_agenda_items(&connection, 10).expect("agenda should query");
        let plain_agenda =
            list_active_plain_agenda_items(&connection, 10).expect("plain agenda should query");
        let due_items = list_active_due_items(&connection, 10).expect("due items should query");
        let exact_memory =
            find_active_memory_by_name(&connection, "runner notes").expect("memory should query");
        let exact_agenda = find_active_agenda_item_by_name(&connection, "doctor visit")
            .expect("agenda item should query");

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].name, "runner notes");
        assert_eq!(search.len(), 1);
        assert_eq!(agenda.len(), 1);
        assert_eq!(plain_agenda.len(), 0);
        assert_eq!(agenda[0].name, "doctor visit");
        assert_eq!(due_items.len(), 1);
        assert_eq!(due_items[0].name, "doctor visit");
        assert_eq!(
            exact_memory.as_ref().map(|memory| memory.name.as_str()),
            Some("runner notes")
        );
        assert_eq!(
            exact_agenda.as_ref().map(|item| item.name.as_str()),
            Some("doctor visit")
        );

        fs::remove_dir_all(home).expect("temp directories should be removed");
    }
}
