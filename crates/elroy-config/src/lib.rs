use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use serde::Deserialize;

const SQLITE_URL_PREFIX: &str = "sqlite:///";

#[derive(Debug, Clone, PartialEq)]
pub struct AppConfig {
    pub home_dir: PathBuf,
    pub config_path: PathBuf,
    pub memory_dir: PathBuf,
    pub agenda_dir: PathBuf,
    pub database_path: PathBuf,
    pub chat_model: String,
    pub fast_model: Option<String>,
    pub embedding_model: String,
    pub embedding_model_size: usize,
    pub max_tokens: usize,
    pub assistant_name: String,
    pub reflect: bool,
    pub enable_assistant_greeting: bool,
    pub min_convo_age_for_greeting_minutes: f64,
    pub max_context_age_minutes: f64,
    pub messages_between_memory: usize,
    pub memories_between_consolidation: usize,
    pub recency_weight: f64,
    pub l2_memory_relevance_distance_threshold: f64,
    pub memory_cluster_similarity_threshold: f64,
    pub max_memory_cluster_size: usize,
    pub min_memory_cluster_size: usize,
    pub memory_reflection_max_words: usize,
    pub messages_between_self_reflection: usize,
    pub memory_recall_classifier_enabled: bool,
    pub memory_recall_classifier_window: usize,
    pub include_base_tools: bool,
    pub exclude_tools: Vec<String>,
    pub async_runtime_enabled: bool,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub fast_model_api_key: Option<String>,
    pub fast_model_api_base: Option<String>,
    pub embedding_model_api_key: Option<String>,
    pub embedding_model_api_base: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub anthropic_base_url: String,
    pub anthropic_api_version: String,
}

impl AppConfig {
    pub fn defaults() -> Self {
        Self::for_home(default_home_dir())
    }

    pub fn load() -> Result<Self, ConfigError> {
        let env = std::env::vars().collect::<HashMap<_, _>>();
        Self::from_env(&env)
    }

    pub fn from_env(env: &HashMap<String, String>) -> Result<Self, ConfigError> {
        let home_dir = env
            .get("ELROY_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(default_home_dir);
        let config_path = env
            .get("ELROY_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir.join("elroy.conf.yaml"));
        let file_config = read_file_config(&config_path)?;

        let mut config = Self::for_home(home_dir.clone());
        config.config_path = config_path;

        if let Some(file_config) = file_config {
            config.apply_file(file_config);
        }

        config.apply_env(env);
        Ok(config)
    }

    fn for_home(home_dir: PathBuf) -> Self {
        let memory_dir = home_dir.join("memories");
        let agenda_dir = home_dir.join("agenda");
        let database_path = home_dir.join("elroy.db");

        Self {
            home_dir,
            config_path: default_home_dir().join("elroy.conf.yaml"),
            memory_dir,
            agenda_dir,
            database_path,
            chat_model: "gpt-5".to_string(),
            fast_model: None,
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_model_size: 1536,
            max_tokens: 100_000,
            assistant_name: "Elroy".to_string(),
            reflect: false,
            enable_assistant_greeting: false,
            min_convo_age_for_greeting_minutes: 5.0,
            max_context_age_minutes: 720.0,
            messages_between_memory: 20,
            memories_between_consolidation: 4,
            recency_weight: 0.0,
            l2_memory_relevance_distance_threshold: 1.4,
            memory_cluster_similarity_threshold: 0.21125,
            max_memory_cluster_size: 5,
            min_memory_cluster_size: 3,
            memory_reflection_max_words: 100,
            messages_between_self_reflection: 10,
            memory_recall_classifier_enabled: true,
            memory_recall_classifier_window: 3,
            include_base_tools: true,
            exclude_tools: Vec::new(),
            async_runtime_enabled: true,
            openai_api_key: None,
            openai_base_url: "https://api.openai.com/v1/responses".to_string(),
            fast_model_api_key: None,
            fast_model_api_base: None,
            embedding_model_api_key: None,
            embedding_model_api_base: None,
            anthropic_api_key: None,
            anthropic_base_url: "https://api.anthropic.com/v1/messages".to_string(),
            anthropic_api_version: "2023-06-01".to_string(),
        }
    }

    fn apply_file(&mut self, file_config: FileConfig) {
        if let Some(memory_dir) = file_config.memory_dir {
            self.memory_dir = PathBuf::from(memory_dir);
        }
        if let Some(agenda_dir) = file_config.agenda_dir {
            self.agenda_dir = PathBuf::from(agenda_dir);
        }
        if let Some(database_url) = file_config.database_url {
            self.database_path = parse_database_path(&database_url);
        }
        if let Some(chat_model) = file_config.chat_model {
            self.chat_model = chat_model;
        }
        if let Some(fast_model) = file_config.fast_model {
            self.fast_model = Some(fast_model);
        }
        if let Some(embedding_model) = file_config.embedding_model {
            self.embedding_model = embedding_model;
        }
        if let Some(embedding_model_size) = file_config.embedding_model_size {
            self.embedding_model_size = embedding_model_size;
        }
        if let Some(max_tokens) = file_config.max_tokens {
            self.max_tokens = max_tokens;
        }
        if let Some(assistant_name) = file_config.default_assistant_name {
            self.assistant_name = assistant_name;
        }
        if let Some(reflect) = file_config.reflect {
            self.reflect = reflect;
        }
        if let Some(enable_assistant_greeting) = file_config.enable_assistant_greeting {
            self.enable_assistant_greeting = enable_assistant_greeting;
        }
        if let Some(min_convo_age_for_greeting_minutes) =
            file_config.min_convo_age_for_greeting_minutes
        {
            self.min_convo_age_for_greeting_minutes = min_convo_age_for_greeting_minutes;
        }
        if let Some(max_context_age_minutes) = file_config.max_context_age_minutes {
            self.max_context_age_minutes = max_context_age_minutes;
        }
        if let Some(messages_between_memory) = file_config.messages_between_memory {
            self.messages_between_memory = messages_between_memory;
        }
        if let Some(memories_between_consolidation) = file_config.memories_between_consolidation {
            self.memories_between_consolidation = memories_between_consolidation;
        }
        if let Some(recency_weight) = file_config.recency_weight {
            self.recency_weight = recency_weight;
        }
        if let Some(l2_memory_relevance_distance_threshold) =
            file_config.l2_memory_relevance_distance_threshold
        {
            self.l2_memory_relevance_distance_threshold = l2_memory_relevance_distance_threshold;
        }
        if let Some(memory_cluster_similarity_threshold) =
            file_config.memory_cluster_similarity_threshold
        {
            self.memory_cluster_similarity_threshold = memory_cluster_similarity_threshold;
        }
        if let Some(max_memory_cluster_size) = file_config.max_memory_cluster_size {
            self.max_memory_cluster_size = max_memory_cluster_size;
        }
        if let Some(min_memory_cluster_size) = file_config.min_memory_cluster_size {
            self.min_memory_cluster_size = min_memory_cluster_size;
        }
        if let Some(memory_reflection_max_words) = file_config.memory_reflection_max_words {
            self.memory_reflection_max_words = memory_reflection_max_words;
        }
        if let Some(messages_between_self_reflection) = file_config.messages_between_self_reflection
        {
            self.messages_between_self_reflection = messages_between_self_reflection;
        }
        if let Some(memory_recall_classifier_enabled) = file_config.memory_recall_classifier_enabled
        {
            self.memory_recall_classifier_enabled = memory_recall_classifier_enabled;
        }
        if let Some(memory_recall_classifier_window) = file_config.memory_recall_classifier_window {
            self.memory_recall_classifier_window = memory_recall_classifier_window;
        }
        if let Some(include_base_tools) = file_config.include_base_tools {
            self.include_base_tools = include_base_tools;
        }
        if let Some(exclude_tools) = file_config.exclude_tools {
            self.exclude_tools = exclude_tools;
        }
        if let Some(openai_api_key) = file_config.openai_api_key {
            self.openai_api_key = Some(openai_api_key);
        }
        if let Some(openai_base_url) = file_config.openai_base_url {
            self.openai_base_url = openai_base_url;
        }
        if let Some(fast_model_api_key) = file_config.fast_model_api_key {
            self.fast_model_api_key = Some(fast_model_api_key);
        }
        if let Some(fast_model_api_base) = file_config.fast_model_api_base {
            self.fast_model_api_base = Some(fast_model_api_base);
        }
        if let Some(embedding_model_api_key) = file_config.embedding_model_api_key {
            self.embedding_model_api_key = Some(embedding_model_api_key);
        }
        if let Some(embedding_model_api_base) = file_config.embedding_model_api_base {
            self.embedding_model_api_base = Some(embedding_model_api_base);
        }
        if let Some(anthropic_api_key) = file_config.anthropic_api_key {
            self.anthropic_api_key = Some(anthropic_api_key);
        }
        if let Some(anthropic_base_url) = file_config.anthropic_base_url {
            self.anthropic_base_url = anthropic_base_url;
        }
        if let Some(anthropic_api_version) = file_config.anthropic_api_version {
            self.anthropic_api_version = anthropic_api_version;
        }
    }

    fn apply_env(&mut self, env: &HashMap<String, String>) {
        if let Some(memory_dir) = env.get("ELROY_MEMORY_DIR") {
            self.memory_dir = PathBuf::from(memory_dir);
        }
        if let Some(agenda_dir) = env.get("ELROY_AGENDA_DIR") {
            self.agenda_dir = PathBuf::from(agenda_dir);
        }
        if let Some(database_url) = env.get("ELROY_DATABASE_URL") {
            self.database_path = parse_database_path(database_url);
        }
        if let Some(chat_model) = env.get("ELROY_CHAT_MODEL") {
            self.chat_model = chat_model.clone();
        }
        if let Some(fast_model) = env.get("ELROY_FAST_MODEL") {
            self.fast_model = Some(fast_model.clone());
        }
        if let Some(embedding_model) = env.get("ELROY_EMBEDDING_MODEL") {
            self.embedding_model = embedding_model.clone();
        }
        if let Some(embedding_model_size) = env.get("ELROY_EMBEDDING_MODEL_SIZE") {
            self.embedding_model_size = parse_usize(embedding_model_size);
        }
        if let Some(max_tokens) = env.get("ELROY_MAX_TOKENS") {
            self.max_tokens = parse_usize(max_tokens);
        }
        if let Some(assistant_name) = env.get("ELROY_DEFAULT_ASSISTANT_NAME") {
            self.assistant_name = assistant_name.clone();
        }
        if let Some(reflect) = env.get("ELROY_REFLECT") {
            self.reflect = parse_bool(reflect);
        }
        if let Some(enable_assistant_greeting) = env.get("ELROY_ENABLE_ASSISTANT_GREETING") {
            self.enable_assistant_greeting = parse_bool(enable_assistant_greeting);
        }
        if let Some(min_convo_age_for_greeting_minutes) =
            env.get("ELROY_MIN_CONVO_AGE_FOR_GREETING_MINUTES")
        {
            self.min_convo_age_for_greeting_minutes = parse_f64(min_convo_age_for_greeting_minutes);
        }
        if let Some(max_context_age_minutes) = env.get("ELROY_MAX_CONTEXT_AGE_MINUTES") {
            self.max_context_age_minutes = parse_f64(max_context_age_minutes);
        }
        if let Some(messages_between_memory) = env.get("ELROY_MESSAGES_BETWEEN_MEMORY") {
            self.messages_between_memory = parse_usize(messages_between_memory);
        }
        if let Some(memories_between_consolidation) =
            env.get("ELROY_MEMORIES_BETWEEN_CONSOLIDATION")
        {
            self.memories_between_consolidation = parse_usize(memories_between_consolidation);
        }
        if let Some(recency_weight) = env.get("ELROY_RECENCY_WEIGHT") {
            self.recency_weight = parse_f64(recency_weight);
        }
        if let Some(l2_memory_relevance_distance_threshold) =
            env.get("ELROY_L2_MEMORY_RELEVANCE_DISTANCE_THRESHOLD")
        {
            self.l2_memory_relevance_distance_threshold =
                parse_f64(l2_memory_relevance_distance_threshold);
        }
        if let Some(memory_cluster_similarity_threshold) =
            env.get("ELROY_MEMORY_CLUSTER_SIMILARITY_THRESHOLD")
        {
            self.memory_cluster_similarity_threshold =
                parse_f64(memory_cluster_similarity_threshold);
        }
        if let Some(max_memory_cluster_size) = env.get("ELROY_MAX_MEMORY_CLUSTER_SIZE") {
            self.max_memory_cluster_size = parse_usize(max_memory_cluster_size);
        }
        if let Some(min_memory_cluster_size) = env.get("ELROY_MIN_MEMORY_CLUSTER_SIZE") {
            self.min_memory_cluster_size = parse_usize(min_memory_cluster_size);
        }
        if let Some(memory_reflection_max_words) = env.get("ELROY_MEMORY_REFLECTION_MAX_WORDS") {
            self.memory_reflection_max_words = parse_usize(memory_reflection_max_words);
        }
        if let Some(messages_between_self_reflection) =
            env.get("ELROY_MESSAGES_BETWEEN_SELF_REFLECTION")
        {
            self.messages_between_self_reflection = parse_usize(messages_between_self_reflection);
        }
        if let Some(memory_recall_classifier_enabled) =
            env.get("ELROY_MEMORY_RECALL_CLASSIFIER_ENABLED")
        {
            self.memory_recall_classifier_enabled = parse_bool(memory_recall_classifier_enabled);
        }
        if let Some(memory_recall_classifier_window) =
            env.get("ELROY_MEMORY_RECALL_CLASSIFIER_WINDOW")
        {
            self.memory_recall_classifier_window = parse_usize(memory_recall_classifier_window);
        }
        if let Some(include_base_tools) = env.get("ELROY_INCLUDE_BASE_TOOLS") {
            self.include_base_tools = parse_bool(include_base_tools);
        }
        if let Some(exclude_tools) = env.get("ELROY_EXCLUDE_TOOLS") {
            self.exclude_tools = parse_csv_list(exclude_tools);
        }
        if let Some(async_runtime_enabled) = env.get("ELROY_ASYNC_RUNTIME") {
            self.async_runtime_enabled = parse_bool(async_runtime_enabled);
        }
        if let Some(openai_api_key) = env.get("OPENAI_API_KEY") {
            self.openai_api_key = Some(openai_api_key.clone());
        }
        if let Some(openai_base_url) = env.get("ELROY_OPENAI_BASE_URL") {
            self.openai_base_url = openai_base_url.clone();
        }
        if let Some(fast_model_api_key) = env.get("ELROY_FAST_MODEL_API_KEY") {
            self.fast_model_api_key = Some(fast_model_api_key.clone());
        }
        if let Some(fast_model_api_base) = env.get("ELROY_FAST_MODEL_API_BASE") {
            self.fast_model_api_base = Some(fast_model_api_base.clone());
        }
        if let Some(embedding_model_api_key) = env
            .get("ELROY_EMBEDDING_MODEL_API_KEY")
            .or(env.get("OPENAI_API_KEY"))
        {
            self.embedding_model_api_key = Some(embedding_model_api_key.clone());
        }
        if let Some(embedding_model_api_base) = env.get("ELROY_EMBEDDING_MODEL_API_BASE") {
            self.embedding_model_api_base = Some(embedding_model_api_base.clone());
        }
        if let Some(anthropic_api_key) = env.get("ANTHROPIC_API_KEY") {
            self.anthropic_api_key = Some(anthropic_api_key.clone());
        }
        if let Some(anthropic_base_url) = env.get("ELROY_ANTHROPIC_BASE_URL") {
            self.anthropic_base_url = anthropic_base_url.clone();
        }
        if let Some(anthropic_api_version) = env.get("ELROY_ANTHROPIC_API_VERSION") {
            self.anthropic_api_version = anthropic_api_version.clone();
        }
    }

    pub fn llm_provider(&self) -> LlmProvider {
        LlmProvider::for_model(&self.chat_model)
    }

    pub fn fast_model_name(&self) -> &str {
        self.fast_model
            .as_deref()
            .unwrap_or(self.chat_model.as_str())
    }

    pub fn fast_llm_provider(&self) -> LlmProvider {
        LlmProvider::for_model(self.fast_model_name())
    }

    pub fn context_refresh_target_tokens(&self) -> usize {
        self.max_tokens / 3
    }
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    chat_model: Option<String>,
    fast_model: Option<String>,
    embedding_model: Option<String>,
    embedding_model_size: Option<usize>,
    max_tokens: Option<usize>,
    default_assistant_name: Option<String>,
    reflect: Option<bool>,
    enable_assistant_greeting: Option<bool>,
    min_convo_age_for_greeting_minutes: Option<f64>,
    max_context_age_minutes: Option<f64>,
    messages_between_memory: Option<usize>,
    memories_between_consolidation: Option<usize>,
    recency_weight: Option<f64>,
    l2_memory_relevance_distance_threshold: Option<f64>,
    memory_cluster_similarity_threshold: Option<f64>,
    max_memory_cluster_size: Option<usize>,
    min_memory_cluster_size: Option<usize>,
    memory_reflection_max_words: Option<usize>,
    messages_between_self_reflection: Option<usize>,
    memory_recall_classifier_enabled: Option<bool>,
    memory_recall_classifier_window: Option<usize>,
    include_base_tools: Option<bool>,
    exclude_tools: Option<Vec<String>>,
    memory_dir: Option<String>,
    agenda_dir: Option<String>,
    database_url: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
    fast_model_api_key: Option<String>,
    fast_model_api_base: Option<String>,
    embedding_model_api_key: Option<String>,
    embedding_model_api_base: Option<String>,
    anthropic_api_key: Option<String>,
    anthropic_base_url: Option<String>,
    anthropic_api_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    OpenAi,
    Anthropic,
}

impl LlmProvider {
    pub fn for_model(model: &str) -> Self {
        if model.starts_with("claude") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    ParseConfig {
        path: PathBuf,
        source: serde_yaml::Error,
    },
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadConfig { path, source } => {
                write!(f, "failed to read config file {}: {source}", path.display())
            }
            Self::ParseConfig { path, source } => {
                write!(
                    f,
                    "failed to parse config file {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn default_home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".elroy")
}

fn read_file_config(path: &Path) -> Result<Option<FileConfig>, ConfigError> {
    if !path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    let config =
        serde_yaml::from_str::<FileConfig>(&raw).map_err(|source| ConfigError::ParseConfig {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(Some(config))
}

fn parse_database_path(value: &str) -> PathBuf {
    if let Some(stripped) = value.strip_prefix(SQLITE_URL_PREFIX) {
        return PathBuf::from(stripped);
    }

    PathBuf::from(value)
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn parse_f64(value: &str) -> f64 {
    value.trim().parse::<f64>().unwrap_or(0.0)
}

fn parse_usize(value: &str) -> usize {
    value.trim().parse::<usize>().unwrap_or(0)
}

fn parse_csv_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

    use super::{AppConfig, LlmProvider};

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let unique = format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos()
        );

        std::env::temp_dir().join(unique)
    }

    #[test]
    fn defaults_preserve_async_first_direction() {
        let config = AppConfig::defaults();

        assert!(config.async_runtime_enabled);
        assert!(config.include_base_tools);
        assert!(config.exclude_tools.is_empty());
        assert_eq!(config.embedding_model, "text-embedding-3-small");
        assert_eq!(config.embedding_model_size, 1536);
        assert_eq!(config.max_tokens, 100_000);
        assert_eq!(config.context_refresh_target_tokens(), 33_333);
        assert!(!config.reflect);
        assert!(!config.enable_assistant_greeting);
        assert_eq!(config.min_convo_age_for_greeting_minutes, 5.0);
        assert_eq!(config.max_context_age_minutes, 720.0);
        assert_eq!(config.messages_between_memory, 20);
        assert_eq!(config.memories_between_consolidation, 4);
        assert_eq!(config.recency_weight, 0.0);
        assert_eq!(config.l2_memory_relevance_distance_threshold, 1.4);
        assert_eq!(config.memory_cluster_similarity_threshold, 0.21125);
        assert_eq!(config.max_memory_cluster_size, 5);
        assert_eq!(config.min_memory_cluster_size, 3);
        assert_eq!(config.memory_reflection_max_words, 100);
        assert_eq!(config.messages_between_self_reflection, 10);
        assert!(config.memory_recall_classifier_enabled);
        assert_eq!(config.memory_recall_classifier_window, 3);
    }

    #[test]
    fn defaults_keep_memory_files_under_home_dir() {
        let config = AppConfig::defaults();

        assert_eq!(config.memory_dir, config.home_dir.join("memories"));
        assert_eq!(config.agenda_dir, config.home_dir.join("agenda"));
        assert_eq!(config.database_path, config.home_dir.join("elroy.db"));
        assert_eq!(config.config_path, config.home_dir.join("elroy.conf.yaml"));
        assert_eq!(
            config.openai_base_url,
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(config.fast_model, None);
        assert_eq!(config.fast_model_api_key, None);
        assert_eq!(config.fast_model_api_base, None);
        assert_eq!(config.embedding_model_api_key, None);
        assert_eq!(config.embedding_model_api_base, None);
        assert_eq!(
            config.anthropic_base_url,
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(config.anthropic_api_version, "2023-06-01");
    }

    #[test]
    fn loads_yaml_config_and_ignores_unknown_keys() {
        let home_dir = unique_temp_dir("elroy-rs-config-file");
        fs::create_dir_all(&home_dir).expect("home dir should be created");
        let config_path = home_dir.join("elroy.conf.yaml");
        fs::write(
            &config_path,
            "chat_model: gpt-5-nano\nfast_model: gpt-5.4-mini\nembedding_model: text-embedding-3-large\nembedding_model_size: 3072\nmax_tokens: 9000\nreflect: true\nenable_assistant_greeting: true\nmin_convo_age_for_greeting_minutes: 15.5\nmax_context_age_minutes: 180.0\nmessages_between_memory: 12\nmemories_between_consolidation: 6\nrecency_weight: 0.125\nl2_memory_relevance_distance_threshold: 1.11\nmemory_cluster_similarity_threshold: 0.33\nmax_memory_cluster_size: 7\nmin_memory_cluster_size: 4\nmemory_reflection_max_words: 42\nmessages_between_self_reflection: 4\nmemory_recall_classifier_enabled: false\nmemory_recall_classifier_window: 7\nexclude_tools:\n  - get_user_preferred_name\n  - get_help\nmemory_dir: /tmp/elroy-memories\nagenda_dir: /tmp/elroy-agenda\ndatabase_url: sqlite:////tmp/elroy.db\nfast_model_api_key: fast-key\nfast_model_api_base: http://localhost:1234/fast\nembedding_model_api_key: embed-key\nembedding_model_api_base: http://localhost:1234/embeddings\nirrelevant_key: ignored\n",
        )
        .expect("config fixture should be written");

        let env = HashMap::from([("ELROY_HOME".to_string(), home_dir.display().to_string())]);
        let config = AppConfig::from_env(&env).expect("config should load");

        assert_eq!(config.chat_model, "gpt-5-nano");
        assert_eq!(config.fast_model.as_deref(), Some("gpt-5.4-mini"));
        assert_eq!(config.embedding_model, "text-embedding-3-large");
        assert_eq!(config.embedding_model_size, 3072);
        assert_eq!(config.max_tokens, 9000);
        assert_eq!(config.context_refresh_target_tokens(), 3000);
        assert_eq!(config.memory_dir, PathBuf::from("/tmp/elroy-memories"));
        assert_eq!(config.agenda_dir, PathBuf::from("/tmp/elroy-agenda"));
        assert_eq!(config.database_path, PathBuf::from("/tmp/elroy.db"));
        assert_eq!(config.config_path, config_path);
        assert_eq!(config.llm_provider(), LlmProvider::OpenAi);
        assert!(config.reflect);
        assert!(config.enable_assistant_greeting);
        assert_eq!(config.min_convo_age_for_greeting_minutes, 15.5);
        assert_eq!(config.max_context_age_minutes, 180.0);
        assert_eq!(config.messages_between_memory, 12);
        assert_eq!(config.memories_between_consolidation, 6);
        assert_eq!(config.recency_weight, 0.125);
        assert_eq!(config.l2_memory_relevance_distance_threshold, 1.11);
        assert_eq!(config.memory_cluster_similarity_threshold, 0.33);
        assert_eq!(config.max_memory_cluster_size, 7);
        assert_eq!(config.min_memory_cluster_size, 4);
        assert_eq!(config.memory_reflection_max_words, 42);
        assert_eq!(config.messages_between_self_reflection, 4);
        assert!(!config.memory_recall_classifier_enabled);
        assert_eq!(config.memory_recall_classifier_window, 7);
        assert_eq!(config.fast_model_api_key.as_deref(), Some("fast-key"));
        assert_eq!(
            config.fast_model_api_base.as_deref(),
            Some("http://localhost:1234/fast")
        );
        assert_eq!(config.embedding_model_api_key.as_deref(), Some("embed-key"));
        assert_eq!(
            config.embedding_model_api_base.as_deref(),
            Some("http://localhost:1234/embeddings")
        );
        assert!(config.include_base_tools);
        assert_eq!(
            config.exclude_tools,
            vec![
                "get_user_preferred_name".to_string(),
                "get_help".to_string()
            ]
        );

        fs::remove_dir_all(home_dir).expect("temp home dir should be removed");
    }

    #[test]
    fn environment_overrides_file_values() {
        let home_dir = unique_temp_dir("elroy-rs-config-env");
        fs::create_dir_all(&home_dir).expect("home dir should be created");
        let config_path = home_dir.join("elroy.conf.yaml");
        fs::write(
            &config_path,
            "chat_model: gpt-5-nano\ndefault_assistant_name: FileElroy\nmemory_dir: /tmp/file-memories\nagenda_dir: /tmp/file-agenda\n",
        )
        .expect("config fixture should be written");

        let env = HashMap::from([
            ("ELROY_HOME".to_string(), home_dir.display().to_string()),
            (
                "ELROY_CHAT_MODEL".to_string(),
                "claude-sonnet-4-5-20250929".to_string(),
            ),
            ("ELROY_FAST_MODEL".to_string(), "gpt-5.4-mini".to_string()),
            (
                "ELROY_EMBEDDING_MODEL".to_string(),
                "text-embedding-3-large".to_string(),
            ),
            ("ELROY_EMBEDDING_MODEL_SIZE".to_string(), "3072".to_string()),
            ("ELROY_MAX_TOKENS".to_string(), "12000".to_string()),
            (
                "ELROY_DEFAULT_ASSISTANT_NAME".to_string(),
                "EnvElroy".to_string(),
            ),
            ("ELROY_REFLECT".to_string(), "true".to_string()),
            (
                "ELROY_ENABLE_ASSISTANT_GREETING".to_string(),
                "true".to_string(),
            ),
            (
                "ELROY_MIN_CONVO_AGE_FOR_GREETING_MINUTES".to_string(),
                "2.5".to_string(),
            ),
            (
                "ELROY_MAX_CONTEXT_AGE_MINUTES".to_string(),
                "45.0".to_string(),
            ),
            ("ELROY_MESSAGES_BETWEEN_MEMORY".to_string(), "8".to_string()),
            (
                "ELROY_MESSAGES_BETWEEN_SELF_REFLECTION".to_string(),
                "6".to_string(),
            ),
            (
                "ELROY_MEMORIES_BETWEEN_CONSOLIDATION".to_string(),
                "9".to_string(),
            ),
            ("ELROY_RECENCY_WEIGHT".to_string(), "0.2".to_string()),
            (
                "ELROY_L2_MEMORY_RELEVANCE_DISTANCE_THRESHOLD".to_string(),
                "1.75".to_string(),
            ),
            (
                "ELROY_MEMORY_CLUSTER_SIMILARITY_THRESHOLD".to_string(),
                "0.44".to_string(),
            ),
            ("ELROY_MAX_MEMORY_CLUSTER_SIZE".to_string(), "8".to_string()),
            ("ELROY_MIN_MEMORY_CLUSTER_SIZE".to_string(), "2".to_string()),
            (
                "ELROY_MEMORY_REFLECTION_MAX_WORDS".to_string(),
                "64".to_string(),
            ),
            (
                "ELROY_MEMORY_RECALL_CLASSIFIER_ENABLED".to_string(),
                "false".to_string(),
            ),
            (
                "ELROY_MEMORY_RECALL_CLASSIFIER_WINDOW".to_string(),
                "9".to_string(),
            ),
            (
                "ELROY_MEMORY_DIR".to_string(),
                "/tmp/env-memories".to_string(),
            ),
            (
                "ELROY_AGENDA_DIR".to_string(),
                "/tmp/env-agenda".to_string(),
            ),
            (
                "ELROY_DATABASE_URL".to_string(),
                "sqlite:////tmp/env.db".to_string(),
            ),
            ("ELROY_INCLUDE_BASE_TOOLS".to_string(), "false".to_string()),
            (
                "ELROY_EXCLUDE_TOOLS".to_string(),
                "get_user_preferred_name, get_help".to_string(),
            ),
            ("ELROY_ASYNC_RUNTIME".to_string(), "false".to_string()),
            ("OPENAI_API_KEY".to_string(), "openai-test-key".to_string()),
            (
                "ANTHROPIC_API_KEY".to_string(),
                "anthropic-test-key".to_string(),
            ),
            (
                "ELROY_OPENAI_BASE_URL".to_string(),
                "http://localhost:1234/openai".to_string(),
            ),
            (
                "ELROY_FAST_MODEL_API_KEY".to_string(),
                "fast-test-key".to_string(),
            ),
            (
                "ELROY_FAST_MODEL_API_BASE".to_string(),
                "http://localhost:1234/fast".to_string(),
            ),
            (
                "ELROY_EMBEDDING_MODEL_API_BASE".to_string(),
                "http://localhost:1234/embeddings".to_string(),
            ),
            (
                "ELROY_ANTHROPIC_BASE_URL".to_string(),
                "http://localhost:1234/anthropic".to_string(),
            ),
            (
                "ELROY_ANTHROPIC_API_VERSION".to_string(),
                "2099-01-01".to_string(),
            ),
        ]);
        let config = AppConfig::from_env(&env).expect("config should load");

        assert_eq!(config.chat_model, "claude-sonnet-4-5-20250929");
        assert_eq!(config.fast_model.as_deref(), Some("gpt-5.4-mini"));
        assert_eq!(config.embedding_model, "text-embedding-3-large");
        assert_eq!(config.embedding_model_size, 3072);
        assert_eq!(config.max_tokens, 12_000);
        assert_eq!(config.context_refresh_target_tokens(), 4_000);
        assert_eq!(config.assistant_name, "EnvElroy");
        assert!(config.reflect);
        assert!(config.enable_assistant_greeting);
        assert_eq!(config.min_convo_age_for_greeting_minutes, 2.5);
        assert_eq!(config.max_context_age_minutes, 45.0);
        assert_eq!(config.messages_between_memory, 8);
        assert_eq!(config.memories_between_consolidation, 9);
        assert_eq!(config.recency_weight, 0.2);
        assert_eq!(config.l2_memory_relevance_distance_threshold, 1.75);
        assert_eq!(config.memory_cluster_similarity_threshold, 0.44);
        assert_eq!(config.max_memory_cluster_size, 8);
        assert_eq!(config.min_memory_cluster_size, 2);
        assert_eq!(config.memory_reflection_max_words, 64);
        assert_eq!(config.messages_between_self_reflection, 6);
        assert!(!config.memory_recall_classifier_enabled);
        assert_eq!(config.memory_recall_classifier_window, 9);
        assert_eq!(config.memory_dir, PathBuf::from("/tmp/env-memories"));
        assert_eq!(config.agenda_dir, PathBuf::from("/tmp/env-agenda"));
        assert_eq!(config.database_path, PathBuf::from("/tmp/env.db"));
        assert!(!config.include_base_tools);
        assert_eq!(
            config.exclude_tools,
            vec![
                "get_user_preferred_name".to_string(),
                "get_help".to_string()
            ]
        );
        assert!(!config.async_runtime_enabled);
        assert_eq!(config.openai_api_key.as_deref(), Some("openai-test-key"));
        assert_eq!(config.fast_model_api_key.as_deref(), Some("fast-test-key"));
        assert_eq!(
            config.fast_model_api_base.as_deref(),
            Some("http://localhost:1234/fast")
        );
        assert_eq!(
            config.embedding_model_api_key.as_deref(),
            Some("openai-test-key")
        );
        assert_eq!(
            config.embedding_model_api_base.as_deref(),
            Some("http://localhost:1234/embeddings")
        );
        assert_eq!(
            config.anthropic_api_key.as_deref(),
            Some("anthropic-test-key")
        );
        assert_eq!(config.openai_base_url, "http://localhost:1234/openai");
        assert_eq!(config.anthropic_base_url, "http://localhost:1234/anthropic");
        assert_eq!(config.anthropic_api_version, "2099-01-01");
        assert_eq!(config.llm_provider(), LlmProvider::Anthropic);
        assert_eq!(config.fast_llm_provider(), LlmProvider::OpenAi);

        fs::remove_dir_all(home_dir).expect("temp home dir should be removed");
    }

    #[test]
    fn non_claude_models_default_to_openai_provider() {
        let config = AppConfig::defaults();

        assert_eq!(config.llm_provider(), LlmProvider::OpenAi);
    }
}
