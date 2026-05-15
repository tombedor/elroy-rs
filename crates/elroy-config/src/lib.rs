use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use serde::Deserialize;

const SQLITE_URL_PREFIX: &str = "sqlite:///";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub home_dir: PathBuf,
    pub config_path: PathBuf,
    pub memory_dir: PathBuf,
    pub agenda_dir: PathBuf,
    pub database_path: PathBuf,
    pub chat_model: String,
    pub assistant_name: String,
    pub include_base_tools: bool,
    pub async_runtime_enabled: bool,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
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
            assistant_name: "Elroy".to_string(),
            include_base_tools: true,
            async_runtime_enabled: true,
            openai_api_key: None,
            openai_base_url: "https://api.openai.com/v1/responses".to_string(),
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
        if let Some(assistant_name) = file_config.default_assistant_name {
            self.assistant_name = assistant_name;
        }
        if let Some(include_base_tools) = file_config.include_base_tools {
            self.include_base_tools = include_base_tools;
        }
        if let Some(openai_api_key) = file_config.openai_api_key {
            self.openai_api_key = Some(openai_api_key);
        }
        if let Some(openai_base_url) = file_config.openai_base_url {
            self.openai_base_url = openai_base_url;
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
        if let Some(assistant_name) = env.get("ELROY_DEFAULT_ASSISTANT_NAME") {
            self.assistant_name = assistant_name.clone();
        }
        if let Some(include_base_tools) = env.get("ELROY_INCLUDE_BASE_TOOLS") {
            self.include_base_tools = parse_bool(include_base_tools);
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
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    chat_model: Option<String>,
    default_assistant_name: Option<String>,
    include_base_tools: Option<bool>,
    memory_dir: Option<String>,
    agenda_dir: Option<String>,
    database_url: Option<String>,
    openai_api_key: Option<String>,
    openai_base_url: Option<String>,
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
            "chat_model: gpt-5-nano\nmemory_dir: /tmp/elroy-memories\nagenda_dir: /tmp/elroy-agenda\ndatabase_url: sqlite:////tmp/elroy.db\nirrelevant_key: ignored\n",
        )
        .expect("config fixture should be written");

        let env = HashMap::from([("ELROY_HOME".to_string(), home_dir.display().to_string())]);
        let config = AppConfig::from_env(&env).expect("config should load");

        assert_eq!(config.chat_model, "gpt-5-nano");
        assert_eq!(config.memory_dir, PathBuf::from("/tmp/elroy-memories"));
        assert_eq!(config.agenda_dir, PathBuf::from("/tmp/elroy-agenda"));
        assert_eq!(config.database_path, PathBuf::from("/tmp/elroy.db"));
        assert_eq!(config.config_path, config_path);
        assert_eq!(config.llm_provider(), LlmProvider::OpenAi);
        assert!(config.include_base_tools);

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
            (
                "ELROY_DEFAULT_ASSISTANT_NAME".to_string(),
                "EnvElroy".to_string(),
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
        assert_eq!(config.assistant_name, "EnvElroy");
        assert_eq!(config.memory_dir, PathBuf::from("/tmp/env-memories"));
        assert_eq!(config.agenda_dir, PathBuf::from("/tmp/env-agenda"));
        assert_eq!(config.database_path, PathBuf::from("/tmp/env.db"));
        assert!(!config.include_base_tools);
        assert!(!config.async_runtime_enabled);
        assert_eq!(config.openai_api_key.as_deref(), Some("openai-test-key"));
        assert_eq!(
            config.anthropic_api_key.as_deref(),
            Some("anthropic-test-key")
        );
        assert_eq!(config.openai_base_url, "http://localhost:1234/openai");
        assert_eq!(config.anthropic_base_url, "http://localhost:1234/anthropic");
        assert_eq!(config.anthropic_api_version, "2099-01-01");
        assert_eq!(config.llm_provider(), LlmProvider::Anthropic);

        fs::remove_dir_all(home_dir).expect("temp home dir should be removed");
    }

    #[test]
    fn non_claude_models_default_to_openai_provider() {
        let config = AppConfig::defaults();

        assert_eq!(config.llm_provider(), LlmProvider::OpenAi);
    }
}
