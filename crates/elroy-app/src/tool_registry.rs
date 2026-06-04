use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use elroy_agenda::agenda_tools;
use elroy_codex::CodexSessionResult;
use elroy_codex::tools::codex_tools;
use elroy_config::AppConfig;
use elroy_context::context_tools as transcript_context_tools;
use elroy_feature_requests::feature_request_tools;
use elroy_memory::tools::memory_tools;
use elroy_tasks::task_tools;
use elroy_tools::{BaseToolCallbacks, ExecutableToolRegistry, base_tools, render_plain_text_table};
use elroy_user::tools::user_tools;

use crate::{refresh_persisted_system_instructions, run_background_codex_completion_followup};

static RESTART_STATE: OnceLock<Mutex<RestartState>> = OnceLock::new();

#[derive(Debug, Default)]
struct RestartState {
    supported: bool,
    pending_resume_prompt: Option<String>,
}

fn restart_state() -> &'static Mutex<RestartState> {
    RESTART_STATE.get_or_init(|| Mutex::new(RestartState::default()))
}

pub(crate) fn enable_session_restart_support() {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.supported = true;
}

pub(crate) fn disable_session_restart_support() {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.supported = false;
    state.pending_resume_prompt = None;
}

fn request_session_restart(resume_prompt: &str) -> Result<(), &'static str> {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    if !state.supported {
        return Err("Session restart is not available in this Elroy runtime.");
    }
    state.pending_resume_prompt = Some(resume_prompt.to_string());
    Ok(())
}

pub(crate) fn consume_session_restart_request() -> Option<String> {
    let mut state = restart_state()
        .lock()
        .expect("restart state lock should work");
    state.pending_resume_prompt.take()
}

pub fn build_live_tool_registry(config: &AppConfig) -> ExecutableToolRegistry {
    build_live_tool_registry_with_codex_bin_and_hook(config, None, None)
}

pub(crate) fn build_live_tool_registry_with_codex_bin_and_hook(
    config: &AppConfig,
    codex_bin_override: Option<PathBuf>,
    codex_completion_hook_override: Option<Arc<dyn Fn(CodexSessionResult) + Send + Sync>>,
) -> ExecutableToolRegistry {
    if !config.include_base_tools {
        return ExecutableToolRegistry::new(vec![]);
    }

    let config_for_codex_completion = config.clone();
    let codex_completion_hook = codex_completion_hook_override.unwrap_or_else(|| {
        Arc::new(move |result| {
            if let Err(error) =
                run_background_codex_completion_followup(&config_for_codex_completion, &result)
            {
                eprintln!(
                    "failed to run background codex completion follow-up for {}: {}",
                    result.session_id, error
                );
            }
        })
    });

    let mut fr_tools_iter = feature_request_tools(config.home_dir.clone()).into_iter();
    let list_feature_requests_tool = fr_tools_iter.next().expect("feature_request_tools[0]");
    let make_feature_request = fr_tools_iter.next().expect("feature_request_tools[1]");
    let edit_feature_request = fr_tools_iter.next().expect("feature_request_tools[2]");

    let excluded_tools = config.exclude_tools.iter().cloned().collect::<HashSet<_>>();
    let config_for_help = config.clone();
    let base_tool_callbacks = BaseToolCallbacks {
        request_restart: Arc::new(|resume_message| {
            request_session_restart(resume_message).map_err(|error| error.to_string())
        }),
        list_commands: Arc::new(move || {
            build_live_tool_registry(&config_for_help)
                .specs()
                .into_iter()
                .map(|spec| (spec.name, spec.description))
                .collect()
        }),
        render_config_report: {
            let config_for_print_config = config.clone();
            Arc::new(move || render_config_report(&config_for_print_config))
        },
        log_path: config.home_dir.join("logs").join("elroy.log"),
    };
    let mut tools = base_tools(base_tool_callbacks);
    tools.extend(memory_tools(config));
    tools.extend(codex_tools(
        config.clone(),
        codex_bin_override,
        codex_completion_hook.clone(),
    ));
    tools.extend(user_tools(
        config.clone(),
        Arc::new(refresh_persisted_system_instructions),
    ));
    tools.extend(task_tools(config.clone()));
    tools.extend(agenda_tools(config));
    tools.extend(transcript_context_tools(config));
    tools.extend(vec![
        list_feature_requests_tool,
        make_feature_request,
        edit_feature_request,
    ]);

    ExecutableToolRegistry::new(
        tools
            .into_iter()
            .filter(|tool| !excluded_tools.contains(&tool.spec().name))
            .collect(),
    )
}

fn render_config_report(config: &AppConfig) -> String {
    let mut redacted_config = config.clone();
    if redacted_config.openai_api_key.is_some() {
        redacted_config.openai_api_key = Some("********".to_string());
    }
    if redacted_config.anthropic_api_key.is_some() {
        redacted_config.anthropic_api_key = Some("********".to_string());
    }
    if redacted_config.fast_model_api_key.is_some() {
        redacted_config.fast_model_api_key = Some("********".to_string());
    }
    if redacted_config.embedding_model_api_key.is_some() {
        redacted_config.embedding_model_api_key = Some("********".to_string());
    }
    let none = "(none)".to_string();
    let rows = vec![
        vec![
            "General".to_string(),
            "Chat Model".to_string(),
            redacted_config.chat_model,
        ],
        vec![
            "General".to_string(),
            "Config Path".to_string(),
            redacted_config.config_path.display().to_string(),
        ],
        vec![
            "General".to_string(),
            "Reflect".to_string(),
            redacted_config.reflect.to_string(),
        ],
        vec![
            "General".to_string(),
            "Show Internal Thought".to_string(),
            redacted_config.show_internal_thought.to_string(),
        ],
        vec![
            "Memory".to_string(),
            "Memories Between Consolidation".to_string(),
            redacted_config.memories_between_consolidation.to_string(),
        ],
        vec![
            "Memory".to_string(),
            "L2 Memory Relevance Distance Threshold".to_string(),
            redacted_config
                .l2_memory_relevance_distance_threshold
                .to_string(),
        ],
        vec![
            "Memory".to_string(),
            "Memory Cluster Similarity".to_string(),
            redacted_config
                .memory_cluster_similarity_threshold
                .to_string(),
        ],
        vec![
            "Memory".to_string(),
            "Max Memory Cluster Size".to_string(),
            redacted_config.max_memory_cluster_size.to_string(),
        ],
        vec![
            "Memory".to_string(),
            "Min Memory Cluster Size".to_string(),
            redacted_config.min_memory_cluster_size.to_string(),
        ],
        vec![
            "API".to_string(),
            "Chat API Key".to_string(),
            redacted_config
                .openai_api_key
                .clone()
                .or(redacted_config.anthropic_api_key.clone())
                .unwrap_or_else(|| none.clone()),
        ],
        vec![
            "API".to_string(),
            "Anthropic API Key".to_string(),
            redacted_config
                .anthropic_api_key
                .unwrap_or_else(|| none.clone()),
        ],
        vec![
            "Tools".to_string(),
            "Exclude Tools".to_string(),
            if redacted_config.exclude_tools.is_empty() {
                none
            } else {
                redacted_config.exclude_tools.join(", ")
            },
        ],
    ];
    render_plain_text_table(
        "Elroy Configuration",
        &["Section", "Setting", "Value"],
        &rows,
    )
}
