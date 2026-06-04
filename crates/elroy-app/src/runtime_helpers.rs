use std::path::Path;

use chrono::Utc;
use elroy_codex::CodexSessionResult;
use elroy_config::{
    AppConfig, LlmProvider, fast_provider_config_from_app_config, provider_config_from_app_config,
};
use elroy_core::LiveProviderModel;
use elroy_db::{
    BootstrapPlan, LOCAL_USER_TOKEN, UserPreferenceRecord, load_user_preferences,
    open_sqlite_connection, replace_context_messages, run_migrations,
};
use elroy_llm::{ConversationMessage, LiveModelClient, MessageRole};
use elroy_self_reflection::{SelfReflectionConfig, SelfReflectionOrchestrator};
use elroy_tui::TuiSnapshot;
use elroy_user::effective_persona;

use crate::{
    AppError, DeferredAutoMemoryWork, PromptCompletion, PromptEventStreamState,
    PromptExecutionOptions, build_context_summary_message, build_live_tool_registry,
    clear_background_status, compress_context_messages, create_memory_file_from_context_messages,
    is_context_refresh_needed, load_snapshot_from_connection, load_validated_runtime_transcript,
    memory_consolidation_settings_from_app_config, recall_model_clients,
    record_memory_creation_and_maybe_consolidate, run_auto_memory_if_needed,
    run_prompt_with_model_and_registry_internal, set_background_status,
    strip_input_message_for_persistence, strip_transient_context_messages,
    synthetic_tool_context_messages,
};

pub(crate) fn finalize_prompt_event_stream(
    mut state: PromptEventStreamState,
) -> Result<PromptCompletion, AppError> {
    let turn_run = state.turn_stream.finish()?;
    let persisted_transcript = strip_input_message_for_persistence(
        strip_transient_context_messages(
            turn_run.transcript,
            state.existing_transcript_len,
            state.transient_context_count,
        ),
        state.existing_transcript_len,
        state.persist_input_message,
    );
    replace_context_messages(
        &mut state.connection,
        LOCAL_USER_TOKEN,
        &persisted_transcript,
    )?;
    let deferred_auto_memory = if state.defer_auto_memory {
        Some(DeferredAutoMemoryWork {
            existing_transcript_len: state.existing_transcript_len,
            transcript: persisted_transcript.clone(),
        })
    } else {
        run_auto_memory_if_needed(
            &mut state.connection,
            &state.bootstrap_plan,
            state.memories_between_consolidation,
            state.memory_consolidation_settings.as_ref(),
            state.existing_transcript_len,
            persisted_transcript.as_slice(),
            state.messages_between_memory,
        )?;
        None
    };
    if !state.defer_self_reflection {
        run_self_reflection_if_needed(
            &state.home_dir,
            persisted_transcript.as_slice(),
            state.messages_between_self_reflection,
        )?;
    }
    Ok(PromptCompletion {
        snapshot: load_snapshot_from_connection(
            &mut state.connection,
            &state.home_dir,
            &state.bootstrap_plan.memory_dir,
            state.show_internal_thought,
        )?,
        deferred_auto_memory,
    })
}

pub(crate) fn cancel_prompt_event_stream(
    mut state: PromptEventStreamState,
) -> Result<TuiSnapshot, AppError> {
    load_snapshot_from_connection(
        &mut state.connection,
        &state.home_dir,
        &state.bootstrap_plan.memory_dir,
        state.show_internal_thought,
    )
}

pub(crate) fn live_provider_model(
    config: &AppConfig,
    preferences: Option<&UserPreferenceRecord>,
) -> Result<LiveProviderModel, AppError> {
    let provider_config =
        provider_config_from_app_config(config).map_err(AppError::ProviderConfig)?;
    let client = LiveModelClient::new(provider_config)
        .map_err(|error| AppError::Runtime(error.to_string()))?;
    Ok(LiveProviderModel::new(
        client,
        effective_persona(preferences, &config.assistant_name),
    ))
}

pub(crate) fn live_fast_provider_model(
    config: &AppConfig,
    preferences: Option<&UserPreferenceRecord>,
) -> Result<LiveProviderModel, AppError> {
    let provider_config =
        fast_provider_config_from_app_config(config).map_err(AppError::ProviderConfig)?;
    let client = LiveModelClient::new(provider_config)
        .map_err(|error| AppError::Runtime(error.to_string()))?;
    Ok(LiveProviderModel::new(
        client,
        effective_persona(preferences, &config.assistant_name),
    ))
}

pub(crate) fn run_self_reflection_if_needed(
    home_dir: &Path,
    transcript: &[ConversationMessage],
    messages_between_self_reflection: usize,
) -> Result<(), AppError> {
    set_background_status("self-reflection", "reflecting on recent conversation...");
    let result = SelfReflectionOrchestrator::new(SelfReflectionConfig {
        messages_between_self_reflection,
    })
    .run(home_dir, transcript);
    clear_background_status("self-reflection");
    result?;
    Ok(())
}

pub(crate) fn refresh_context_if_needed(
    connection: &mut rusqlite::Connection,
    config: &AppConfig,
    bootstrap_plan: &BootstrapPlan,
) -> Result<bool, AppError> {
    let transcript = load_validated_runtime_transcript(
        connection,
        &config.assistant_name,
        config.llm_provider() == LlmProvider::Anthropic,
    )?;
    if !is_context_refresh_needed(&transcript, config.max_tokens) {
        return Ok(false);
    }

    set_background_status("context-refresh", "refreshing context...");

    let result = (|| {
        let compressed = compress_context_messages(
            &transcript,
            config.context_refresh_target_tokens(),
            config.max_context_age_minutes,
        );
        if transcript
            .iter()
            .any(|message| message.role == MessageRole::User)
        {
            let (name, text) = elroy_recall::formulate_memory_from_transcript(&transcript);
            create_memory_file_from_context_messages(
                &bootstrap_plan.memory_dir,
                &name,
                &text,
                &transcript,
            )?;
            elroy_db::bootstrap_database(bootstrap_plan)
                .map_err(|error| AppError::Runtime(error.to_string()))?;
            *connection = open_sqlite_connection(&bootstrap_plan.database_path)?;
            record_memory_creation_and_maybe_consolidate(
                connection,
                bootstrap_plan,
                config.memories_between_consolidation,
                Some(&memory_consolidation_settings_from_app_config(config)),
            )?;
        }

        let mut refreshed_transcript = compressed;
        if transcript
            .iter()
            .any(|message| message.role == MessageRole::User)
        {
            let tool_call_id = format!(
                "context-summary-{}",
                Utc::now()
                    .timestamp_nanos_opt()
                    .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000)
            );
            refreshed_transcript.extend(synthetic_tool_context_messages(
                tool_call_id,
                "context_summary",
                "{}",
                build_context_summary_message(connection, config, &transcript),
            ));
        }

        replace_context_messages(connection, LOCAL_USER_TOKEN, &refreshed_transcript)?;
        Ok(true)
    })();

    clear_background_status("context-refresh");
    result
}

pub(crate) fn run_background_codex_completion_followup(
    config: &AppConfig,
    result: &CodexSessionResult,
) -> Result<(), AppError> {
    let mut connection = open_sqlite_connection(&config.database_path)?;
    run_migrations(&mut connection)?;
    let preferences = load_user_preferences(&connection, LOCAL_USER_TOKEN)?;
    let model = live_provider_model(config, preferences.as_ref())?;
    let prompt = codex_completion_followup_prompt(result);
    run_prompt_with_model_and_registry_internal(
        &mut connection,
        &prompt,
        &model,
        recall_model_clients(Some(&model)),
        build_live_tool_registry(config),
        PromptExecutionOptions {
            role: MessageRole::User,
            persist_input_message: false,
            force_tool: None,
            assistant_name: &config.assistant_name,
            ensure_alternating_roles: config.llm_provider() == LlmProvider::Anthropic,
            home_dir: &config.home_dir,
            bootstrap_plan: BootstrapPlan::from_config(config),
            messages_between_memory: config.messages_between_memory,
            memories_between_consolidation: config.memories_between_consolidation,
            memory_consolidation_settings: Some(memory_consolidation_settings_from_app_config(
                config,
            )),
            messages_between_self_reflection: config.messages_between_self_reflection,
            defer_auto_memory: false,
            defer_self_reflection: false,
            memory_recall_classifier_enabled: config.memory_recall_classifier_enabled,
            memory_recall_classifier_window: config.memory_recall_classifier_window,
            reflect: config.reflect,
        },
    )?;
    Ok(())
}

pub(crate) fn codex_completion_followup_prompt(result: &CodexSessionResult) -> String {
    format!(
        "A background Codex session completed.\n\nSession: {}\nRepository: {}\nWorktree: {}\nSession branch: {}\nTarget branch: {}\nStatus: {}\nSummary:\n{}\n\nRespond to the user about the outcome and decide whether any follow-up action is needed.",
        result.session_id,
        result.repo_path,
        result.worktree_path.as_deref().unwrap_or("n/a"),
        result.session_branch.as_deref().unwrap_or("n/a"),
        result.target_branch.as_deref().unwrap_or("n/a"),
        result.status,
        result.summary,
    )
}
