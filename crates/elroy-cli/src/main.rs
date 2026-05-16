use std::env;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use elroy_app::{AppRuntime, MessageProcessOptions};
use elroy_config::AppConfig;
use elroy_core::{AppSession, TurnContext};
use elroy_db::{BootstrapInventory, BootstrapPlan, bootstrap_database};
use elroy_llm::StreamEvent;
use elroy_tui::{
    PromptUpdate, SidebarAction, SidebarSection, TuiContextMessage, TuiPromptStream, TuiRunResult,
    TuiRuntime, TuiSidebarDetail, run_with_snapshot_and_runtime,
};

const RESTART_RESUME_MESSAGE_ENV: &str = "ELROY_RESTART_RESUME_MESSAGE";

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = AppConfig::load().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });
    let runtime = AppRuntime::new(config.clone());

    if args.iter().any(|arg| arg == "--tui") {
        let snapshot = runtime.load_snapshot().unwrap_or_default();
        let mut tui_runtime = CliTuiRuntime::new(runtime);
        let result =
            run_with_snapshot_and_runtime(snapshot, &mut tui_runtime).unwrap_or_else(|error| {
                eprintln!("{error}");
                std::process::exit(1);
            });
        if let TuiRunResult::RestartRequested(resume_message) = result {
            restart_current_process(&args, &resume_message);
        }
        return;
    }

    if let Some(prompt) = prompt_arg(&args) {
        run_live_prompt(&runtime, &prompt);
        return;
    }

    let session = AppSession::new("local-user", config.assistant_name.clone());
    let turn = TurnContext::new("bootstrap", session, config.clone());
    let bootstrap = BootstrapPlan::from_config(&config);
    let inventory = BootstrapInventory::discover(&bootstrap);

    println!("elroy-rs workspace bootstrap");
    println!("assistant: {}", turn.session.assistant_name);
    println!("chat model: {}", turn.config.chat_model);
    println!("config path: {}", turn.config.config_path.display());
    println!("memory dir: {}", bootstrap.memory_dir.display());
    println!("agenda dir: {}", bootstrap.agenda_dir.display());
    println!("database path: {}", bootstrap.database_path.display());
    println!("memory files discovered: {}", inventory.memory_files.len());
    println!("agenda files discovered: {}", inventory.agenda_files.len());

    let bootstrap_result = bootstrap_database(&bootstrap).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });
    println!(
        "bootstrap persisted documents: {}",
        bootstrap_result.persisted_documents
    );
    println!("derived memories: {}", bootstrap_result.synced_memories);
    println!(
        "derived agenda items: {}",
        bootstrap_result.synced_agenda_items
    );
}

fn prompt_arg(args: &[String]) -> Option<String> {
    let index = args.iter().position(|arg| arg == "--prompt")?;
    let value = args.get(index + 1..)?;
    if value.is_empty() {
        return None;
    }
    Some(value.join(" "))
}

fn run_live_prompt(runtime: &AppRuntime, prompt: &str) {
    let mut prompt_stream = runtime
        .process_message_stream(prompt, MessageProcessOptions::default())
        .unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });

    for event in prompt_stream.by_ref() {
        match event {
            StreamEvent::AssistantResponse { content } => println!("{content}"),
            StreamEvent::AssistantInternalThought { content } => eprintln!("thinking: {content}"),
            StreamEvent::AssistantToolResult { content, is_error } => {
                let label = if is_error {
                    "tool error"
                } else {
                    "tool result"
                };
                println!("{label}: {content}");
            }
            StreamEvent::StatusUpdate { content } => eprintln!("status: {content}"),
            StreamEvent::ToolCallRequested(call) => {
                println!("tool requested: {} {}", call.name, call.arguments_json);
            }
        }
    }

    if let Err(error) = prompt_stream.into_snapshot() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn restart_current_process(args: &[String], resume_message: &str) -> ! {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        let current_exe = env::current_exe().unwrap_or_else(|error| {
            eprintln!("failed to resolve current executable for restart: {error}");
            std::process::exit(1);
        });

        let mut command = std::process::Command::new(&current_exe);
        command.args(args);
        command.env(RESTART_RESUME_MESSAGE_ENV, resume_message);
        let error = command.exec();
        eprintln!("failed to restart elroy-rs process: {error}");
        std::process::exit(1);
    }

    #[cfg(not(unix))]
    {
        let current_exe = env::current_exe().unwrap_or_else(|error| {
            eprintln!("failed to resolve current executable for restart: {error}");
            std::process::exit(1);
        });
        let status = std::process::Command::new(&current_exe)
            .args(args)
            .env(RESTART_RESUME_MESSAGE_ENV, resume_message)
            .status()
            .unwrap_or_else(|error| {
                eprintln!("failed to restart elroy-rs process: {error}");
                std::process::exit(1);
            });
        std::process::exit(status.code().unwrap_or(1));
    }
}

struct CliTuiRuntime {
    runtime: AppRuntime,
    deferred_context_refresh: Option<BackgroundRefreshTask>,
    deferred_context_refresh_error: Option<String>,
    deferred_self_reflection: Option<BackgroundRefreshTask>,
    deferred_self_reflection_error: Option<String>,
}

struct CliPromptStream {
    inner: elroy_app::PromptEventStream,
}

struct BackgroundRefreshTask {
    handle: JoinHandle<()>,
    error: Arc<Mutex<Option<String>>>,
}

impl BackgroundRefreshTask {
    fn spawn(task: impl FnOnce() -> Result<(), String> + Send + 'static) -> Self {
        let error = Arc::new(Mutex::new(None));
        let error_sink = Arc::clone(&error);
        let handle = thread::spawn(move || {
            if let Err(task_error) = task() {
                *error_sink
                    .lock()
                    .expect("background refresh error lock should work") = Some(task_error);
            }
        });
        Self { handle, error }
    }

    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    fn join(self) -> Option<String> {
        let _ = self.handle.join();
        self.error
            .lock()
            .expect("background refresh error lock should work")
            .take()
    }
}

impl CliTuiRuntime {
    fn new(runtime: AppRuntime) -> Self {
        runtime.enable_restart_support();
        Self {
            runtime,
            deferred_context_refresh: None,
            deferred_context_refresh_error: None,
            deferred_self_reflection: None,
            deferred_self_reflection_error: None,
        }
    }

    fn poll_deferred_context_refresh(&mut self) {
        let Some(task) = self.deferred_context_refresh.as_ref() else {
            return;
        };
        if !task.is_finished() {
            return;
        }

        let task = self
            .deferred_context_refresh
            .take()
            .expect("finished deferred refresh task should exist");
        if let Some(error) = task.join() {
            self.deferred_context_refresh_error = Some(error);
        }
    }

    fn clear_deferred_context_refresh_error(&mut self) {
        self.deferred_context_refresh_error = None;
    }

    fn poll_deferred_self_reflection(&mut self) {
        let Some(task) = self.deferred_self_reflection.as_ref() else {
            return;
        };
        if !task.is_finished() {
            return;
        }

        let task = self
            .deferred_self_reflection
            .take()
            .expect("finished deferred self reflection task should exist");
        if let Some(error) = task.join() {
            self.deferred_self_reflection_error = Some(error);
        }
    }

    fn clear_deferred_self_reflection_error(&mut self) {
        self.deferred_self_reflection_error = None;
    }
}

impl Drop for CliTuiRuntime {
    fn drop(&mut self) {
        self.runtime.disable_restart_support();
    }
}

impl TuiRuntime for CliTuiRuntime {
    fn load_snapshot(&mut self) -> Result<elroy_tui::TuiSnapshot, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.runtime
            .load_snapshot()
            .map_err(|error| error.to_string())
    }

    fn execute_slash_command(
        &mut self,
        prompt: &str,
    ) -> Result<Option<elroy_tui::TuiSnapshot>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.clear_deferred_context_refresh_error();
        self.clear_deferred_self_reflection_error();
        self.runtime
            .execute_slash_command(prompt)
            .map_err(|error| error.to_string())
    }

    fn submit_prompt(&mut self, prompt: &str) -> Result<elroy_tui::TuiSnapshot, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.clear_deferred_context_refresh_error();
        self.clear_deferred_self_reflection_error();
        self.runtime
            .process_message(prompt, MessageProcessOptions::default())
            .map(|result| result.snapshot)
            .map_err(|error| error.to_string())
    }

    fn start_prompt_stream(&mut self, prompt: &str) -> Result<Box<dyn TuiPromptStream>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.clear_deferred_context_refresh_error();
        self.clear_deferred_self_reflection_error();
        self.runtime
            .process_message_stream(
                prompt,
                MessageProcessOptions {
                    defer_self_reflection: true,
                    ..MessageProcessOptions::default()
                },
            )
            .map(|inner| Box::new(CliPromptStream { inner }) as Box<dyn TuiPromptStream>)
            .map_err(|error| error.to_string())
    }

    fn start_startup_prompt_stream(&mut self) -> Result<Option<Box<dyn TuiPromptStream>>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.clear_deferred_context_refresh_error();
        self.clear_deferred_self_reflection_error();
        let restart_resume_message = std::env::var(RESTART_RESUME_MESSAGE_ENV).ok();
        self.runtime
            .startup_prompt_stream(restart_resume_message.as_deref())
            .map(|stream| {
                stream.map(|inner| Box::new(CliPromptStream { inner }) as Box<dyn TuiPromptStream>)
            })
            .map_err(|error| error.to_string())
    }

    fn start_restart_prompt_stream(
        &mut self,
        resume_message: &str,
    ) -> Result<Box<dyn TuiPromptStream>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.clear_deferred_context_refresh_error();
        self.clear_deferred_self_reflection_error();
        self.runtime
            .restart_prompt_stream(resume_message)
            .map(|inner| Box::new(CliPromptStream { inner }) as Box<dyn TuiPromptStream>)
            .map_err(|error| error.to_string())
    }

    fn take_restart_request(&mut self) -> Result<Option<String>, String> {
        Ok(self.runtime.consume_restart_request())
    }

    fn load_context_messages(&mut self) -> Result<Vec<TuiContextMessage>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.runtime
            .load_context_messages()
            .map(|messages| {
                messages
                    .into_iter()
                    .filter_map(|message| {
                        Some(TuiContextMessage {
                            id: message.id?,
                            role: match message.role {
                                elroy_llm::MessageRole::System => "system".to_string(),
                                elroy_llm::MessageRole::User => "user".to_string(),
                                elroy_llm::MessageRole::Assistant => "assistant".to_string(),
                                elroy_llm::MessageRole::Tool => "tool".to_string(),
                            },
                            content: message.content.unwrap_or_default(),
                        })
                    })
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    fn refresh_context_if_needed(&mut self) -> Result<(), String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        if self.deferred_context_refresh.is_some() {
            return Ok(());
        }

        self.clear_deferred_context_refresh_error();
        let runtime = self.runtime.clone();
        self.deferred_context_refresh = Some(BackgroundRefreshTask::spawn(move || {
            runtime
                .refresh_context_if_needed()
                .map(|_| ())
                .map_err(|error| error.to_string())
        }));
        Ok(())
    }

    fn run_self_reflection_if_needed(&mut self) -> Result<(), String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        if self.deferred_self_reflection.is_some() {
            return Ok(());
        }

        self.clear_deferred_self_reflection_error();
        let runtime = self.runtime.clone();
        self.deferred_self_reflection = Some(BackgroundRefreshTask::spawn(move || {
            runtime
                .run_self_reflection_if_needed()
                .map_err(|error| error.to_string())
        }));
        Ok(())
    }

    fn background_status(&mut self) -> Result<Option<String>, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        if let Some(error) = &self.deferred_context_refresh_error {
            return Ok(Some(format!("context refresh failed: {error}")));
        }
        if let Some(error) = &self.deferred_self_reflection_error {
            return Ok(Some(format!("self reflection failed: {error}")));
        }
        Ok(self.runtime.background_status())
    }

    fn open_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
    ) -> Result<TuiSidebarDetail, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.runtime
            .open_sidebar_item(section, title)
            .map_err(|error| error.to_string())
    }

    fn mutate_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
        action: SidebarAction,
    ) -> Result<elroy_tui::TuiSnapshot, String> {
        self.poll_deferred_context_refresh();
        self.poll_deferred_self_reflection();
        self.runtime
            .mutate_sidebar_item(section, title, action)
            .map_err(|error| error.to_string())
    }
}

impl TuiPromptStream for CliPromptStream {
    fn next_update(&mut self) -> Result<Option<PromptUpdate>, String> {
        Ok(self.inner.next().map(|event| match event {
            StreamEvent::AssistantResponse { content } => PromptUpdate::AssistantDelta(content),
            StreamEvent::AssistantInternalThought { content } => {
                PromptUpdate::InternalThought(content)
            }
            StreamEvent::AssistantToolResult { content, is_error } => {
                PromptUpdate::ToolResult { content, is_error }
            }
            StreamEvent::StatusUpdate { content } => PromptUpdate::Status(content),
            StreamEvent::ToolCallRequested(call) => PromptUpdate::ToolCall {
                name: call.name,
                arguments_json: call.arguments_json,
            },
        }))
    }

    fn finalize(self: Box<Self>) -> Result<elroy_tui::TuiSnapshot, String> {
        self.inner
            .into_snapshot()
            .map_err(|error| error.to_string())
    }

    fn cancel(self: Box<Self>) -> Result<elroy_tui::TuiSnapshot, String> {
        self.inner.cancel().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::mpsc;
    use std::time::Duration;

    use elroy_tui::TuiRuntime;

    use super::{AppConfig, AppRuntime, BackgroundRefreshTask, CliTuiRuntime, prompt_arg};

    #[test]
    fn prompt_arg_collects_remaining_words() {
        let args = vec![
            "--prompt".to_string(),
            "hello".to_string(),
            "there".to_string(),
        ];

        assert_eq!(prompt_arg(&args).as_deref(), Some("hello there"));
    }

    #[test]
    fn background_refresh_task_captures_errors() {
        let task = BackgroundRefreshTask::spawn(|| Err("boom".to_string()));
        while !task.is_finished() {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(task.join().as_deref(), Some("boom"));
    }

    #[test]
    fn background_refresh_task_reports_running_state() {
        let (tx, rx) = mpsc::channel();
        let task = BackgroundRefreshTask::spawn(move || {
            rx.recv().expect("signal should arrive");
            Ok(())
        });

        assert!(!task.is_finished());
        tx.send(()).expect("task should receive completion signal");
        while !task.is_finished() {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(task.join(), None);
    }

    #[test]
    fn cli_tui_runtime_surfaces_deferred_refresh_failures_in_background_status() {
        let config = AppConfig::from_env(&HashMap::new()).expect("config should load");
        let mut runtime = CliTuiRuntime {
            runtime: AppRuntime::new(config),
            deferred_context_refresh: Some(BackgroundRefreshTask::spawn(|| {
                Err("refresh exploded".to_string())
            })),
            deferred_context_refresh_error: None,
            deferred_self_reflection: None,
            deferred_self_reflection_error: None,
        };

        let background_status = loop {
            let status = runtime
                .background_status()
                .expect("background status should load");
            if status.is_some() {
                break status;
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        assert_eq!(
            background_status.as_deref(),
            Some("context refresh failed: refresh exploded")
        );
        assert!(runtime.deferred_context_refresh.is_none());
    }

    #[test]
    fn cli_tui_runtime_surfaces_deferred_self_reflection_failures_in_background_status() {
        let config = AppConfig::from_env(&HashMap::new()).expect("config should load");
        let mut runtime = CliTuiRuntime {
            runtime: AppRuntime::new(config),
            deferred_context_refresh: None,
            deferred_context_refresh_error: None,
            deferred_self_reflection: Some(BackgroundRefreshTask::spawn(|| {
                Err("reflection exploded".to_string())
            })),
            deferred_self_reflection_error: None,
        };

        let background_status = loop {
            let status = runtime
                .background_status()
                .expect("background status should load");
            if status.is_some() {
                break status;
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        assert_eq!(
            background_status.as_deref(),
            Some("self reflection failed: reflection exploded")
        );
        assert!(runtime.deferred_self_reflection.is_none());
    }

    #[test]
    fn prompt_arg_ignores_missing_prompt_flag() {
        let args = vec!["--tui".to_string()];
        assert_eq!(prompt_arg(&args), None);
    }
}
