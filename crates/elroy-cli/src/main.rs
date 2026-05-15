use std::env;

use elroy_app::AppRuntime;
use elroy_config::AppConfig;
use elroy_core::{AppSession, TurnContext};
use elroy_db::{BootstrapInventory, BootstrapPlan, bootstrap_database};
use elroy_llm::StreamEvent;
use elroy_tui::{SidebarAction, SidebarSection, TuiRuntime, run_with_snapshot_and_runtime};

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
        run_with_snapshot_and_runtime(snapshot, &mut tui_runtime).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        });
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
    let prompt_run = runtime.submit_prompt(prompt).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });

    for event in prompt_run.events {
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
}

struct CliTuiRuntime {
    runtime: AppRuntime,
}

impl CliTuiRuntime {
    fn new(runtime: AppRuntime) -> Self {
        Self { runtime }
    }
}

impl TuiRuntime for CliTuiRuntime {
    fn submit_prompt(&mut self, prompt: &str) -> Result<elroy_tui::TuiSnapshot, String> {
        self.runtime
            .submit_prompt(prompt)
            .map(|result| result.snapshot)
            .map_err(|error| error.to_string())
    }

    fn open_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
    ) -> Result<String, String> {
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
        self.runtime
            .mutate_sidebar_item(section, title, action)
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::prompt_arg;

    #[test]
    fn prompt_arg_collects_remaining_words() {
        let args = vec![
            "--prompt".to_string(),
            "hello".to_string(),
            "there".to_string(),
        ];

        assert_eq!(prompt_arg(&args).as_deref(), Some("hello there"));
    }
}
