use std::collections::HashSet;
use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Widget;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarSection {
    Memories,
    Agenda,
    Improvements,
    FeatureRequests,
    CodexSessions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandPane {
    Conversation,
    Sidebar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusTarget {
    Input,
    Command(CommandPane),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiIntent {
    SubmitPrompt,
    CancelPrompt,
    HistoryPrevious,
    HistoryNext,
    CompleteInput,
    MoveUp,
    MoveDown,
    OpenSelected,
    ArchiveSelected,
    CompleteSelected,
    DeleteSelected,
    Noop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiApp {
    pub title: String,
    pub model_name: String,
    pub status: String,
    pub prompt_active: bool,
    pub command_active: bool,
    pub background_status: Option<String>,
    pub input: String,
    pub input_completions: Vec<String>,
    pub input_history: Vec<String>,
    pub history_index: Option<usize>,
    pub history_draft: Option<String>,
    pub command_palette: Option<CommandPaletteState>,
    pub command_form: Option<CommandFormState>,
    pub detail_modal: Option<DetailModalState>,
    pub sidebar_section: SidebarSection,
    pub focus: FocusTarget,
    pub last_command_pane: CommandPane,
    pub conversation_lines: Vec<String>,
    pub conversation_scroll: usize,
    pub follow_conversation_output: bool,
    pub memory_titles: Vec<String>,
    pub agenda_titles: Vec<String>,
    pub improvement_titles: Vec<String>,
    pub feature_request_titles: Vec<String>,
    pub codex_session_titles: Vec<String>,
    pub selected_sidebar_index: usize,
    pub rendered_context_message_ids: HashSet<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiExit {
    Quit,
    Continue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiRunResult {
    Quit,
    RestartRequested(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarAction {
    Archive,
    Complete,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiSidebarDetail {
    pub title: String,
    pub content: String,
    pub can_complete: bool,
    pub destructive_action: Option<SidebarAction>,
    pub destructive_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiCommandParameter {
    pub name: String,
    pub optional: bool,
    pub default_text: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiCommandForm {
    pub command_name: String,
    pub description: String,
    pub parameters: Vec<TuiCommandParameter>,
    pub initial_values: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiCommandPaletteAction {
    FocusMemories,
    FocusAgenda,
    FocusImprovements,
    FocusFeatureRequests,
    FocusCodexSessions,
    ToolCommand(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiCommandPaletteEntry {
    pub title: String,
    pub description: String,
    pub action: TuiCommandPaletteAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiCommandExecution {
    pub command_name: String,
    pub display_name: String,
    pub values: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiSlashCommandAction {
    NotHandled,
    Execute(TuiCommandExecution),
    OpenForm(TuiCommandForm),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailModalState {
    pub title: String,
    pub content: String,
    pub can_complete: bool,
    pub destructive_action: Option<SidebarAction>,
    pub destructive_label: Option<String>,
    pub confirming_destructive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandFormFieldState {
    pub name: String,
    pub value: String,
    pub optional: bool,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandFormState {
    pub command_name: String,
    pub description: String,
    pub fields: Vec<CommandFormFieldState>,
    pub selected_field: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandPaletteState {
    pub all_entries: Vec<TuiCommandPaletteEntry>,
    pub query: String,
    pub filtered_indices: Vec<usize>,
    pub selected_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptUpdate {
    AssistantDelta(String),
    InternalThought(String),
    ToolCall {
        name: String,
        arguments_json: String,
    },
    ToolResult {
        content: String,
        is_error: bool,
    },
    Status(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiContextMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
}

pub trait TuiPromptStream {
    fn next_update(&mut self) -> Result<Option<PromptUpdate>, String>;
    fn finalize(self: Box<Self>) -> Result<TuiSnapshot, String>;
    fn cancel(self: Box<Self>) -> Result<TuiSnapshot, String>;
}

pub trait TuiRuntime {
    fn load_snapshot(&mut self) -> Result<TuiSnapshot, String>;
    fn load_command_palette_entries(&mut self) -> Result<Vec<TuiCommandPaletteEntry>, String>;
    fn launch_named_command(&mut self, name: &str) -> Result<TuiSlashCommandAction, String>;
    fn handle_slash_command(&mut self, prompt: &str) -> Result<TuiSlashCommandAction, String>;
    fn start_command_execution(&mut self, command: TuiCommandExecution) -> Result<(), String>;
    fn poll_command_execution(&mut self) -> Result<Option<TuiSnapshot>, String>;
    fn submit_prompt(&mut self, prompt: &str) -> Result<TuiSnapshot, String>;
    fn start_prompt_stream(&mut self, prompt: &str) -> Result<Box<dyn TuiPromptStream>, String>;
    fn start_startup_prompt_stream(&mut self) -> Result<Option<Box<dyn TuiPromptStream>>, String>;
    fn start_restart_prompt_stream(
        &mut self,
        resume_message: &str,
    ) -> Result<Box<dyn TuiPromptStream>, String>;
    fn take_restart_request(&mut self) -> Result<Option<String>, String>;
    fn load_context_messages(&mut self) -> Result<Vec<TuiContextMessage>, String>;
    fn refresh_context_if_needed(&mut self) -> Result<(), String>;
    fn run_self_reflection_if_needed(&mut self) -> Result<(), String>;
    fn background_status(&mut self) -> Result<Option<String>, String>;
    fn open_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
    ) -> Result<TuiSidebarDetail, String>;
    fn mutate_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
        action: SidebarAction,
    ) -> Result<TuiSnapshot, String>;
}

pub fn run() -> io::Result<TuiRunResult> {
    run_with_snapshot(TuiSnapshot::default())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TuiSnapshot {
    pub conversation_lines: Vec<String>,
    pub memory_titles: Vec<String>,
    pub agenda_titles: Vec<String>,
    pub input_completions: Vec<String>,
    pub improvement_titles: Vec<String>,
    pub feature_request_titles: Vec<String>,
    pub codex_session_titles: Vec<String>,
    pub model_name: Option<String>,
    pub status: Option<String>,
}

pub fn run_with_snapshot(snapshot: TuiSnapshot) -> io::Result<TuiRunResult> {
    run_with_snapshot_and_runtime(snapshot, &mut NoopRuntime)
}

pub fn run_with_snapshot_and_runtime<R: TuiRuntime>(
    snapshot: TuiSnapshot,
    runtime: &mut R,
) -> io::Result<TuiRunResult> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = TuiApp::from_snapshot(snapshot);
    if let Ok(context_messages) = runtime.load_context_messages() {
        app.mark_context_messages_rendered(&context_messages);
    }
    let mut pending_prompt = start_startup_prompt_stream(&mut app, runtime);
    let result = run_event_loop(&mut terminal, &mut app, runtime, &mut pending_prompt);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    pending_prompt: &mut Option<PendingPrompt>,
) -> io::Result<TuiRunResult> {
    let mut context_poll_ready = pending_prompt.is_none();
    let mut last_context_poll = Instant::now();
    let mut deferred_context_refresh_at = None;
    let mut previous_background_status = None;

    loop {
        if let Some(result) = drive_runtime_tick(
            app,
            runtime,
            pending_prompt,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        ) {
            return Ok(result);
        }
        maybe_enable_context_polling_after_prompt_completion(
            pending_prompt,
            &mut context_poll_ready,
            &mut last_context_poll,
            Instant::now(),
        );
        maybe_poll_context_updates(
            app,
            runtime,
            context_poll_ready,
            &mut last_context_poll,
            Instant::now(),
        );
        terminal.draw(|frame| {
            app.render(frame.area(), frame.buffer_mut());
        })?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        match event::read()? {
            Event::Key(key) => {
                if apply_key_event(app, key, runtime, pending_prompt) == TuiExit::Quit {
                    return Ok(TuiRunResult::Quit);
                }
            }
            Event::Paste(text) => {
                apply_paste_event(app, &text);
            }
            Event::Mouse(mouse) => {
                apply_mouse_event(app, mouse);
            }
            Event::Resize(_, _) => {
                app.status = "terminal resized".to_string();
            }
            _ => {}
        }
    }
}

fn maybe_enable_context_polling_after_prompt_completion(
    pending_prompt: &Option<PendingPrompt>,
    context_poll_ready: &mut bool,
    last_context_poll: &mut Instant,
    now: Instant,
) {
    if !*context_poll_ready && pending_prompt.is_none() {
        *context_poll_ready = true;
        *last_context_poll = now - Duration::from_secs(1);
    }
}

fn drive_runtime_tick(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    pending_prompt: &mut Option<PendingPrompt>,
    now: Instant,
    deferred_context_refresh_at: &mut Option<Instant>,
    previous_background_status: &mut Option<String>,
) -> Option<TuiRunResult> {
    if let Some(resume_message) = maybe_complete_command_execution(app, runtime) {
        return Some(TuiRunResult::RestartRequested(resume_message));
    }
    match advance_prompt_stream(app, runtime, pending_prompt) {
        PromptAdvance::CompletedTurn => {
            *deferred_context_refresh_at = Some(now + Duration::from_secs(5));
        }
        PromptAdvance::RestartRequested(resume_message) => {
            return Some(TuiRunResult::RestartRequested(resume_message));
        }
        PromptAdvance::Noop => {}
    }
    app.prompt_active = pending_prompt.is_some();
    let current_background_status = runtime.background_status().unwrap_or(None);
    maybe_refresh_snapshot_after_background_completion(
        app,
        runtime,
        pending_prompt.is_some(),
        app.command_active,
        previous_background_status.as_deref(),
        current_background_status.as_deref(),
    );
    app.background_status = current_background_status.clone();
    let foreground_blocked = pending_prompt.is_some() || app.command_active;
    let background_completion_pending =
        previous_background_status.is_some() && current_background_status.is_none();
    if !(foreground_blocked && background_completion_pending) {
        *previous_background_status = current_background_status;
    }
    maybe_run_deferred_context_refresh(
        app,
        runtime,
        pending_prompt.is_some(),
        app.command_active,
        now,
        deferred_context_refresh_at,
    );
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PromptAdvance {
    Noop,
    CompletedTurn,
    RestartRequested(String),
}

fn start_startup_prompt_stream(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
) -> Option<PendingPrompt> {
    match runtime.start_startup_prompt_stream() {
        Ok(Some(stream)) => {
            app.status = "thinking...".to_string();
            Some(PendingPrompt {
                submitted_prompt: None,
                schedule_self_reflection: false,
                before_ids: app.rendered_context_message_ids.clone(),
                stream,
            })
        }
        Ok(None) => None,
        Err(error) => {
            app.status = format!("startup failed: {error}");
            None
        }
    }
}

fn apply_key_event(
    app: &mut TuiApp,
    key: KeyEvent,
    runtime: &mut impl TuiRuntime,
    pending_prompt: &mut Option<PendingPrompt>,
) -> TuiExit {
    if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.status = "quit".to_string();
        return TuiExit::Quit;
    }

    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        apply_intent_with_runtime(app, UiIntent::CancelPrompt, runtime, pending_prompt);
        return TuiExit::Continue;
    }

    if app.command_palette.is_some() {
        handle_command_palette_key(app, key, runtime);
        return TuiExit::Continue;
    }

    if app.command_form.is_some() {
        handle_command_form_key(app, key, runtime);
        return TuiExit::Continue;
    }

    if app.detail_modal.is_some() {
        if let Some(token) = modal_key_token(key) {
            let intent = app.handle_modal_key(token);
            apply_intent_with_runtime(app, intent, runtime, pending_prompt);
        }
        return TuiExit::Continue;
    }

    if key.code == KeyCode::Char('p') && key.modifiers.contains(KeyModifiers::CONTROL) {
        match runtime.load_command_palette_entries() {
            Ok(entries) => {
                app.open_command_palette(entries);
                app.status = "command palette opened".to_string();
            }
            Err(error) => {
                app.status = format!("command palette failed: {error}");
            }
        }
        return TuiExit::Continue;
    }

    if app.focus == FocusTarget::Input {
        match key.code {
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.reset_prompt_history_navigation();
                app.input.push(ch);
                app.status = "editing prompt".to_string();
                return TuiExit::Continue;
            }
            KeyCode::Backspace => {
                app.reset_prompt_history_navigation();
                app.input.pop();
                app.status = "editing prompt".to_string();
                return TuiExit::Continue;
            }
            _ => {}
        }
    }

    if let Some(token) = key_event_token(key) {
        let intent = app.handle_key(token);
        apply_intent_with_runtime(app, intent, runtime, pending_prompt);
    }

    TuiExit::Continue
}

fn apply_intent_with_runtime(
    app: &mut TuiApp,
    intent: UiIntent,
    runtime: &mut impl TuiRuntime,
    pending_prompt: &mut Option<PendingPrompt>,
) {
    match intent {
        UiIntent::SubmitPrompt => {
            let submitted = app.input.trim().to_string();
            if submitted.is_empty() {
                app.status = "prompt was empty".to_string();
                return;
            }
            if pending_prompt.is_some() || app.command_active {
                app.status = "Wait for the current task to finish before sending another message."
                    .to_string();
                return;
            }
            match runtime.handle_slash_command(&submitted) {
                Ok(TuiSlashCommandAction::Execute(command)) => {
                    match start_command_execution(app, runtime, command) {
                        Ok(()) => {
                            app.record_submitted_prompt(&submitted);
                            app.input.clear();
                        }
                        Err(error) => {
                            app.status = format!("command launch failed: {error}");
                        }
                    }
                    return;
                }
                Ok(TuiSlashCommandAction::OpenForm(form)) => {
                    app.record_submitted_prompt(&submitted);
                    app.input.clear();
                    app.open_command_form(form);
                    app.status = format!(
                        "editing slash command: /{}",
                        app.command_form
                            .as_ref()
                            .expect("command form should open")
                            .command_name
                    );
                    return;
                }
                Ok(TuiSlashCommandAction::NotHandled) => {}
                Err(error) => {
                    app.status = format!("slash command failed: {error}");
                    return;
                }
            }
            match runtime.start_prompt_stream(&submitted) {
                Ok(stream) => {
                    app.conversation_lines.push(format!("user: {submitted}"));
                    app.follow_conversation_output = true;
                    app.record_submitted_prompt(&submitted);
                    app.input.clear();
                    app.status = "thinking...".to_string();
                    *pending_prompt = Some(PendingPrompt {
                        submitted_prompt: Some(submitted),
                        schedule_self_reflection: true,
                        before_ids: app.rendered_context_message_ids.clone(),
                        stream,
                    });
                }
                Err(error) => {
                    app.status = format!("prompt failed: {error}");
                }
            }
        }
        UiIntent::CancelPrompt => {
            if let Some(pending) = pending_prompt.take() {
                match pending.stream.cancel() {
                    Ok(snapshot) => {
                        app.apply_snapshot(snapshot);
                        app.focus = FocusTarget::Input;
                        app.follow_conversation_output = true;
                        app.status = "Chat stream cancelled".to_string();
                    }
                    Err(error) => {
                        app.status = format!("cancel failed: {error}");
                    }
                }
                return;
            }
            if app.focus == FocusTarget::Input {
                app.input.clear();
                app.status = "cleared prompt".to_string();
            }
        }
        UiIntent::OpenSelected => {
            let label = app
                .active_sidebar_items()
                .get(app.selected_sidebar_index)
                .cloned()
                .unwrap_or_else(|| "unknown entry".to_string());
            match runtime.open_sidebar_item(app.sidebar_section, &label) {
                Ok(detail) => {
                    app.open_detail_modal(detail);
                    app.status = format!("open selected {label}");
                }
                Err(error) => {
                    app.status = format!("open failed: {error}");
                }
            }
        }
        UiIntent::ArchiveSelected | UiIntent::CompleteSelected | UiIntent::DeleteSelected => {
            let label = app
                .active_sidebar_items()
                .get(app.selected_sidebar_index)
                .cloned()
                .unwrap_or_else(|| "unknown entry".to_string());
            let action = match intent {
                UiIntent::ArchiveSelected => SidebarAction::Archive,
                UiIntent::CompleteSelected => SidebarAction::Complete,
                UiIntent::DeleteSelected => SidebarAction::Delete,
                _ => unreachable!(),
            };
            match runtime.mutate_sidebar_item(app.sidebar_section, &label, action) {
                Ok(snapshot) => {
                    app.apply_snapshot(snapshot);
                    app.detail_modal = None;
                    app.status = format!("updated selected {label}");
                }
                Err(error) => {
                    app.status = format!("update failed: {error}");
                }
            }
        }
        _ => app.apply_intent(intent),
    }
}

fn start_command_execution(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    command: TuiCommandExecution,
) -> Result<(), String> {
    if app.command_active || app.prompt_active {
        return Err(
            "Wait for the current task to finish before sending another message.".to_string(),
        );
    }

    let display_name = command.display_name.clone();
    match runtime.start_command_execution(command) {
        Ok(()) => {
            app.command_active = true;
            app.focus = FocusTarget::Input;
            app.follow_conversation_output = true;
            app.status = format!("running command: /{display_name}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn apply_paste_event(app: &mut TuiApp, text: &str) {
    let flattened = flatten_pasted_text(text);
    if flattened.is_empty() {
        return;
    }

    if app.focus == FocusTarget::Input {
        app.reset_prompt_history_navigation();
        app.input.push_str(&flattened);
        app.status = "editing prompt".to_string();
    }
}

fn apply_mouse_event(app: &mut TuiApp, mouse: MouseEvent) {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            if app.can_mouse_scroll_conversation() {
                app.scroll_conversation_up();
            }
        }
        MouseEventKind::ScrollDown => {
            if app.can_mouse_scroll_conversation() {
                app.scroll_conversation_down();
            }
        }
        _ => {}
    }
}

fn maybe_complete_command_execution(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
) -> Option<String> {
    if !app.command_active {
        return None;
    }

    match runtime.poll_command_execution() {
        Ok(Some(snapshot)) => {
            app.command_active = false;
            app.apply_snapshot(snapshot);
            app.focus = FocusTarget::Input;
            app.follow_conversation_output = true;
            match runtime.take_restart_request() {
                Ok(Some(resume_message)) => return Some(resume_message),
                Ok(None) => {}
                Err(error) => {
                    app.status = format!("restart failed: {error}");
                }
            }
        }
        Ok(None) => {}
        Err(error) => {
            app.command_active = false;
            app.status = format!("command failed: {error}");
            app.focus = FocusTarget::Input;
            app.follow_conversation_output = true;
        }
    }
    None
}

fn flatten_pasted_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn wrapped_line_count(text: &str, width: usize) -> usize {
    let width = width.max(1);
    let mut total = 0usize;

    for line in text.split('\n') {
        let char_count = line.chars().count();
        total += if char_count == 0 {
            1
        } else {
            char_count.div_ceil(width)
        };
    }

    total.max(1)
}

struct PendingPrompt {
    submitted_prompt: Option<String>,
    schedule_self_reflection: bool,
    before_ids: HashSet<i64>,
    stream: Box<dyn TuiPromptStream>,
}

fn advance_prompt_stream(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    pending_prompt: &mut Option<PendingPrompt>,
) -> PromptAdvance {
    let Some(pending) = pending_prompt.as_mut() else {
        return PromptAdvance::Noop;
    };
    match pending.stream.next_update() {
        Ok(Some(update)) => apply_prompt_update(app, update),
        Ok(None) => {
            let pending = pending_prompt.take().expect("pending prompt should exist");
            match pending.stream.finalize() {
                Ok(snapshot) => {
                    app.apply_snapshot(snapshot);
                    app.focus = FocusTarget::Input;
                    app.follow_conversation_output = true;
                    let context_messages = runtime.load_context_messages().unwrap_or_default();
                    if let Some(submitted_prompt) = pending.submitted_prompt {
                        app.mark_messages_rendered_after_chat_turn(
                            &submitted_prompt,
                            &pending.before_ids,
                            &context_messages,
                        );
                        app.status = format!("submitted prompt: {submitted_prompt}");
                    } else {
                        app.mark_messages_rendered_after_bootstrap_stream(
                            &pending.before_ids,
                            &context_messages,
                        );
                    }
                    match runtime.take_restart_request() {
                        Ok(Some(resume_message)) => {
                            return PromptAdvance::RestartRequested(resume_message);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            app.status = format!("restart failed: {error}");
                            return PromptAdvance::CompletedTurn;
                        }
                    }
                    if pending.schedule_self_reflection
                        && let Err(error) = runtime.run_self_reflection_if_needed()
                    {
                        app.status = format!("self reflection failed: {error}");
                        return PromptAdvance::CompletedTurn;
                    }
                }
                Err(error) => {
                    app.status = format!("prompt failed: {error}");
                }
            }
            return PromptAdvance::CompletedTurn;
        }
        Err(error) => {
            pending_prompt.take();
            app.status = format!("prompt failed: {error}");
        }
    }
    PromptAdvance::Noop
}

fn maybe_run_deferred_context_refresh(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    prompt_active: bool,
    command_active: bool,
    now: Instant,
    deferred_context_refresh_at: &mut Option<Instant>,
) {
    let Some(deadline) = *deferred_context_refresh_at else {
        return;
    };
    if prompt_active || command_active || now < deadline {
        return;
    }

    if let Err(error) = runtime.refresh_context_if_needed() {
        app.status = format!("context refresh failed: {error}");
    }
    *deferred_context_refresh_at = None;
}

fn maybe_refresh_snapshot_after_background_completion(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    prompt_active: bool,
    command_active: bool,
    previous_background_status: Option<&str>,
    current_background_status: Option<&str>,
) {
    if prompt_active
        || command_active
        || previous_background_status.is_none()
        || current_background_status.is_some()
    {
        return;
    }

    if let Ok(mut snapshot) = runtime.load_snapshot() {
        snapshot.status = None;
        app.apply_snapshot(snapshot);
    }
}

fn poll_context_updates(app: &mut TuiApp, runtime: &mut impl TuiRuntime) {
    let Ok(context_messages) = runtime.load_context_messages() else {
        return;
    };
    let first_unseen = context_messages
        .iter()
        .position(|message| !app.rendered_context_message_ids.contains(&message.id));
    let Some(first_unseen) = first_unseen else {
        return;
    };

    let trailing_start = context_messages
        .iter()
        .rposition(|message| app.rendered_context_message_ids.contains(&message.id))
        .map_or(0, |index| index + 1);
    let trailing_unseen = context_messages[trailing_start..]
        .iter()
        .filter(|message| !app.rendered_context_message_ids.contains(&message.id))
        .cloned()
        .collect::<Vec<_>>();

    if trailing_start <= first_unseen {
        if !trailing_unseen.is_empty() {
            app.render_new_context_messages(&trailing_unseen);
        }
        return;
    }

    if !trailing_unseen.is_empty() {
        app.render_new_context_messages(&trailing_unseen);
    }
    app.mark_context_messages_rendered(&context_messages[..trailing_start]);
}

fn maybe_poll_context_updates(
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
    context_poll_ready: bool,
    last_context_poll: &mut Instant,
    now: Instant,
) {
    if !context_poll_ready || app.prompt_active || app.command_active {
        return;
    }
    if now.duration_since(*last_context_poll) < Duration::from_secs(1) {
        return;
    }

    poll_context_updates(app, runtime);
    *last_context_poll = now;
}

fn apply_prompt_update(app: &mut TuiApp, update: PromptUpdate) {
    match update {
        PromptUpdate::AssistantDelta(delta) => {
            if let Some(last) = app.conversation_lines.last_mut()
                && last.starts_with("assistant: ")
            {
                last.push_str(&delta);
            } else {
                app.conversation_lines.push(format!("assistant: {delta}"));
            }
            if app.focus != FocusTarget::Command(CommandPane::Conversation) {
                app.follow_conversation_output = true;
            }
        }
        PromptUpdate::InternalThought(content) => {
            app.status = format!("thinking: {content}");
        }
        PromptUpdate::ToolCall {
            name,
            arguments_json,
        } => {
            app.conversation_lines
                .push(format!("tool requested: {name} {arguments_json}"));
            if app.focus != FocusTarget::Command(CommandPane::Conversation) {
                app.follow_conversation_output = true;
            }
        }
        PromptUpdate::ToolResult { content, is_error } => {
            let label = if is_error {
                "tool error"
            } else {
                "tool result"
            };
            app.conversation_lines.push(format!("{label}: {content}"));
            if app.focus != FocusTarget::Command(CommandPane::Conversation) {
                app.follow_conversation_output = true;
            }
        }
        PromptUpdate::Status(content) => {
            app.status = content;
        }
    }
}

fn key_event_token(key: KeyEvent) -> Option<&'static str> {
    match (key.code, key.modifiers) {
        (KeyCode::Enter, _) => Some("enter"),
        (KeyCode::Esc, _) => Some("escape"),
        (KeyCode::Up, _) => Some("up"),
        (KeyCode::Down, _) => Some("down"),
        (KeyCode::Tab, KeyModifiers::SHIFT) => Some("shift+tab"),
        (KeyCode::BackTab, _) => Some("shift+tab"),
        (KeyCode::Tab, _) => Some("tab"),
        (KeyCode::Char('m'), KeyModifiers::CONTROL) => Some("ctrl+m"),
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => Some("ctrl+a"),
        (KeyCode::Char('j'), KeyModifiers::NONE) => Some("j"),
        (KeyCode::Char('k'), KeyModifiers::NONE) => Some("k"),
        (KeyCode::Char('m'), KeyModifiers::NONE) => Some("m"),
        (KeyCode::Char('g'), KeyModifiers::NONE) => Some("g"),
        (KeyCode::Char('i'), KeyModifiers::NONE) => Some("i"),
        (KeyCode::Char('a'), KeyModifiers::NONE) => Some("a"),
        (KeyCode::Char('c'), KeyModifiers::NONE) => Some("c"),
        (KeyCode::Char('d'), KeyModifiers::NONE) => Some("d"),
        _ => None,
    }
}

impl TuiApp {
    pub fn bootstrap() -> Self {
        Self {
            title: "Elroy".to_string(),
            model_name: "gpt-5".to_string(),
            status: "bootstrap".to_string(),
            prompt_active: false,
            command_active: false,
            background_status: None,
            input: String::new(),
            input_completions: Vec::new(),
            input_history: Vec::new(),
            history_index: None,
            history_draft: None,
            command_palette: None,
            command_form: None,
            detail_modal: None,
            sidebar_section: SidebarSection::Memories,
            focus: FocusTarget::Input,
            last_command_pane: CommandPane::Conversation,
            conversation_lines: vec!["Conversation history and streaming output".to_string()],
            conversation_scroll: 0,
            follow_conversation_output: true,
            memory_titles: Vec::new(),
            agenda_titles: Vec::new(),
            improvement_titles: Vec::new(),
            feature_request_titles: Vec::new(),
            codex_session_titles: Vec::new(),
            selected_sidebar_index: 0,
            rendered_context_message_ids: HashSet::new(),
        }
    }

    pub fn from_snapshot(snapshot: TuiSnapshot) -> Self {
        let mut app = Self::bootstrap();
        app.apply_snapshot(snapshot);
        app
    }

    pub fn apply_snapshot(&mut self, snapshot: TuiSnapshot) {
        if !snapshot.conversation_lines.is_empty() {
            self.conversation_lines = snapshot.conversation_lines;
        }
        self.memory_titles = snapshot.memory_titles;
        self.agenda_titles = snapshot.agenda_titles;
        self.input_completions = snapshot.input_completions;
        self.improvement_titles = snapshot.improvement_titles;
        self.feature_request_titles = snapshot.feature_request_titles;
        self.codex_session_titles = snapshot.codex_session_titles;
        if let Some(model_name) = snapshot.model_name {
            self.model_name = model_name;
        }
        if let Some(status) = snapshot.status {
            self.status = status;
        }
        let active_len = self.active_sidebar_items().len();
        if self.selected_sidebar_index >= active_len {
            self.selected_sidebar_index = active_len.saturating_sub(1);
        }
    }

    pub fn mark_context_messages_rendered(&mut self, context_messages: &[TuiContextMessage]) {
        self.rendered_context_message_ids
            .extend(context_messages.iter().map(|message| message.id));
    }

    pub fn render_new_context_messages(&mut self, unseen_messages: &[TuiContextMessage]) {
        self.conversation_lines
            .extend(unseen_messages.iter().map(format_context_message_line));
        self.mark_context_messages_rendered(unseen_messages);
        if self.focus != FocusTarget::Command(CommandPane::Conversation) {
            self.follow_conversation_output = true;
        }
    }

    pub fn mark_messages_rendered_after_bootstrap_stream(
        &mut self,
        before_ids: &HashSet<i64>,
        context_messages: &[TuiContextMessage],
    ) {
        self.mark_trailing_context_messages_rendered(before_ids, context_messages, None, None);
    }

    pub fn mark_messages_rendered_after_chat_turn(
        &mut self,
        text: &str,
        before_ids: &HashSet<i64>,
        context_messages: &[TuiContextMessage],
    ) {
        self.mark_trailing_context_messages_rendered(
            before_ids,
            context_messages,
            Some("user"),
            Some(text),
        );
    }

    fn mark_trailing_context_messages_rendered(
        &mut self,
        before_ids: &HashSet<i64>,
        context_messages: &[TuiContextMessage],
        anchor_role: Option<&str>,
        anchor_content: Option<&str>,
    ) {
        let new_messages = context_messages
            .iter()
            .filter(|message| !before_ids.contains(&message.id))
            .cloned()
            .collect::<Vec<_>>();
        if new_messages.is_empty() {
            return;
        }

        let mut start_index = 0usize;
        if let (Some(anchor_role), Some(anchor_content)) = (anchor_role, anchor_content) {
            let Some(index) = new_messages.iter().rposition(|message| {
                message.role == anchor_role && message.content == anchor_content
            }) else {
                return;
            };
            start_index = index;
        }
        self.mark_context_messages_rendered(&new_messages[start_index..]);
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let input_height = self.input_box_height(area.width, area.height);
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(input_height),
                Constraint::Length(1),
            ])
            .split(area);
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(36)])
            .split(vertical[0]);
        let conversation_height = body[0].height.saturating_sub(2) as usize;

        Paragraph::new(self.conversation_lines.join("\n"))
            .block(
                Block::default()
                    .title(self.title.as_str())
                    .borders(Borders::ALL),
            )
            .scroll((
                self.effective_conversation_scroll(conversation_height) as u16,
                0,
            ))
            .render(body[0], buf);

        Paragraph::new(self.sidebar_text())
            .block(
                Block::default()
                    .title("Relevant Context")
                    .borders(Borders::ALL),
            )
            .render(body[1], buf);

        Paragraph::new(self.input.as_str())
            .block(Block::default().title("Input").borders(Borders::ALL))
            .render(vertical[1], buf);

        let footer = format!("{} | {}", self.footer_status_text(), self.footer_hints());
        Paragraph::new(Line::styled(
            footer,
            Style::default().add_modifier(Modifier::BOLD),
        ))
        .render(vertical[2], buf);

        if let Some(command_palette) = &self.command_palette {
            self.render_command_palette_modal(command_palette, area, buf);
        } else if let Some(command_form) = &self.command_form {
            self.render_command_form_modal(command_form, area, buf);
        } else if let Some(detail_modal) = &self.detail_modal {
            self.render_detail_modal(detail_modal, area, buf);
        }
    }

    fn sidebar_header_lines(&self) -> Vec<String> {
        match self.sidebar_section {
            SidebarSection::Memories => vec![
                "Memories [active] | Agenda".to_string(),
                "Improvements | Requests | Codex".to_string(),
            ],
            SidebarSection::Agenda => vec![
                "Memories | Agenda [active]".to_string(),
                "Improvements | Requests | Codex".to_string(),
            ],
            SidebarSection::Improvements => vec![
                "Memories | Agenda".to_string(),
                "Improvements [active] | Requests".to_string(),
                "Codex".to_string(),
            ],
            SidebarSection::FeatureRequests => vec![
                "Memories | Agenda".to_string(),
                "Improvements | Requests [active]".to_string(),
                "Codex".to_string(),
            ],
            SidebarSection::CodexSessions => vec![
                "Memories | Agenda".to_string(),
                "Improvements | Requests".to_string(),
                "Codex [active]".to_string(),
            ],
        }
    }

    fn input_box_height(&self, total_width: u16, total_height: u16) -> u16 {
        let available_width = total_width.saturating_sub(2).max(1) as usize;
        let wrapped_lines = wrapped_line_count(self.input.as_str(), available_width);
        let content_height = wrapped_lines.max(1) as u16;
        let desired = content_height.saturating_add(2).max(3);
        let max_height = total_height.saturating_sub(2).max(3);
        desired.min(max_height)
    }

    fn sidebar_text(&self) -> String {
        let mut lines = self.sidebar_header_lines();
        let items = self.active_sidebar_items();
        if items.is_empty() {
            lines.push("No entries loaded".to_string());
            return lines.join("\n");
        }

        for (index, item) in items.iter().enumerate() {
            let marker = if index == self.selected_sidebar_index {
                ">"
            } else {
                " "
            };
            lines.push(format!("{marker} {item}"));
        }
        lines.join("\n")
    }

    fn active_sidebar_items(&self) -> &[String] {
        match self.sidebar_section {
            SidebarSection::Memories => &self.memory_titles,
            SidebarSection::Agenda => &self.agenda_titles,
            SidebarSection::Improvements => &self.improvement_titles,
            SidebarSection::FeatureRequests => &self.feature_request_titles,
            SidebarSection::CodexSessions => &self.codex_session_titles,
        }
    }

    fn effective_conversation_scroll(&self, viewport_height: usize) -> usize {
        let max_scroll = self
            .conversation_lines
            .len()
            .saturating_sub(viewport_height.max(1));
        if self.follow_conversation_output {
            max_scroll
        } else {
            self.conversation_scroll.min(max_scroll)
        }
    }

    pub fn footer_hints(&self) -> &'static str {
        match self.focus {
            FocusTarget::Input => {
                "Esc command mode  Ctrl+C clear/cancel  Ctrl+M memories  Ctrl+A agenda  r improvements  f requests  s codex sessions  Ctrl+D quit"
            }
            FocusTarget::Command(_) => {
                "Tab switch pane  j/k move  Enter open  c complete  d archive/delete  i/a/Esc chat mode"
            }
            FocusTarget::Unknown => "Recovering focus",
        }
    }

    pub fn footer_status_text(&self) -> String {
        if self.prompt_active || self.command_active {
            return self.status.clone();
        }
        if let Some(background_status) = &self.background_status {
            return format!("● {}  ⟳ {}", self.model_name, background_status);
        }
        format!("● {}", self.model_name)
    }

    fn render_detail_modal(&self, detail_modal: &DetailModalState, area: Rect, buf: &mut Buffer) {
        let modal_area = centered_rect(area, 72, 60);
        Clear.render(modal_area, buf);
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(modal_area);
        Paragraph::new(detail_modal.content.as_str())
            .block(
                Block::default()
                    .title(detail_modal.title.as_str())
                    .borders(Borders::ALL),
            )
            .render(sections[0], buf);
        Paragraph::new(detail_modal.footer_text())
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
            .render(sections[1], buf);
    }

    fn render_command_form_modal(
        &self,
        command_form: &CommandFormState,
        area: Rect,
        buf: &mut Buffer,
    ) {
        let modal_area = centered_rect(area, 72, 60);
        Clear.render(modal_area, buf);
        let mut lines = vec![
            format!("/{}", command_form.command_name),
            command_form.description.clone(),
            String::new(),
        ];
        for (index, field) in command_form.fields.iter().enumerate() {
            let marker = if index == command_form.selected_field {
                ">"
            } else {
                " "
            };
            let optional = if field.optional { " (optional)" } else { "" };
            lines.push(format!(
                "{marker} {}{}: {}",
                field.name, optional, field.value
            ));
        }
        if let Some(error) = &command_form.error {
            lines.push(String::new());
            lines.push(format!("Error: {error}"));
        }
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(modal_area);
        Paragraph::new(lines.join("\n"))
            .block(Block::default().title("Command").borders(Borders::ALL))
            .render(sections[0], buf);
        Paragraph::new(command_form.footer_text())
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
            .render(sections[1], buf);
    }

    fn render_command_palette_modal(
        &self,
        command_palette: &CommandPaletteState,
        area: Rect,
        buf: &mut Buffer,
    ) {
        let modal_area = centered_rect(area, 72, 60);
        Clear.render(modal_area, buf);
        let mut lines = vec![format!("Search: {}", command_palette.query), String::new()];
        if command_palette.filtered_indices.is_empty() {
            lines.push("  No matching commands".to_string());
        }
        for (index, entry_index) in command_palette.filtered_indices.iter().enumerate() {
            let Some(entry) = command_palette.all_entries.get(*entry_index) else {
                continue;
            };
            let marker = if index == command_palette.selected_index {
                ">"
            } else {
                " "
            };
            lines.push(format!("{marker} {}  {}", entry.title, entry.description));
        }
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(modal_area);
        Paragraph::new(lines.join("\n"))
            .block(
                Block::default()
                    .title("Command Palette")
                    .borders(Borders::ALL),
            )
            .render(sections[0], buf);
        Paragraph::new(command_palette.footer_text())
            .block(Block::default().borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM))
            .render(sections[1], buf);
    }

    pub fn open_detail_modal(&mut self, detail: TuiSidebarDetail) {
        self.detail_modal = Some(DetailModalState {
            title: detail.title,
            content: detail.content,
            can_complete: detail.can_complete,
            destructive_action: detail.destructive_action,
            destructive_label: detail.destructive_label,
            confirming_destructive: false,
        });
    }

    pub fn open_command_form(&mut self, command_form: TuiCommandForm) {
        self.command_form = Some(CommandFormState::from(command_form));
    }

    pub fn open_command_palette(&mut self, mut entries: Vec<TuiCommandPaletteEntry>) {
        entries.insert(
            0,
            TuiCommandPaletteEntry {
                title: "Focus Memories".to_string(),
                description: "Switch the sidebar to memories".to_string(),
                action: TuiCommandPaletteAction::FocusMemories,
            },
        );
        entries.insert(
            1,
            TuiCommandPaletteEntry {
                title: "Focus Agenda".to_string(),
                description: "Switch the sidebar to agenda".to_string(),
                action: TuiCommandPaletteAction::FocusAgenda,
            },
        );
        entries.insert(
            2,
            TuiCommandPaletteEntry {
                title: "Focus Improvements".to_string(),
                description: "Switch the sidebar to improvements".to_string(),
                action: TuiCommandPaletteAction::FocusImprovements,
            },
        );
        entries.insert(
            3,
            TuiCommandPaletteEntry {
                title: "Focus Feature Requests".to_string(),
                description: "Switch the sidebar to feature requests".to_string(),
                action: TuiCommandPaletteAction::FocusFeatureRequests,
            },
        );
        entries.insert(
            4,
            TuiCommandPaletteEntry {
                title: "Focus Codex Sessions".to_string(),
                description: "Switch the sidebar to codex sessions".to_string(),
                action: TuiCommandPaletteAction::FocusCodexSessions,
            },
        );
        entries.insert(
            5,
            TuiCommandPaletteEntry {
                title: "Refresh System Instructions".to_string(),
                description: "Rebuild the system instructions for the current conversation"
                    .to_string(),
                action: TuiCommandPaletteAction::ToolCommand(
                    "refresh_system_instructions".to_string(),
                ),
            },
        );
        entries.insert(
            6,
            TuiCommandPaletteEntry {
                title: "Reset Messages".to_string(),
                description: "Clear the current conversation".to_string(),
                action: TuiCommandPaletteAction::ToolCommand("reset_messages".to_string()),
            },
        );
        let mut command_palette = CommandPaletteState {
            all_entries: entries,
            query: String::new(),
            filtered_indices: Vec::new(),
            selected_index: 0,
        };
        command_palette.apply_query();
        self.command_palette = Some(command_palette);
    }

    fn focus_sidebar_section(&mut self, section: SidebarSection) {
        self.sidebar_section = section;
        self.selected_sidebar_index = 0;
        self.focus = FocusTarget::Command(CommandPane::Sidebar);
        self.last_command_pane = CommandPane::Sidebar;
    }

    fn record_submitted_prompt(&mut self, submitted: &str) {
        self.reset_prompt_history_navigation();
        self.input_history.retain(|entry| entry != submitted);
        self.input_history.insert(0, submitted.to_string());
    }

    fn reset_prompt_history_navigation(&mut self) {
        self.history_index = None;
        self.history_draft = None;
    }

    fn recall_previous_prompt(&mut self) {
        if self.input_history.is_empty() {
            self.status = "prompt history previous".to_string();
            return;
        }

        let next_index = match self.history_index {
            Some(index) if index + 1 < self.input_history.len() => index + 1,
            Some(index) => index,
            None => {
                self.history_draft = Some(self.input.clone());
                0
            }
        };
        self.history_index = Some(next_index);
        self.input = self.input_history[next_index].clone();
        self.status = "prompt history previous".to_string();
    }

    fn recall_next_prompt(&mut self) {
        let Some(index) = self.history_index else {
            self.status = "prompt history next".to_string();
            return;
        };

        if index == 0 {
            self.history_index = None;
            self.input = self.history_draft.take().unwrap_or_default();
        } else {
            let next_index = index - 1;
            self.history_index = Some(next_index);
            self.input = self.input_history[next_index].clone();
        }
        self.status = "prompt history next".to_string();
    }

    fn accept_input_completion(&mut self) {
        let prefix = self.input.trim();
        if prefix.is_empty() {
            self.status = "input completion requested".to_string();
            return;
        }

        let matched_completion = self
            .input_completions
            .iter()
            .find(|suggestion| {
                suggestion.len() != prefix.len()
                    && suggestion
                        .to_lowercase()
                        .starts_with(&prefix.to_lowercase())
            })
            .cloned();

        if let Some(completion) = matched_completion {
            self.reset_prompt_history_navigation();
            self.input = completion;
        }
        self.status = "input completion requested".to_string();
    }

    pub fn handle_modal_key(&mut self, key: &str) -> UiIntent {
        let Some(detail_modal) = self.detail_modal.as_mut() else {
            return UiIntent::Noop;
        };

        match key {
            "escape" | "enter" | "q" => {
                self.detail_modal = None;
                UiIntent::Noop
            }
            "c" if detail_modal.can_complete => UiIntent::CompleteSelected,
            "d" => {
                let Some(action) = detail_modal.destructive_action else {
                    return UiIntent::Noop;
                };
                if detail_modal.confirming_destructive {
                    return match action {
                        SidebarAction::Archive => UiIntent::ArchiveSelected,
                        SidebarAction::Delete => UiIntent::DeleteSelected,
                        SidebarAction::Complete => UiIntent::CompleteSelected,
                    };
                }
                detail_modal.confirming_destructive = true;
                UiIntent::Noop
            }
            _ => {
                if detail_modal.confirming_destructive {
                    detail_modal.confirming_destructive = false;
                }
                UiIntent::Noop
            }
        }
    }

    pub fn handle_key(&mut self, key: &str) -> UiIntent {
        if self.focus == FocusTarget::Unknown {
            self.focus = FocusTarget::Input;
            return UiIntent::Noop;
        }

        match key {
            "ctrl+m" => {
                self.focus_sidebar_section(SidebarSection::Memories);
                UiIntent::Noop
            }
            "ctrl+a" => {
                self.focus_sidebar_section(SidebarSection::Agenda);
                UiIntent::Noop
            }
            "r" => {
                self.focus_sidebar_section(SidebarSection::Improvements);
                UiIntent::Noop
            }
            "f" => {
                self.focus_sidebar_section(SidebarSection::FeatureRequests);
                UiIntent::Noop
            }
            "s" => {
                self.focus_sidebar_section(SidebarSection::CodexSessions);
                UiIntent::Noop
            }
            "escape" => self.handle_escape(),
            _ => match self.focus {
                FocusTarget::Input => self.handle_chat_key(key),
                FocusTarget::Command(_) => self.handle_command_key(key),
                FocusTarget::Unknown => UiIntent::Noop,
            },
        }
    }

    pub fn apply_intent(&mut self, intent: UiIntent) {
        match intent {
            UiIntent::SubmitPrompt => {
                let submitted = self.input.trim().to_string();
                if submitted.is_empty() {
                    self.status = "prompt was empty".to_string();
                } else {
                    self.status = format!("submitted prompt: {submitted}");
                    self.input.clear();
                }
            }
            UiIntent::CancelPrompt => {
                self.status = "cancel requested".to_string();
            }
            UiIntent::HistoryPrevious => {
                self.recall_previous_prompt();
            }
            UiIntent::HistoryNext => {
                self.recall_next_prompt();
            }
            UiIntent::CompleteInput => {
                self.accept_input_completion();
            }
            UiIntent::MoveUp => {
                if self.focus == FocusTarget::Command(CommandPane::Conversation) {
                    self.scroll_conversation_up();
                } else {
                    if self.selected_sidebar_index > 0 {
                        self.selected_sidebar_index -= 1;
                    }
                    self.status = "moved selection up".to_string();
                }
            }
            UiIntent::MoveDown => {
                if self.focus == FocusTarget::Command(CommandPane::Conversation) {
                    self.scroll_conversation_down();
                } else {
                    let len = self.active_sidebar_items().len();
                    if self.selected_sidebar_index + 1 < len {
                        self.selected_sidebar_index += 1;
                    }
                    self.status = "moved selection down".to_string();
                }
            }
            UiIntent::OpenSelected => {
                let label = self
                    .active_sidebar_items()
                    .get(self.selected_sidebar_index)
                    .cloned()
                    .unwrap_or_else(|| "unknown entry".to_string());
                self.status = format!("open selected {label}");
            }
            UiIntent::ArchiveSelected | UiIntent::CompleteSelected | UiIntent::DeleteSelected => {
                self.status = "sidebar mutation requested".to_string();
            }
            UiIntent::Noop => {}
        }
    }

    fn handle_escape(&mut self) -> UiIntent {
        match self.focus {
            FocusTarget::Input => {
                self.focus = FocusTarget::Command(self.last_command_pane);
                UiIntent::Noop
            }
            FocusTarget::Command(pane) => {
                self.last_command_pane = pane;
                self.focus = FocusTarget::Input;
                self.follow_conversation_output = true;
                UiIntent::Noop
            }
            FocusTarget::Unknown => {
                self.focus = FocusTarget::Input;
                self.follow_conversation_output = true;
                UiIntent::Noop
            }
        }
    }

    fn handle_chat_key(&mut self, key: &str) -> UiIntent {
        match key {
            "enter" => UiIntent::SubmitPrompt,
            "up" => UiIntent::HistoryPrevious,
            "down" => UiIntent::HistoryNext,
            "tab" => UiIntent::CompleteInput,
            _ => UiIntent::Noop,
        }
    }

    fn handle_command_key(&mut self, key: &str) -> UiIntent {
        match key {
            "j" | "down" => UiIntent::MoveDown,
            "k" | "up" => UiIntent::MoveUp,
            "tab" | "shift+tab" => {
                self.toggle_command_pane();
                UiIntent::Noop
            }
            "m" => {
                self.focus_sidebar_section(SidebarSection::Memories);
                UiIntent::Noop
            }
            "g" => {
                self.focus_sidebar_section(SidebarSection::Agenda);
                UiIntent::Noop
            }
            "r" => {
                self.focus_sidebar_section(SidebarSection::Improvements);
                UiIntent::Noop
            }
            "f" => {
                self.focus_sidebar_section(SidebarSection::FeatureRequests);
                UiIntent::Noop
            }
            "s" => {
                self.focus_sidebar_section(SidebarSection::CodexSessions);
                UiIntent::Noop
            }
            "left" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    self.move_sidebar_section(-1);
                }
                UiIntent::Noop
            }
            "right" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    self.move_sidebar_section(1);
                }
                UiIntent::Noop
            }
            "enter" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    UiIntent::OpenSelected
                } else {
                    UiIntent::Noop
                }
            }
            "c" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    match self.sidebar_section {
                        SidebarSection::Agenda
                        | SidebarSection::Improvements
                        | SidebarSection::FeatureRequests => UiIntent::CompleteSelected,
                        SidebarSection::Memories | SidebarSection::CodexSessions => UiIntent::Noop,
                    }
                } else {
                    UiIntent::Noop
                }
            }
            "d" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    match self.sidebar_section {
                        SidebarSection::Memories => UiIntent::ArchiveSelected,
                        SidebarSection::Agenda => UiIntent::DeleteSelected,
                        SidebarSection::Improvements | SidebarSection::FeatureRequests => {
                            UiIntent::Noop
                        }
                        SidebarSection::CodexSessions => UiIntent::Noop,
                    }
                } else {
                    UiIntent::Noop
                }
            }
            "i" | "a" => {
                if let FocusTarget::Command(pane) = self.focus {
                    self.last_command_pane = pane;
                }
                self.focus = FocusTarget::Input;
                UiIntent::Noop
            }
            _ => UiIntent::Noop,
        }
    }

    fn toggle_command_pane(&mut self) {
        self.focus = match self.focus {
            FocusTarget::Command(CommandPane::Conversation) => {
                self.last_command_pane = CommandPane::Sidebar;
                FocusTarget::Command(CommandPane::Sidebar)
            }
            FocusTarget::Command(CommandPane::Sidebar) => {
                self.last_command_pane = CommandPane::Conversation;
                FocusTarget::Command(CommandPane::Conversation)
            }
            _ => FocusTarget::Command(self.last_command_pane),
        };
    }

    fn move_sidebar_section(&mut self, delta: isize) {
        const ORDER: [SidebarSection; 5] = [
            SidebarSection::Memories,
            SidebarSection::Agenda,
            SidebarSection::Improvements,
            SidebarSection::FeatureRequests,
            SidebarSection::CodexSessions,
        ];
        let current_index = ORDER
            .iter()
            .position(|section| *section == self.sidebar_section)
            .unwrap_or(0) as isize;
        let next_index = (current_index + delta).rem_euclid(ORDER.len() as isize) as usize;
        self.focus_sidebar_section(ORDER[next_index]);
    }

    fn can_mouse_scroll_conversation(&self) -> bool {
        matches!(
            self.focus,
            FocusTarget::Input | FocusTarget::Command(CommandPane::Conversation)
        ) && self.command_palette.is_none()
            && self.command_form.is_none()
            && self.detail_modal.is_none()
    }

    fn scroll_conversation_up(&mut self) {
        self.follow_conversation_output = false;
        self.conversation_scroll = self.conversation_scroll.saturating_sub(1);
        self.status = "scrolled conversation up".to_string();
    }

    fn scroll_conversation_down(&mut self) {
        let max_scroll = self.conversation_lines.len().saturating_sub(1);
        self.follow_conversation_output = false;
        self.conversation_scroll = self.conversation_scroll.saturating_add(1).min(max_scroll);
        self.status = "scrolled conversation down".to_string();
    }
}

impl DetailModalState {
    fn footer_text(&self) -> String {
        if self.confirming_destructive {
            return "Press D again to confirm deletion, any other key to cancel".to_string();
        }

        let mut actions = Vec::new();
        if self.can_complete {
            actions.push("C: complete".to_string());
        }
        if let Some(label) = &self.destructive_label {
            actions.push(format!("D: {label}"));
        }
        actions.push("Escape/Enter/Q: close".to_string());
        actions.join("  |  ")
    }
}

impl From<TuiCommandForm> for CommandFormState {
    fn from(value: TuiCommandForm) -> Self {
        let fields = value
            .parameters
            .into_iter()
            .map(|parameter| CommandFormFieldState {
                value: value
                    .initial_values
                    .iter()
                    .find(|(name, _)| *name == parameter.name)
                    .map(|(_, field_value)| field_value.clone())
                    .unwrap_or(parameter.default_text),
                name: parameter.name,
                optional: parameter.optional,
                suggestions: parameter.suggestions,
            })
            .collect::<Vec<_>>();
        let selected_field = fields
            .iter()
            .position(|field| !field.optional && field.value.trim().is_empty())
            .unwrap_or(0);
        Self {
            command_name: value.command_name,
            description: value.description,
            fields,
            selected_field,
            error: None,
        }
    }
}

impl CommandFormState {
    fn footer_text(&self) -> String {
        "Type to edit  Tab complete/move  Shift+Tab move  Enter run  Escape cancel".to_string()
    }

    fn move_selection(&mut self, delta: isize) {
        if self.fields.is_empty() {
            return;
        }
        let len = self.fields.len() as isize;
        let current = self.selected_field as isize;
        self.selected_field = (current + delta).rem_euclid(len) as usize;
        self.error = None;
    }

    fn selected_field_mut(&mut self) -> Option<&mut CommandFormFieldState> {
        self.fields.get_mut(self.selected_field)
    }

    fn accept_selected_field_suggestion(&mut self) -> bool {
        let Some(field) = self.selected_field_mut() else {
            return false;
        };
        let prefix = field.value.trim();
        let Some(suggestion) = field.suggestions.iter().find(|suggestion| {
            suggestion.len() != prefix.len()
                && suggestion
                    .to_ascii_lowercase()
                    .starts_with(&prefix.to_ascii_lowercase())
        }) else {
            return false;
        };
        field.value = suggestion.clone();
        self.error = None;
        true
    }
}

impl CommandPaletteState {
    fn footer_text(&self) -> String {
        "Type to filter  Up/Down move  Enter select  Backspace edit  Escape close".to_string()
    }

    fn move_selection(&mut self, delta: isize) {
        if self.filtered_indices.is_empty() {
            return;
        }
        let len = self.filtered_indices.len() as isize;
        let current = self.selected_index as isize;
        self.selected_index = (current + delta).rem_euclid(len) as usize;
    }

    fn apply_query(&mut self) {
        let query = self.query.to_ascii_lowercase();
        self.filtered_indices = self
            .all_entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let haystack =
                    format!("{} {}", entry.title, entry.description).to_ascii_lowercase();
                haystack.contains(&query).then_some(index)
            })
            .collect();
        if self.filtered_indices.is_empty() || self.selected_index >= self.filtered_indices.len() {
            self.selected_index = 0;
        }
    }

    fn append_query(&mut self, ch: char) {
        self.query.push(ch);
        self.apply_query();
    }

    fn backspace_query(&mut self) {
        self.query.pop();
        self.apply_query();
    }

    fn selected_entry(&self) -> Option<TuiCommandPaletteEntry> {
        let index = *self.filtered_indices.get(self.selected_index)?;
        self.all_entries.get(index).cloned()
    }
}

fn handle_command_palette_key(app: &mut TuiApp, key: KeyEvent, runtime: &mut impl TuiRuntime) {
    let Some(command_palette) = app.command_palette.as_mut() else {
        return;
    };

    match key.code {
        KeyCode::Esc => {
            app.command_palette = None;
        }
        KeyCode::Backspace => {
            command_palette.backspace_query();
        }
        KeyCode::Up => {
            command_palette.move_selection(-1);
        }
        KeyCode::Down | KeyCode::Tab => {
            command_palette.move_selection(1);
        }
        KeyCode::BackTab => {
            command_palette.move_selection(-1);
        }
        KeyCode::Enter => {
            let Some(entry) = command_palette.selected_entry() else {
                app.status = "no matching command".to_string();
                return;
            };
            match entry.action {
                TuiCommandPaletteAction::FocusMemories => {
                    app.command_palette = None;
                    app.focus_sidebar_section(SidebarSection::Memories);
                    app.status = "focused memories".to_string();
                }
                TuiCommandPaletteAction::FocusAgenda => {
                    app.command_palette = None;
                    app.focus_sidebar_section(SidebarSection::Agenda);
                    app.status = "focused agenda".to_string();
                }
                TuiCommandPaletteAction::FocusImprovements => {
                    app.command_palette = None;
                    app.focus_sidebar_section(SidebarSection::Improvements);
                    app.status = "focused improvements".to_string();
                }
                TuiCommandPaletteAction::FocusFeatureRequests => {
                    app.command_palette = None;
                    app.focus_sidebar_section(SidebarSection::FeatureRequests);
                    app.status = "focused feature requests".to_string();
                }
                TuiCommandPaletteAction::FocusCodexSessions => {
                    app.command_palette = None;
                    app.focus_sidebar_section(SidebarSection::CodexSessions);
                    app.status = "focused codex sessions".to_string();
                }
                TuiCommandPaletteAction::ToolCommand(name) => {
                    match runtime.launch_named_command(&name) {
                        Ok(TuiSlashCommandAction::Execute(command)) => {
                            match start_command_execution(app, runtime, command) {
                                Ok(()) => {
                                    app.command_palette = None;
                                }
                                Err(error) => {
                                    app.status = format!("command launch failed: {error}");
                                }
                            }
                        }
                        Ok(TuiSlashCommandAction::OpenForm(form)) => {
                            app.command_palette = None;
                            app.open_command_form(form);
                            app.status = format!(
                                "editing command: /{}",
                                app.command_form
                                    .as_ref()
                                    .expect("command form should open")
                                    .command_name
                            );
                        }
                        Ok(TuiSlashCommandAction::NotHandled) => {
                            app.status = format!("command launch failed: {name}");
                        }
                        Err(error) => {
                            app.status = format!("command launch failed: {error}");
                        }
                    }
                }
            }
        }
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            command_palette.append_query(ch);
        }
        _ => {}
    }
}

fn handle_command_form_key(app: &mut TuiApp, key: KeyEvent, runtime: &mut impl TuiRuntime) {
    let Some(command_form) = app.command_form.as_mut() else {
        return;
    };

    match (key.code, key.modifiers) {
        (KeyCode::Esc, _) => {
            app.command_form = None;
        }
        (KeyCode::Tab, KeyModifiers::SHIFT) | (KeyCode::BackTab, _) => {
            command_form.move_selection(-1);
        }
        (KeyCode::Tab, _) => {
            if !command_form.accept_selected_field_suggestion() {
                command_form.move_selection(1);
            }
        }
        (KeyCode::Down, _) => {
            command_form.move_selection(1);
        }
        (KeyCode::Up, _) => {
            command_form.move_selection(-1);
        }
        (KeyCode::Backspace, _) => {
            if let Some(field) = command_form.selected_field_mut() {
                field.value.pop();
                command_form.error = None;
            }
        }
        (KeyCode::Char(ch), modifiers) if !modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(field) = command_form.selected_field_mut() {
                field.value.push(ch);
                command_form.error = None;
            }
        }
        (KeyCode::Enter, _) => {
            if let Some(field) = command_form
                .fields
                .iter()
                .find(|field| !field.optional && field.value.trim().is_empty())
            {
                command_form.error = Some(format!("Missing required value for '{}'", field.name));
                command_form.selected_field = command_form
                    .fields
                    .iter()
                    .position(|candidate| candidate.name == field.name)
                    .unwrap_or(command_form.selected_field);
                return;
            }
            let command_name = command_form.command_name.clone();
            let values = command_form
                .fields
                .iter()
                .map(|field| (field.name.clone(), field.value.trim().to_string()))
                .collect::<Vec<_>>();
            let command = TuiCommandExecution {
                command_name: command_name.clone(),
                display_name: command_name,
                values,
            };
            match start_command_execution(app, runtime, command) {
                Ok(()) => {
                    app.command_form = None;
                }
                Err(error) => {
                    if let Some(form) = app.command_form.as_mut() {
                        form.error = Some(error);
                    }
                }
            }
        }
        _ => {}
    }
}

fn centered_rect(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn modal_key_token(key: KeyEvent) -> Option<&'static str> {
    match key.code {
        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => match ch {
            'c' => Some("c"),
            'd' => Some("d"),
            'q' => Some("q"),
            _ => Some("other"),
        },
        _ => key_event_token(key),
    }
}

fn format_context_message_line(message: &TuiContextMessage) -> String {
    format!("{}: {}", message.role, message.content)
}

struct NoopRuntime;

struct NoopPromptStream;

impl TuiPromptStream for NoopPromptStream {
    fn next_update(&mut self) -> Result<Option<PromptUpdate>, String> {
        Ok(None)
    }

    fn finalize(self: Box<Self>) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }

    fn cancel(self: Box<Self>) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }
}

impl TuiRuntime for NoopRuntime {
    fn load_snapshot(&mut self) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }

    fn load_command_palette_entries(&mut self) -> Result<Vec<TuiCommandPaletteEntry>, String> {
        Ok(vec![])
    }

    fn launch_named_command(&mut self, _name: &str) -> Result<TuiSlashCommandAction, String> {
        Ok(TuiSlashCommandAction::NotHandled)
    }

    fn handle_slash_command(&mut self, _prompt: &str) -> Result<TuiSlashCommandAction, String> {
        Ok(TuiSlashCommandAction::NotHandled)
    }

    fn start_command_execution(&mut self, _command: TuiCommandExecution) -> Result<(), String> {
        Ok(())
    }

    fn poll_command_execution(&mut self) -> Result<Option<TuiSnapshot>, String> {
        Ok(None)
    }

    fn submit_prompt(&mut self, _prompt: &str) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }

    fn start_prompt_stream(&mut self, _prompt: &str) -> Result<Box<dyn TuiPromptStream>, String> {
        Ok(Box::new(NoopPromptStream))
    }

    fn start_startup_prompt_stream(&mut self) -> Result<Option<Box<dyn TuiPromptStream>>, String> {
        Ok(None)
    }

    fn start_restart_prompt_stream(
        &mut self,
        _resume_message: &str,
    ) -> Result<Box<dyn TuiPromptStream>, String> {
        Err("restart not available".to_string())
    }

    fn take_restart_request(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }

    fn load_context_messages(&mut self) -> Result<Vec<TuiContextMessage>, String> {
        Ok(vec![])
    }

    fn refresh_context_if_needed(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn run_self_reflection_if_needed(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn background_status(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }

    fn open_sidebar_item(
        &mut self,
        _section: SidebarSection,
        _title: &str,
    ) -> Result<TuiSidebarDetail, String> {
        Ok(TuiSidebarDetail {
            title: "sidebar detail".to_string(),
            content: "sidebar detail unavailable".to_string(),
            can_complete: false,
            destructive_action: None,
            destructive_label: None,
        })
    }

    fn mutate_sidebar_item(
        &mut self,
        _section: SidebarSection,
        _title: &str,
        _action: SidebarAction,
    ) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    use super::{
        CommandFormState, CommandPane, FocusTarget, PendingPrompt, PromptAdvance, PromptUpdate,
        SidebarAction, SidebarSection, TuiApp, TuiCommandExecution, TuiCommandForm,
        TuiCommandPaletteAction, TuiCommandPaletteEntry, TuiCommandParameter, TuiContextMessage,
        TuiExit, TuiPromptStream, TuiRunResult, TuiRuntime, TuiSidebarDetail,
        TuiSlashCommandAction, TuiSnapshot, UiIntent, advance_prompt_stream,
        apply_intent_with_runtime, apply_key_event, apply_mouse_event, apply_paste_event,
        drive_runtime_tick, key_event_token, maybe_complete_command_execution,
        maybe_enable_context_polling_after_prompt_completion, maybe_poll_context_updates,
        maybe_refresh_snapshot_after_background_completion, maybe_run_deferred_context_refresh,
        poll_context_updates, start_startup_prompt_stream,
    };

    fn select_command_palette_entry(app: &mut TuiApp, title: &str) {
        let palette = app
            .command_palette
            .as_mut()
            .expect("command palette should open");
        let index = palette
            .filtered_indices
            .iter()
            .position(|entry_index| palette.all_entries[*entry_index].title == title)
            .expect("command palette entry should exist");
        palette.selected_index = index;
    }

    #[derive(Default)]
    struct FakeRuntime {
        snapshot: TuiSnapshot,
        command_palette_entries: Vec<TuiCommandPaletteEntry>,
        launch_named_command_action: Option<TuiSlashCommandAction>,
        slash_command_action: Option<TuiSlashCommandAction>,
        slash_command_error: Option<String>,
        started_command_executions: Vec<TuiCommandExecution>,
        start_command_execution_error: Option<String>,
        completed_command_execution_snapshot: Option<TuiSnapshot>,
        command_execution_error: Option<String>,
        submitted_prompts: Vec<String>,
        self_reflection_runs: usize,
        last_opened: Option<(SidebarSection, String)>,
        last_mutation: Option<(SidebarSection, String, SidebarAction)>,
        startup_stream: Option<FakePromptStream>,
        restart_stream: Option<FakePromptStream>,
        pending_restart_request: Option<String>,
        context_messages: Vec<TuiContextMessage>,
        background_status: Option<String>,
        refresh_context_calls: usize,
    }

    struct FakePromptStream {
        updates: Vec<PromptUpdate>,
        finalized_snapshot: TuiSnapshot,
        cancelled_snapshot: TuiSnapshot,
    }

    impl TuiPromptStream for FakePromptStream {
        fn next_update(&mut self) -> Result<Option<PromptUpdate>, String> {
            if self.updates.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.updates.remove(0)))
            }
        }

        fn finalize(self: Box<Self>) -> Result<TuiSnapshot, String> {
            Ok(self.finalized_snapshot)
        }

        fn cancel(self: Box<Self>) -> Result<TuiSnapshot, String> {
            Ok(self.cancelled_snapshot)
        }
    }

    impl TuiRuntime for FakeRuntime {
        fn load_snapshot(&mut self) -> Result<TuiSnapshot, String> {
            Ok(self.snapshot.clone())
        }

        fn load_command_palette_entries(&mut self) -> Result<Vec<TuiCommandPaletteEntry>, String> {
            Ok(self.command_palette_entries.clone())
        }

        fn launch_named_command(&mut self, _name: &str) -> Result<TuiSlashCommandAction, String> {
            Ok(self
                .launch_named_command_action
                .clone()
                .unwrap_or(TuiSlashCommandAction::NotHandled))
        }

        fn handle_slash_command(&mut self, prompt: &str) -> Result<TuiSlashCommandAction, String> {
            if prompt.starts_with('/') {
                if let Some(error) = &self.slash_command_error {
                    return Err(error.clone());
                }
                Ok(self
                    .slash_command_action
                    .clone()
                    .unwrap_or(TuiSlashCommandAction::NotHandled))
            } else {
                Ok(TuiSlashCommandAction::NotHandled)
            }
        }

        fn start_command_execution(&mut self, command: TuiCommandExecution) -> Result<(), String> {
            if let Some(error) = self.start_command_execution_error.clone() {
                return Err(error);
            }
            self.started_command_executions.push(command);
            Ok(())
        }

        fn poll_command_execution(&mut self) -> Result<Option<TuiSnapshot>, String> {
            if let Some(error) = self.command_execution_error.take() {
                return Err(error);
            }
            Ok(self.completed_command_execution_snapshot.take())
        }

        fn submit_prompt(&mut self, prompt: &str) -> Result<TuiSnapshot, String> {
            self.submitted_prompts.push(prompt.to_string());
            Ok(TuiSnapshot {
                conversation_lines: vec![
                    format!("user: {prompt}"),
                    "assistant: runtime response".to_string(),
                ],
                memory_titles: vec!["Fresh Memory".to_string()],
                agenda_titles: vec!["Fresh Agenda".to_string()],
                improvement_titles: vec!["Fresh Improvement".to_string()],
                feature_request_titles: vec!["Fresh Request".to_string()],
                codex_session_titles: vec!["Fresh Session".to_string()],
                model_name: None,
                status: Some("runtime updated".to_string()),
                ..TuiSnapshot::default()
            })
        }

        fn start_prompt_stream(
            &mut self,
            prompt: &str,
        ) -> Result<Box<dyn TuiPromptStream>, String> {
            self.submitted_prompts.push(prompt.to_string());
            Ok(Box::new(FakePromptStream {
                updates: vec![
                    PromptUpdate::Status("thinking...".to_string()),
                    PromptUpdate::AssistantDelta("runtime ".to_string()),
                    PromptUpdate::AssistantDelta("response".to_string()),
                ],
                finalized_snapshot: TuiSnapshot {
                    conversation_lines: vec![
                        format!("user: {prompt}"),
                        "assistant: runtime response".to_string(),
                    ],
                    memory_titles: vec!["Fresh Memory".to_string()],
                    agenda_titles: vec!["Fresh Agenda".to_string()],
                    improvement_titles: vec!["Fresh Improvement".to_string()],
                    feature_request_titles: vec!["Fresh Request".to_string()],
                    codex_session_titles: vec!["Fresh Session".to_string()],
                    model_name: None,
                    status: Some("runtime updated".to_string()),
                    ..TuiSnapshot::default()
                },
                cancelled_snapshot: TuiSnapshot {
                    conversation_lines: vec!["assistant: cancelled".to_string()],
                    status: Some("cancelled".to_string()),
                    ..TuiSnapshot::default()
                },
            }))
        }

        fn start_startup_prompt_stream(
            &mut self,
        ) -> Result<Option<Box<dyn TuiPromptStream>>, String> {
            Ok(self
                .startup_stream
                .take()
                .map(|stream| Box::new(stream) as Box<dyn TuiPromptStream>))
        }

        fn start_restart_prompt_stream(
            &mut self,
            _resume_message: &str,
        ) -> Result<Box<dyn TuiPromptStream>, String> {
            self.restart_stream
                .take()
                .map(|stream| Box::new(stream) as Box<dyn TuiPromptStream>)
                .ok_or_else(|| "restart stream missing".to_string())
        }

        fn take_restart_request(&mut self) -> Result<Option<String>, String> {
            Ok(self.pending_restart_request.take())
        }

        fn load_context_messages(&mut self) -> Result<Vec<TuiContextMessage>, String> {
            Ok(self.context_messages.clone())
        }

        fn refresh_context_if_needed(&mut self) -> Result<(), String> {
            self.refresh_context_calls += 1;
            Ok(())
        }

        fn run_self_reflection_if_needed(&mut self) -> Result<(), String> {
            self.self_reflection_runs += 1;
            Ok(())
        }

        fn background_status(&mut self) -> Result<Option<String>, String> {
            Ok(self.background_status.clone())
        }

        fn open_sidebar_item(
            &mut self,
            section: SidebarSection,
            title: &str,
        ) -> Result<TuiSidebarDetail, String> {
            self.last_opened = Some((section, title.to_string()));
            let (can_complete, destructive_action, destructive_label) = match section {
                SidebarSection::Memories => (
                    false,
                    Some(SidebarAction::Archive),
                    Some("archive".to_string()),
                ),
                SidebarSection::Agenda => (
                    true,
                    Some(SidebarAction::Delete),
                    Some("delete".to_string()),
                ),
                SidebarSection::Improvements | SidebarSection::FeatureRequests => {
                    (true, None, None)
                }
                SidebarSection::CodexSessions => (false, None, None),
            };
            let content = if section == SidebarSection::CodexSessions {
                "Status: completed\nRepo: /tmp/sample\n\nSummary:\nCodex inspected the parser.\n\nLatest Agent Message:\nParser inspection complete."
                    .to_string()
            } else {
                format!("opened detail for {title}")
            };
            Ok(TuiSidebarDetail {
                title: title.to_string(),
                content,
                can_complete,
                destructive_action,
                destructive_label,
            })
        }

        fn mutate_sidebar_item(
            &mut self,
            section: SidebarSection,
            title: &str,
            action: SidebarAction,
        ) -> Result<TuiSnapshot, String> {
            self.last_mutation = Some((section, title.to_string(), action));
            Ok(TuiSnapshot {
                conversation_lines: vec!["assistant: refreshed".to_string()],
                memory_titles: vec!["After Mutation".to_string()],
                agenda_titles: vec!["Agenda After Mutation".to_string()],
                improvement_titles: vec!["Improvement After Mutation".to_string()],
                feature_request_titles: vec!["Request After Mutation".to_string()],
                codex_session_titles: vec!["Session After Mutation".to_string()],
                model_name: None,
                status: Some("mutation updated".to_string()),
                ..TuiSnapshot::default()
            })
        }
    }

    fn rendered_text(app: &TuiApp) -> String {
        let area = Rect::new(0, 0, 80, 20);
        let mut buf = Buffer::empty(area);
        app.render(area, &mut buf);

        buf.content
            .chunks(area.width as usize)
            .map(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn bootstrap_state_defaults_to_memories_sidebar() {
        let app = TuiApp::bootstrap();

        assert_eq!(app.sidebar_section, SidebarSection::Memories);
        assert_eq!(app.status, "bootstrap");
        assert_eq!(app.focus, FocusTarget::Input);
        assert_eq!(app.model_name, "gpt-5");
    }

    #[test]
    fn render_contains_core_elroy_panes() {
        let app = TuiApp::bootstrap();
        let text = rendered_text(&app);

        assert!(text.contains("Elroy"));
        assert!(text.contains("Relevant Context"));
        assert!(text.contains("Memories [active] | Agenda"));
        assert!(text.contains("Improvements | Requests | Codex"));
        assert!(text.contains("Input"));
        assert!(text.contains("● gpt-5"));
        assert!(text.contains("Esc command mode"));
    }

    #[test]
    fn input_box_height_grows_for_wrapped_text() {
        let mut app = TuiApp::bootstrap();
        let initial = app.input_box_height(50, 20);

        app.input = "This is a long message that should wrap in the composer instead of scrolling horizontally off screen.".to_string();

        assert!(app.input_box_height(50, 20) > initial);
    }

    #[test]
    fn input_box_height_keeps_body_visible_on_short_terminal() {
        let mut app = TuiApp::bootstrap();
        app.input = "x".repeat(400);

        assert!(app.input_box_height(40, 8) <= 6);
    }

    #[test]
    fn render_switches_sidebar_label_when_agenda_is_active() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Agenda;
        let text = rendered_text(&app);

        assert!(text.contains("Memories | Agenda [active]"));
        assert!(text.contains("Improvements | Requests | Codex"));
    }

    #[test]
    fn render_switches_sidebar_label_when_improvements_are_active() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Improvements;
        let text = rendered_text(&app);

        assert!(text.contains("Memories | Agenda"));
        assert!(text.contains("Improvements [active] | Requests"));
        assert!(text.contains("Codex"));
    }

    #[test]
    fn render_switches_sidebar_label_when_feature_requests_are_active() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::FeatureRequests;
        let text = rendered_text(&app);

        assert!(text.contains("Memories | Agenda"));
        assert!(text.contains("Improvements | Requests [active]"));
        assert!(text.contains("Codex"));
    }

    #[test]
    fn render_switches_sidebar_label_when_codex_sessions_are_active() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::CodexSessions;
        let text = rendered_text(&app);

        assert!(text.contains("Memories | Agenda"));
        assert!(text.contains("Improvements | Requests"));
        assert!(text.contains("Codex [active]"));
    }

    #[test]
    fn snapshot_render_uses_persisted_conversation_and_sidebar_data() {
        let app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["user: hello".to_string(), "assistant: hi".to_string()],
            memory_titles: vec!["Runner Notes".to_string()],
            agenda_titles: vec!["Doctor Visit".to_string()],
            improvement_titles: vec!["Improve correction handling (open)".to_string()],
            feature_request_titles: vec!["General export feature (open)".to_string()],
            codex_session_titles: vec!["sample (completed) thread-123".to_string()],
            model_name: Some("gpt-test".to_string()),
            status: Some("loaded snapshot".to_string()),
            ..TuiSnapshot::default()
        });
        let text = rendered_text(&app);

        assert!(text.contains("user: hello"));
        assert!(text.contains("assistant: hi"));
        assert!(text.contains("> Runner Notes"));
        assert!(text.contains("● gpt-test"));
    }

    #[test]
    fn escape_toggles_between_chat_and_last_command_pane() {
        let mut app = TuiApp::bootstrap();

        assert_eq!(app.handle_key("escape"), UiIntent::Noop);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Conversation));
        assert_eq!(app.handle_key("escape"), UiIntent::Noop);
        assert_eq!(app.focus, FocusTarget::Input);
    }

    #[test]
    fn global_sidebar_shortcuts_focus_sidebar_and_switch_section() {
        let mut app = TuiApp::bootstrap();

        app.handle_key("ctrl+a");
        assert_eq!(app.sidebar_section, SidebarSection::Agenda);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));

        app.handle_key("ctrl+m");
        assert_eq!(app.sidebar_section, SidebarSection::Memories);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));

        app.handle_key("r");
        assert_eq!(app.sidebar_section, SidebarSection::Improvements);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));

        app.handle_key("f");
        assert_eq!(app.sidebar_section, SidebarSection::FeatureRequests);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));

        app.handle_key("s");
        assert_eq!(app.sidebar_section, SidebarSection::CodexSessions);
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
    }

    #[test]
    fn sidebar_left_right_switch_sections_in_order() {
        let mut app = TuiApp::bootstrap();
        app.handle_key("ctrl+m");

        app.handle_key("right");
        assert_eq!(app.sidebar_section, SidebarSection::Agenda);
        app.handle_key("right");
        assert_eq!(app.sidebar_section, SidebarSection::Improvements);
        app.handle_key("right");
        assert_eq!(app.sidebar_section, SidebarSection::FeatureRequests);
        app.handle_key("right");
        assert_eq!(app.sidebar_section, SidebarSection::CodexSessions);
        app.handle_key("left");
        assert_eq!(app.sidebar_section, SidebarSection::FeatureRequests);
    }

    #[test]
    fn sidebar_section_switches_reset_selection_to_first_item() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Memory A".to_string(), "Memory B".to_string()];
        app.agenda_titles = vec!["Agenda A".to_string(), "Agenda B".to_string()];
        app.improvement_titles = vec!["Improvement A".to_string()];
        app.selected_sidebar_index = 1;

        app.handle_key("ctrl+a");
        assert_eq!(app.sidebar_section, SidebarSection::Agenda);
        assert_eq!(app.selected_sidebar_index, 0);

        app.selected_sidebar_index = 1;
        app.handle_key("right");
        assert_eq!(app.sidebar_section, SidebarSection::Improvements);
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn command_mode_tab_toggles_between_conversation_and_sidebar() {
        let mut app = TuiApp::bootstrap();
        app.handle_key("escape");
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Conversation));

        app.handle_key("tab");
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));

        app.handle_key("shift+tab");
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Conversation));
    }

    #[test]
    fn command_mode_movement_and_open_intents_match_ui_spec() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["One".to_string(), "Two".to_string()];
        app.handle_key("ctrl+m");

        let down = app.handle_key("j");
        assert_eq!(down, UiIntent::MoveDown);
        app.apply_intent(down);
        assert_eq!(app.selected_sidebar_index, 1);
        let up = app.handle_key("k");
        assert_eq!(up, UiIntent::MoveUp);
        app.apply_intent(up);
        assert_eq!(app.selected_sidebar_index, 0);
        assert_eq!(app.handle_key("enter"), UiIntent::OpenSelected);
        assert_eq!(app.handle_key("d"), UiIntent::ArchiveSelected);

        app.handle_key("ctrl+a");
        assert_eq!(app.handle_key("c"), UiIntent::CompleteSelected);
        assert_eq!(app.handle_key("d"), UiIntent::DeleteSelected);

        app.handle_key("r");
        assert_eq!(app.handle_key("c"), UiIntent::CompleteSelected);
        assert_eq!(app.handle_key("d"), UiIntent::Noop);

        app.handle_key("s");
        assert_eq!(app.handle_key("d"), UiIntent::Noop);
    }

    #[test]
    fn command_mode_conversation_keys_scroll_history_instead_of_sidebar() {
        let mut app = TuiApp::bootstrap();
        app.conversation_lines = vec![
            "line 0".to_string(),
            "line 1".to_string(),
            "line 2".to_string(),
        ];
        app.memory_titles = vec!["One".to_string(), "Two".to_string()];
        app.selected_sidebar_index = 1;
        app.handle_key("escape");
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Conversation));

        let down = app.handle_key("j");
        assert_eq!(down, UiIntent::MoveDown);
        app.apply_intent(down);
        assert!(!app.follow_conversation_output);
        assert_eq!(app.conversation_scroll, 1);
        assert_eq!(app.selected_sidebar_index, 1);
        assert_eq!(app.status, "scrolled conversation down");

        let up = app.handle_key("k");
        assert_eq!(up, UiIntent::MoveUp);
        app.apply_intent(up);
        assert_eq!(app.conversation_scroll, 0);
        assert_eq!(app.selected_sidebar_index, 1);
        assert_eq!(app.status, "scrolled conversation up");
    }

    #[test]
    fn mouse_scroll_on_chat_input_scrolls_conversation_history() {
        let mut app = TuiApp::bootstrap();
        app.conversation_lines = vec![
            "line 0".to_string(),
            "line 1".to_string(),
            "line 2".to_string(),
        ];
        app.follow_conversation_output = true;
        app.conversation_scroll = 1;
        app.focus = FocusTarget::Input;

        apply_mouse_event(
            &mut app,
            MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            },
        );

        assert_eq!(app.conversation_scroll, 0);
        assert!(!app.follow_conversation_output);
        assert_eq!(app.status, "scrolled conversation up");

        apply_mouse_event(
            &mut app,
            MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            },
        );

        assert_eq!(app.conversation_scroll, 1);
        assert_eq!(app.status, "scrolled conversation down");
    }

    #[test]
    fn mouse_scroll_is_ignored_when_modal_is_open() {
        let mut app = TuiApp::bootstrap();
        app.conversation_lines = vec!["line 0".to_string(), "line 1".to_string()];
        app.focus = FocusTarget::Input;
        app.command_form = Some(CommandFormState {
            command_name: "create_memory".to_string(),
            description: "desc".to_string(),
            fields: vec![],
            selected_field: 0,
            error: None,
        });

        apply_mouse_event(
            &mut app,
            MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            },
        );

        assert_eq!(app.conversation_scroll, 0);
        assert!(app.follow_conversation_output);
    }

    #[test]
    fn escaping_from_conversation_browse_reenables_following_latest_output() {
        let mut app = TuiApp::bootstrap();
        app.handle_key("escape");
        let down = app.handle_key("j");
        app.apply_intent(down);
        assert!(!app.follow_conversation_output);

        app.handle_key("escape");

        assert_eq!(app.focus, FocusTarget::Input);
        assert!(app.follow_conversation_output);
    }

    #[test]
    fn render_applies_conversation_scroll_offset() {
        let mut app = TuiApp::bootstrap();
        app.title = "Elroy".to_string();
        app.conversation_lines = vec![
            "line 0".to_string(),
            "line 1".to_string(),
            "line 2".to_string(),
            "line 3".to_string(),
            "line 4".to_string(),
        ];
        app.conversation_scroll = 2;
        app.follow_conversation_output = false;
        let area = Rect::new(0, 0, 80, 8);
        let mut buf = Buffer::empty(area);
        app.render(area, &mut buf);
        let text = buf
            .content
            .chunks(area.width as usize)
            .map(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains("line 0"));
        assert!(!text.contains("line 1"));
        assert!(text.contains("line 2"));
    }

    #[test]
    fn render_follows_latest_conversation_output_by_default() {
        let mut app = TuiApp::bootstrap();
        app.title = "Elroy".to_string();
        app.conversation_lines = vec![
            "line 0".to_string(),
            "line 1".to_string(),
            "line 2".to_string(),
            "line 3".to_string(),
            "line 4".to_string(),
        ];
        let area = Rect::new(0, 0, 80, 8);
        let mut buf = Buffer::empty(area);
        app.render(area, &mut buf);
        let text = buf
            .content
            .chunks(area.width as usize)
            .map(|row| {
                row.iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains("line 0"));
        assert!(text.contains("line 4"));
    }

    #[test]
    fn chat_mode_keys_do_not_change_focus() {
        let mut app = TuiApp::bootstrap();

        assert_eq!(app.handle_key("enter"), UiIntent::SubmitPrompt);
        assert_eq!(app.handle_key("up"), UiIntent::HistoryPrevious);
        assert_eq!(app.handle_key("down"), UiIntent::HistoryNext);
        assert_eq!(app.handle_key("tab"), UiIntent::CompleteInput);
        assert_eq!(app.focus, FocusTarget::Input);
    }

    #[test]
    fn multiline_paste_is_flattened_in_chat_input() {
        let mut app = TuiApp::bootstrap();

        apply_paste_event(&mut app, "---\ntitle: note\n---\nhello world");

        assert_eq!(app.input, "--- title: note --- hello world");
        assert_eq!(app.status, "editing prompt");
    }

    #[test]
    fn paste_is_ignored_outside_chat_input() {
        let mut app = TuiApp::bootstrap();
        app.focus = FocusTarget::Command(CommandPane::Sidebar);

        apply_paste_event(&mut app, "hello\nworld");

        assert_eq!(app.input, "");
    }

    #[test]
    fn unknown_focus_recovers_to_chat_input() {
        let mut app = TuiApp::bootstrap();
        app.focus = FocusTarget::Unknown;

        assert_eq!(app.handle_key("j"), UiIntent::Noop);
        assert_eq!(app.focus, FocusTarget::Input);
    }

    #[test]
    fn footer_hints_change_with_focus_mode() {
        let mut app = TuiApp::bootstrap();
        assert!(app.footer_hints().contains("Esc command mode"));

        app.handle_key("escape");
        assert!(app.footer_hints().contains("Tab switch pane"));
    }

    #[test]
    fn idle_footer_does_not_duplicate_command_palette_hint() {
        let mut app = TuiApp::bootstrap();
        app.model_name = "gpt-test".to_string();

        let footer = app.footer_status_text();
        let hints = app.footer_hints();
        let combined = format!("{footer} | {hints}");

        assert!(!footer.contains("Ctrl+P"));
        assert!(!footer.to_lowercase().contains("commands"));
        assert!(!combined.contains("Ctrl+P"));
        assert!(!combined.to_lowercase().contains("commands"));
    }

    #[test]
    fn footer_status_text_prefers_background_status_when_idle() {
        let mut app = TuiApp::bootstrap();
        app.model_name = "gpt-test".to_string();
        app.background_status = Some("syncing memories...".to_string());

        assert_eq!(
            app.footer_status_text(),
            "● gpt-test  ⟳ syncing memories..."
        );
    }

    #[test]
    fn footer_status_text_prefers_active_status_during_prompt() {
        let mut app = TuiApp::bootstrap();
        app.prompt_active = true;
        app.status = "thinking...".to_string();
        app.background_status = Some("syncing memories...".to_string());

        assert_eq!(app.footer_status_text(), "thinking...");
    }

    #[test]
    fn footer_status_text_prefers_active_status_during_command_action() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        app.status = "running command: /help".to_string();
        app.background_status = Some("running command...".to_string());

        assert_eq!(app.footer_status_text(), "running command: /help");
    }

    #[test]
    fn key_event_token_maps_terminal_keys_to_existing_ui_tokens() {
        assert_eq!(
            key_event_token(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::CONTROL)),
            Some("ctrl+m")
        );
        assert_eq!(
            key_event_token(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
            Some("shift+tab")
        );
        assert_eq!(
            key_event_token(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            Some("escape")
        );
    }

    #[test]
    fn chat_input_up_down_cycles_prompt_history() {
        let mut app = TuiApp::bootstrap();
        app.input_history = vec!["latest prompt".to_string(), "older prompt".to_string()];

        let intent = app.handle_key("up");
        app.apply_intent(intent);
        assert_eq!(app.input, "latest prompt");

        let intent = app.handle_key("up");
        app.apply_intent(intent);
        assert_eq!(app.input, "older prompt");

        let intent = app.handle_key("down");
        app.apply_intent(intent);
        assert_eq!(app.input, "latest prompt");

        let intent = app.handle_key("down");
        app.apply_intent(intent);
        assert_eq!(app.input, "");
    }

    #[test]
    fn chat_input_history_restores_in_progress_draft() {
        let mut app = TuiApp::bootstrap();
        app.input_history = vec!["latest prompt".to_string(), "older prompt".to_string()];
        app.input = "draft".to_string();

        app.apply_intent(UiIntent::HistoryPrevious);
        assert_eq!(app.input, "latest prompt");

        app.apply_intent(UiIntent::HistoryNext);
        assert_eq!(app.input, "draft");
    }

    #[test]
    fn chat_input_tab_accepts_matching_completion() {
        let mut app = TuiApp::bootstrap();
        app.input = "desk".to_string();
        app.input_completions = vec!["desk reset".to_string(), "doctor follow-up".to_string()];

        app.apply_intent(UiIntent::CompleteInput);

        assert_eq!(app.input, "desk reset");
        assert_eq!(app.status, "input completion requested");
    }

    #[test]
    fn chat_input_tab_completes_slash_command_name() {
        let mut app = TuiApp::bootstrap();
        app.input = "/re".to_string();
        app.input_completions = vec!["/reset_messages".to_string()];

        app.apply_intent(UiIntent::CompleteInput);

        assert_eq!(app.input, "/reset_messages");
        assert_eq!(app.status, "input completion requested");
    }

    #[test]
    fn slash_command_submit_executes_without_starting_prompt_stream() {
        let mut app = TuiApp::bootstrap();
        app.input = "/help".to_string();
        let mut runtime = FakeRuntime {
            slash_command_action: Some(TuiSlashCommandAction::Execute(TuiCommandExecution {
                command_name: "get_help".to_string(),
                display_name: "help".to_string(),
                values: vec![],
            })),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert!(pending.is_none());
        assert!(runtime.submitted_prompts.is_empty());
        assert_eq!(app.input, "");
        assert_eq!(app.focus, FocusTarget::Input);
        assert!(app.command_active);
        assert_eq!(app.status, "running command: /help");
        assert_eq!(
            runtime.started_command_executions,
            vec![TuiCommandExecution {
                command_name: "get_help".to_string(),
                display_name: "help".to_string(),
                values: vec![],
            }]
        );
        runtime.completed_command_execution_snapshot = Some(TuiSnapshot {
            conversation_lines: vec!["tool result: Available commands...".to_string()],
            status: Some("slash command executed: /help".to_string()),
            ..TuiSnapshot::default()
        });
        maybe_complete_command_execution(&mut app, &mut runtime);
        assert!(!app.command_active);
        assert_eq!(app.status, "slash command executed: /help");
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("tool result: Available commands...")
        );
        assert_eq!(app.input_history, vec!["/help".to_string()]);
    }

    #[test]
    fn slash_command_submit_error_preserves_input_and_does_not_start_prompt_stream() {
        let mut app = TuiApp::bootstrap();
        app.input = "/show_memory".to_string();
        let mut runtime = FakeRuntime {
            slash_command_error: Some("Missing required value for 'memory_name'".to_string()),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert!(pending.is_none());
        assert!(runtime.submitted_prompts.is_empty());
        assert_eq!(app.input, "/show_memory");
        assert_eq!(
            app.status,
            "slash command failed: Missing required value for 'memory_name'"
        );
        assert!(app.input_history.is_empty());
    }

    #[test]
    fn slash_prefixed_unknown_input_falls_back_to_plain_chat_stream() {
        let mut app = TuiApp::bootstrap();
        app.input = "/set_assistant_name".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert!(pending.is_some());
        assert_eq!(
            runtime.submitted_prompts,
            vec!["/set_assistant_name".to_string()]
        );
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("user: /set_assistant_name")
        );
        assert_eq!(app.status, "thinking...");
        assert_eq!(app.input, "");
        assert_eq!(app.input_history, vec!["/set_assistant_name".to_string()]);
    }

    #[test]
    fn slash_command_submit_opens_prefilled_command_form() {
        let mut app = TuiApp::bootstrap();
        app.input = "/create_memory trip".to_string();
        let mut runtime = FakeRuntime {
            slash_command_action: Some(TuiSlashCommandAction::OpenForm(TuiCommandForm {
                command_name: "create_memory".to_string(),
                description: "Create a memory".to_string(),
                parameters: vec![
                    TuiCommandParameter {
                        name: "name".to_string(),
                        optional: false,
                        default_text: String::new(),
                        suggestions: vec![],
                    },
                    TuiCommandParameter {
                        name: "text".to_string(),
                        optional: false,
                        default_text: String::new(),
                        suggestions: vec![],
                    },
                ],
                initial_values: vec![("name".to_string(), "trip".to_string())],
            })),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert!(pending.is_none());
        assert!(runtime.submitted_prompts.is_empty());
        assert_eq!(app.input, "");
        assert_eq!(app.input_history, vec!["/create_memory trip".to_string()]);
        let command_form = app.command_form.as_ref().expect("command form should open");
        assert_eq!(command_form.command_name, "create_memory");
        assert_eq!(command_form.selected_field, 1);
        assert_eq!(command_form.fields[0].name, "name");
        assert_eq!(command_form.fields[0].value, "trip");
        assert_eq!(command_form.fields[1].name, "text");
        assert_eq!(command_form.fields[1].value, "");
    }

    #[test]
    fn command_form_submit_executes_runtime_command() {
        let mut app = TuiApp::bootstrap();
        app.open_command_form(TuiCommandForm {
            command_name: "create_memory".to_string(),
            description: "Create a memory".to_string(),
            parameters: vec![
                TuiCommandParameter {
                    name: "name".to_string(),
                    optional: false,
                    default_text: String::new(),
                    suggestions: vec![],
                },
                TuiCommandParameter {
                    name: "text".to_string(),
                    optional: false,
                    default_text: String::new(),
                    suggestions: vec![],
                },
            ],
            initial_values: vec![("name".to_string(), "trip".to_string())],
        });
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('e'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_form.is_none());
        assert_eq!(app.focus, FocusTarget::Input);
        assert_eq!(
            runtime.started_command_executions,
            vec![TuiCommandExecution {
                command_name: "create_memory".to_string(),
                display_name: "create_memory".to_string(),
                values: vec![
                    ("name".to_string(), "trip".to_string()),
                    ("text".to_string(), "note".to_string()),
                ],
            }]
        );
        assert!(app.command_active);
        assert_eq!(app.status, "running command: /create_memory");
        runtime.completed_command_execution_snapshot = Some(TuiSnapshot {
            conversation_lines: vec!["tool result: submitted /create_memory".to_string()],
            status: Some("slash command executed: /create_memory".to_string()),
            ..TuiSnapshot::default()
        });
        maybe_complete_command_execution(&mut app, &mut runtime);
        assert!(!app.command_active);
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("tool result: submitted /create_memory")
        );
        assert_eq!(app.status, "slash command executed: /create_memory");
    }

    #[test]
    fn background_command_runtime_failure_returns_focus_and_surfaces_status() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        app.focus = FocusTarget::Command(CommandPane::Conversation);
        app.status = "running command: /create_memory".to_string();
        app.follow_conversation_output = false;
        let mut runtime = FakeRuntime {
            command_execution_error: Some("background worker exploded".to_string()),
            ..FakeRuntime::default()
        };

        let restart = maybe_complete_command_execution(&mut app, &mut runtime);

        assert_eq!(restart, None);
        assert!(!app.command_active);
        assert_eq!(app.focus, FocusTarget::Input);
        assert_eq!(app.status, "command failed: background worker exploded");
        assert!(app.follow_conversation_output);
    }

    #[test]
    fn command_form_submit_keeps_form_open_when_command_is_already_running() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        app.open_command_form(TuiCommandForm {
            command_name: "create_memory".to_string(),
            description: "Create a memory".to_string(),
            parameters: vec![TuiCommandParameter {
                name: "name".to_string(),
                optional: false,
                default_text: String::new(),
                suggestions: vec![],
            }],
            initial_values: vec![("name".to_string(), "trip".to_string())],
        });
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_form = app
            .command_form
            .as_ref()
            .expect("command form should remain open");
        assert_eq!(
            command_form.error.as_deref(),
            Some("Wait for the current task to finish before sending another message.")
        );
        assert!(runtime.started_command_executions.is_empty());
    }

    #[test]
    fn command_form_submit_keeps_form_open_when_prompt_stream_is_active() {
        let mut app = TuiApp::bootstrap();
        app.prompt_active = true;
        app.open_command_form(TuiCommandForm {
            command_name: "create_memory".to_string(),
            description: "Create a memory".to_string(),
            parameters: vec![TuiCommandParameter {
                name: "name".to_string(),
                optional: false,
                default_text: String::new(),
                suggestions: vec![],
            }],
            initial_values: vec![("name".to_string(), "trip".to_string())],
        });
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_form = app
            .command_form
            .as_ref()
            .expect("command form should remain open");
        assert_eq!(
            command_form.error.as_deref(),
            Some("Wait for the current task to finish before sending another message.")
        );
        assert!(runtime.started_command_executions.is_empty());
    }

    #[test]
    fn command_form_submit_keeps_form_open_when_command_launch_fails() {
        let mut app = TuiApp::bootstrap();
        app.open_command_form(TuiCommandForm {
            command_name: "create_memory".to_string(),
            description: "Create a memory".to_string(),
            parameters: vec![TuiCommandParameter {
                name: "name".to_string(),
                optional: false,
                default_text: String::new(),
                suggestions: vec![],
            }],
            initial_values: vec![("name".to_string(), "trip".to_string())],
        });
        let mut runtime = FakeRuntime {
            start_command_execution_error: Some("memory_name must be a string".to_string()),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_form = app
            .command_form
            .as_ref()
            .expect("command form should remain open");
        assert_eq!(
            command_form.error.as_deref(),
            Some("memory_name must be a string")
        );
        assert!(runtime.started_command_executions.is_empty());
    }

    #[test]
    fn command_action_keeps_input_editable_and_blocks_submit() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        app.status = "running command: /refresh_system_instructions".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert_eq!(app.input, "dr");
        assert!(pending.is_none());
        assert!(runtime.submitted_prompts.is_empty());
        assert_eq!(
            app.status,
            "Wait for the current task to finish before sending another message."
        );
    }

    #[test]
    fn command_form_tab_accepts_matching_suggestion_before_moving() {
        let mut app = TuiApp::bootstrap();
        app.open_command_form(TuiCommandForm {
            command_name: "show_task".to_string(),
            description: "Show a task".to_string(),
            parameters: vec![
                TuiCommandParameter {
                    name: "name".to_string(),
                    optional: false,
                    default_text: String::new(),
                    suggestions: vec!["trip note".to_string()],
                },
                TuiCommandParameter {
                    name: "text".to_string(),
                    optional: true,
                    default_text: String::new(),
                    suggestions: vec![],
                },
            ],
            initial_values: vec![],
        });
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_form = app
            .command_form
            .as_ref()
            .expect("command form should remain open");
        assert_eq!(command_form.fields[0].value, "trip note");
        assert_eq!(command_form.selected_field, 0);

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_form = app
            .command_form
            .as_ref()
            .expect("command form should remain open");
        assert_eq!(command_form.selected_field, 1);
    }

    #[test]
    fn ctrl_p_opens_command_palette_from_input() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime {
            command_palette_entries: vec![TuiCommandPaletteEntry {
                title: "/create_memory".to_string(),
                description: "Create a memory".to_string(),
                action: TuiCommandPaletteAction::ToolCommand("create_memory".to_string()),
            }],
            ..FakeRuntime::default()
        };
        let mut pending = None;

        let exit = apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            &mut runtime,
            &mut pending,
        );

        assert_eq!(exit, TuiExit::Continue);
        let command_palette = app.command_palette.as_ref().expect("palette should open");
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Focus Memories")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Focus Agenda")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Focus Improvements")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Focus Feature Requests")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Focus Codex Sessions")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Refresh System Instructions")
        );
        assert!(
            command_palette
                .all_entries
                .iter()
                .any(|entry| entry.title == "Reset Messages")
        );
        assert_eq!(app.status, "command palette opened");
    }

    #[test]
    fn command_palette_can_execute_system_command_entry() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Refresh System Instructions");
        let mut runtime = FakeRuntime {
            launch_named_command_action: Some(TuiSlashCommandAction::Execute(
                TuiCommandExecution {
                    command_name: "refresh_system_instructions".to_string(),
                    display_name: "refresh_system_instructions".to_string(),
                    values: vec![],
                },
            )),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Input);
        assert!(app.command_active);
        assert_eq!(
            runtime.started_command_executions,
            vec![TuiCommandExecution {
                command_name: "refresh_system_instructions".to_string(),
                display_name: "refresh_system_instructions".to_string(),
                values: vec![],
            }]
        );
        runtime.completed_command_execution_snapshot = Some(TuiSnapshot {
            conversation_lines: vec!["assistant: refreshed".to_string()],
            status: Some("slash command executed: /refresh_system_instructions".to_string()),
            ..TuiSnapshot::default()
        });
        maybe_complete_command_execution(&mut app, &mut runtime);
        assert!(!app.command_active);
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("assistant: refreshed")
        );
        assert_eq!(
            app.status,
            "slash command executed: /refresh_system_instructions"
        );
    }

    #[test]
    fn command_palette_can_focus_memories_section() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Agenda;
        app.selected_sidebar_index = 1;
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Focus Memories");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.last_command_pane, CommandPane::Sidebar);
        assert_eq!(app.sidebar_section, SidebarSection::Memories);
        assert_eq!(app.status, "focused memories");
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn command_palette_can_focus_agenda_section() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Memories;
        app.selected_sidebar_index = 1;
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Focus Agenda");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.last_command_pane, CommandPane::Sidebar);
        assert_eq!(app.sidebar_section, SidebarSection::Agenda);
        assert_eq!(app.status, "focused agenda");
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn command_palette_can_focus_improvements_section() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Memories;
        app.selected_sidebar_index = 1;
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Focus Improvements");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.last_command_pane, CommandPane::Sidebar);
        assert_eq!(app.sidebar_section, SidebarSection::Improvements);
        assert_eq!(app.status, "focused improvements");
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn command_palette_can_focus_feature_requests_section() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Focus Feature Requests");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.last_command_pane, CommandPane::Sidebar);
        assert_eq!(app.sidebar_section, SidebarSection::FeatureRequests);
        assert_eq!(app.status, "focused feature requests");
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn command_palette_can_focus_codex_sessions_section() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![]);
        select_command_palette_entry(&mut app, "Focus Codex Sessions");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.last_command_pane, CommandPane::Sidebar);
        assert_eq!(app.sidebar_section, SidebarSection::CodexSessions);
        assert_eq!(app.status, "focused codex sessions");
        assert_eq!(app.selected_sidebar_index, 0);
    }

    #[test]
    fn background_command_completion_can_return_restart_request() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        let mut runtime = FakeRuntime {
            completed_command_execution_snapshot: Some(TuiSnapshot {
                status: Some("slash command executed: /restart_session".to_string()),
                ..TuiSnapshot::default()
            }),
            pending_restart_request: Some("Restarted successfully. Ready to continue.".to_string()),
            ..FakeRuntime::default()
        };

        let restart = maybe_complete_command_execution(&mut app, &mut runtime);

        assert_eq!(
            restart.as_deref(),
            Some("Restarted successfully. Ready to continue.")
        );
        assert!(!app.command_active);
        assert_eq!(app.focus, FocusTarget::Input);
    }

    #[test]
    fn runtime_tick_returns_restart_requested_after_background_command_completion() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        let mut runtime = FakeRuntime {
            completed_command_execution_snapshot: Some(TuiSnapshot {
                status: Some("slash command executed: /restart_session".to_string()),
                ..TuiSnapshot::default()
            }),
            pending_restart_request: Some("Restarted successfully. Ready to continue.".to_string()),
            ..FakeRuntime::default()
        };
        let mut pending = None;
        let mut deferred_context_refresh_at = None;
        let mut previous_background_status = None;

        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(
            result,
            Some(TuiRunResult::RestartRequested(
                "Restarted successfully. Ready to continue.".to_string()
            ))
        );
        assert!(!app.command_active);
        assert_eq!(app.focus, FocusTarget::Input);
        assert!(runtime.pending_restart_request.is_none());
    }

    #[test]
    fn runtime_tick_returns_restart_requested_after_prompt_completion() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime {
            pending_restart_request: Some("Restarted successfully. Ready to continue.".to_string()),
            ..FakeRuntime::default()
        };
        let mut pending = Some(PendingPrompt {
            submitted_prompt: Some("hello runtime".to_string()),
            schedule_self_reflection: true,
            before_ids: HashSet::new(),
            stream: Box::new(FakePromptStream {
                updates: vec![],
                finalized_snapshot: TuiSnapshot {
                    conversation_lines: vec!["assistant: runtime response".to_string()],
                    ..TuiSnapshot::default()
                },
                cancelled_snapshot: TuiSnapshot::default(),
            }),
        });
        let mut deferred_context_refresh_at = None;
        let mut previous_background_status = None;

        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(
            result,
            Some(TuiRunResult::RestartRequested(
                "Restarted successfully. Ready to continue.".to_string()
            ))
        );
        assert!(pending.is_none());
        assert_eq!(app.focus, FocusTarget::Input);
        assert_eq!(app.status, "submitted prompt: hello runtime");
        assert_eq!(runtime.self_reflection_runs, 0);
        assert!(runtime.pending_restart_request.is_none());
    }

    #[test]
    fn command_palette_enter_launches_command_form() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![TuiCommandPaletteEntry {
            title: "/create_memory".to_string(),
            description: "Create a memory".to_string(),
            action: TuiCommandPaletteAction::ToolCommand("create_memory".to_string()),
        }]);
        select_command_palette_entry(&mut app, "/create_memory");
        let mut runtime = FakeRuntime {
            launch_named_command_action: Some(TuiSlashCommandAction::OpenForm(TuiCommandForm {
                command_name: "create_memory".to_string(),
                description: "Create a memory".to_string(),
                parameters: vec![
                    TuiCommandParameter {
                        name: "name".to_string(),
                        optional: false,
                        default_text: String::new(),
                        suggestions: vec![],
                    },
                    TuiCommandParameter {
                        name: "text".to_string(),
                        optional: false,
                        default_text: String::new(),
                        suggestions: vec![],
                    },
                ],
                initial_values: vec![],
            })),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert!(app.command_form.is_some());
        assert_eq!(app.status, "editing command: /create_memory");
    }

    #[test]
    fn command_palette_keeps_modal_open_when_command_launch_is_blocked() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![TuiCommandPaletteEntry {
            title: "/refresh_system_instructions".to_string(),
            description: "Refresh system instructions".to_string(),
            action: TuiCommandPaletteAction::ToolCommand("refresh_system_instructions".to_string()),
        }]);
        select_command_palette_entry(&mut app, "/refresh_system_instructions");
        app.command_active = true;
        let mut runtime = FakeRuntime {
            launch_named_command_action: Some(TuiSlashCommandAction::Execute(
                TuiCommandExecution {
                    command_name: "refresh_system_instructions".to_string(),
                    display_name: "refresh_system_instructions".to_string(),
                    values: vec![],
                },
            )),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_some());
        assert_eq!(
            app.status,
            "command launch failed: Wait for the current task to finish before sending another message."
        );
    }

    #[test]
    fn command_palette_keeps_modal_open_when_prompt_stream_is_active() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![TuiCommandPaletteEntry {
            title: "/refresh_system_instructions".to_string(),
            description: "Refresh system instructions".to_string(),
            action: TuiCommandPaletteAction::ToolCommand("refresh_system_instructions".to_string()),
        }]);
        select_command_palette_entry(&mut app, "/refresh_system_instructions");
        app.prompt_active = true;
        let mut runtime = FakeRuntime {
            launch_named_command_action: Some(TuiSlashCommandAction::Execute(
                TuiCommandExecution {
                    command_name: "refresh_system_instructions".to_string(),
                    display_name: "refresh_system_instructions".to_string(),
                    values: vec![],
                },
            )),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_some());
        assert_eq!(
            app.status,
            "command launch failed: Wait for the current task to finish before sending another message."
        );
    }

    #[test]
    fn command_palette_filters_entries_from_typed_query() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![
            TuiCommandPaletteEntry {
                title: "/create_memory".to_string(),
                description: "Create a memory".to_string(),
                action: TuiCommandPaletteAction::ToolCommand("create_memory".to_string()),
            },
            TuiCommandPaletteEntry {
                title: "/show_memory".to_string(),
                description: "Show a memory".to_string(),
                action: TuiCommandPaletteAction::ToolCommand("show_memory".to_string()),
            },
        ]);
        let mut runtime = FakeRuntime {
            launch_named_command_action: Some(TuiSlashCommandAction::OpenForm(TuiCommandForm {
                command_name: "show_memory".to_string(),
                description: "Show a memory".to_string(),
                parameters: vec![TuiCommandParameter {
                    name: "memory_name".to_string(),
                    optional: false,
                    default_text: String::new(),
                    suggestions: vec![],
                }],
                initial_values: vec![],
            })),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('w'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        let command_palette = app
            .command_palette
            .as_ref()
            .expect("command palette should remain open");
        assert_eq!(command_palette.query, "show");
        assert_eq!(command_palette.filtered_indices.len(), 1);
        assert_eq!(
            command_palette.selected_entry().map(|entry| entry.title),
            Some("/show_memory".to_string())
        );

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_none());
        assert_eq!(app.status, "editing command: /show_memory");
        assert!(app.command_form.is_some());
    }

    #[test]
    fn command_palette_reports_no_match_without_closing() {
        let mut app = TuiApp::bootstrap();
        app.open_command_palette(vec![TuiCommandPaletteEntry {
            title: "/create_memory".to_string(),
            description: "Create a memory".to_string(),
            action: TuiCommandPaletteAction::ToolCommand("create_memory".to_string()),
        }]);
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );

        assert!(app.command_palette.is_some());
        assert_eq!(app.status, "no matching command");
    }

    #[test]
    fn submitting_prompt_records_history_for_later_recall() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello runtime".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert_eq!(app.input_history, vec!["hello runtime".to_string()]);
        app.apply_intent(UiIntent::HistoryPrevious);
        assert_eq!(app.input, "hello runtime");
    }

    #[test]
    fn apply_key_event_appends_input_and_submits_prompt() {
        let mut app = TuiApp::bootstrap();
        let mut pending = None;

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
                &mut FakeRuntime::default(),
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
                &mut FakeRuntime::default(),
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.input, "hi");

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut FakeRuntime::default(),
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.status, "thinking...");
        assert!(app.input.is_empty());
        assert!(pending.is_some());
    }

    #[test]
    fn ctrl_d_requests_quit() {
        let mut app = TuiApp::bootstrap();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
                &mut FakeRuntime::default(),
                &mut None,
            ),
            TuiExit::Quit
        );
        assert_eq!(app.status, "quit");
    }

    #[test]
    fn ctrl_c_clears_input_when_no_stream_is_active() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;
        app.input = "draft".to_string();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.input, "");
        assert_eq!(app.status, "cleared prompt");
    }

    #[test]
    fn ctrl_c_cancels_active_stream_and_preserves_draft() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;
        app.input = "hello".to_string();

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        app.input = "draft".to_string();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.input, "draft");
        assert_eq!(app.status, "Chat stream cancelled");
        assert!(pending.is_none());
    }

    #[test]
    fn streaming_draft_survives_browse_mode_switches() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;
        app.input = "hello".to_string();

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        app.input = "draft".to_string();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('m'), KeyModifiers::CONTROL),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.focus, FocusTarget::Command(CommandPane::Sidebar));
        assert_eq!(app.sidebar_section, SidebarSection::Memories);
        assert_eq!(app.input, "draft");

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.sidebar_section, SidebarSection::Agenda);
        assert_eq!(app.input, "draft");

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(app.focus, FocusTarget::Input);
        assert_eq!(app.input, "draft");
    }

    #[test]
    fn streaming_draft_survives_sidebar_and_status_updates() {
        let mut app = TuiApp::bootstrap();
        app.prompt_active = true;
        app.input = "draft".to_string();

        app.apply_snapshot(TuiSnapshot {
            memory_titles: vec!["Trip note".to_string()],
            agenda_titles: vec!["Call mom".to_string()],
            input_completions: vec!["Trip note".to_string()],
            status: Some("loading context...".to_string()),
            ..TuiSnapshot::default()
        });

        assert_eq!(app.input, "draft");
        assert_eq!(app.memory_titles, vec!["Trip note".to_string()]);
        assert_eq!(app.agenda_titles, vec!["Call mom".to_string()]);
        assert_eq!(app.input_completions, vec!["Trip note".to_string()]);
        assert_eq!(app.status, "loading context...");
    }

    #[test]
    fn runtime_submit_replaces_snapshot_data() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello runtime".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        while pending.is_some() {
            advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        }

        assert_eq!(runtime.submitted_prompts, vec!["hello runtime".to_string()]);
        assert!(
            app.conversation_lines
                .iter()
                .any(|line| line.contains("assistant: runtime response"))
        );
        assert!(app.memory_titles.iter().any(|line| line == "Fresh Memory"));
        assert!(app.input.is_empty());
    }

    #[test]
    fn runtime_open_shows_detail_modal() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Runner Notes".to_string()];
        app.handle_key("ctrl+m");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime, &mut pending);

        assert_eq!(
            runtime.last_opened,
            Some((SidebarSection::Memories, "Runner Notes".to_string()))
        );
        let detail_modal = app.detail_modal.as_ref().expect("detail modal should open");
        assert_eq!(detail_modal.title, "Runner Notes");
        assert_eq!(
            detail_modal.footer_text(),
            "D: archive  |  Escape/Enter/Q: close"
        );

        app.codex_session_titles = vec!["sample (completed) thread-123".to_string()];
        app.handle_key("s");
        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime, &mut pending);
        assert_eq!(
            runtime.last_opened,
            Some((
                SidebarSection::CodexSessions,
                "sample (completed) thread-123".to_string()
            ))
        );
        let detail_modal = app.detail_modal.as_ref().expect("codex modal should open");
        assert_eq!(detail_modal.footer_text(), "Escape/Enter/Q: close");
        assert!(detail_modal.content.contains("Status: completed"));
        assert!(detail_modal.content.contains("Summary:"));
        assert!(detail_modal.content.contains("Latest Agent Message:"));
    }

    #[test]
    fn codex_detail_modal_ignores_destructive_shortcuts() {
        let mut app = TuiApp::bootstrap();
        app.codex_session_titles = vec!["sample (completed) thread-123".to_string()];
        app.handle_key("s");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime, &mut pending);
        assert!(app.detail_modal.is_some());

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );

        assert!(app.detail_modal.is_some());
        assert_eq!(runtime.last_mutation, None);
        assert_eq!(
            app.detail_modal
                .as_ref()
                .expect("codex modal should remain open")
                .footer_text(),
            "Escape/Enter/Q: close"
        );
    }

    #[test]
    fn runtime_mutation_refreshes_snapshot_data() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Runner Notes".to_string()];
        app.handle_key("ctrl+m");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(
            &mut app,
            UiIntent::ArchiveSelected,
            &mut runtime,
            &mut pending,
        );

        assert_eq!(
            runtime.last_mutation,
            Some((
                SidebarSection::Memories,
                "Runner Notes".to_string(),
                SidebarAction::Archive
            ))
        );
        assert!(
            app.memory_titles
                .iter()
                .any(|title| title == "After Mutation")
        );
        assert!(app.status.contains("updated selected"));
        assert!(app.detail_modal.is_none());
    }

    #[test]
    fn completed_background_status_refreshes_snapshot_data() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };

        maybe_refresh_snapshot_after_background_completion(
            &mut app,
            &mut runtime,
            false,
            false,
            Some("refreshing context..."),
            None,
        );

        assert_eq!(app.memory_titles, vec!["Fresh Memory".to_string()]);
    }

    #[test]
    fn completed_background_status_refresh_preserves_local_status() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        app.status = "editing command: /create_memory".to_string();
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                status: Some("loaded persisted transcript and sidebar data".to_string()),
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };

        maybe_refresh_snapshot_after_background_completion(
            &mut app,
            &mut runtime,
            false,
            false,
            Some("refreshing context..."),
            None,
        );

        assert_eq!(app.memory_titles, vec!["Fresh Memory".to_string()]);
        assert_eq!(app.status, "editing command: /create_memory");
    }

    #[test]
    fn active_prompt_does_not_refresh_snapshot_from_background_completion() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };

        maybe_refresh_snapshot_after_background_completion(
            &mut app,
            &mut runtime,
            true,
            false,
            Some("refreshing context..."),
            None,
        );

        assert_eq!(app.memory_titles, vec!["Stale Memory".to_string()]);
    }

    #[test]
    fn active_command_does_not_refresh_snapshot_from_background_completion() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };

        maybe_refresh_snapshot_after_background_completion(
            &mut app,
            &mut runtime,
            false,
            true,
            Some("refreshing context..."),
            None,
        );

        assert_eq!(app.memory_titles, vec!["Stale Memory".to_string()]);
    }

    #[test]
    fn runtime_tick_defers_background_snapshot_refresh_until_prompt_stream_finishes() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };
        let mut pending = Some(PendingPrompt {
            submitted_prompt: Some("hello runtime".to_string()),
            schedule_self_reflection: false,
            before_ids: HashSet::new(),
            stream: Box::new(FakePromptStream {
                updates: vec![PromptUpdate::Status("thinking...".to_string())],
                finalized_snapshot: TuiSnapshot::default(),
                cancelled_snapshot: TuiSnapshot::default(),
            }),
        });
        let mut deferred_context_refresh_at = None;
        let mut previous_background_status = Some("refreshing context...".to_string());

        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert!(pending.is_some());
        assert!(app.prompt_active);
        assert_eq!(app.memory_titles, vec!["Stale Memory".to_string()]);
        assert_eq!(
            previous_background_status,
            Some("refreshing context...".to_string())
        );

        pending = None;
        app.prompt_active = false;
        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert_eq!(app.memory_titles, vec!["Fresh Memory".to_string()]);
        assert_eq!(previous_background_status, None);
    }

    #[test]
    fn runtime_tick_defers_background_snapshot_refresh_until_command_action_finishes() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            memory_titles: vec!["Stale Memory".to_string()],
            ..TuiSnapshot::default()
        });
        app.command_active = true;
        let mut runtime = FakeRuntime {
            snapshot: TuiSnapshot {
                memory_titles: vec!["Fresh Memory".to_string()],
                ..TuiSnapshot::default()
            },
            ..FakeRuntime::default()
        };
        let mut pending = None;
        let mut deferred_context_refresh_at = None;
        let mut previous_background_status = Some("refreshing context...".to_string());

        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert!(app.command_active);
        assert_eq!(app.memory_titles, vec!["Stale Memory".to_string()]);
        assert_eq!(
            previous_background_status,
            Some("refreshing context...".to_string())
        );

        app.command_active = false;
        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            Instant::now(),
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert_eq!(app.memory_titles, vec!["Fresh Memory".to_string()]);
        assert_eq!(previous_background_status, None);
    }

    #[test]
    fn completed_user_prompt_triggers_deferred_self_reflection() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime {
            context_messages: vec![TuiContextMessage {
                id: 1,
                role: "assistant".to_string(),
                content: "done".to_string(),
            }],
            ..FakeRuntime::default()
        };
        let mut pending_prompt = Some(PendingPrompt {
            submitted_prompt: Some("remember this".to_string()),
            schedule_self_reflection: true,
            before_ids: HashSet::new(),
            stream: Box::new(FakePromptStream {
                updates: vec![],
                finalized_snapshot: TuiSnapshot::default(),
                cancelled_snapshot: TuiSnapshot::default(),
            }),
        });

        let advance = advance_prompt_stream(&mut app, &mut runtime, &mut pending_prompt);

        assert_eq!(advance, PromptAdvance::CompletedTurn);
        assert_eq!(runtime.self_reflection_runs, 1);
    }

    #[test]
    fn agenda_detail_modal_supports_complete_and_delete_confirmation() {
        let mut app = TuiApp::bootstrap();
        app.agenda_titles = vec!["Pay rent [2000-01-01 09:00] (Due)".to_string()];
        app.handle_key("ctrl+a");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime, &mut pending);
        let detail_modal = app.detail_modal.as_ref().expect("detail modal should open");
        assert_eq!(
            detail_modal.footer_text(),
            "C: complete  |  D: delete  |  Escape/Enter/Q: close"
        );

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        let detail_modal = app
            .detail_modal
            .as_ref()
            .expect("detail modal should still be open");
        assert_eq!(
            detail_modal.footer_text(),
            "Press D again to confirm deletion, any other key to cancel"
        );

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        let detail_modal = app
            .detail_modal
            .as_ref()
            .expect("detail modal should remain open");
        assert_eq!(
            detail_modal.footer_text(),
            "C: complete  |  D: delete  |  Escape/Enter/Q: close"
        );

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE),
                &mut runtime,
                &mut pending,
            ),
            TuiExit::Continue
        );
        assert_eq!(
            runtime.last_mutation,
            Some((
                SidebarSection::Agenda,
                "Pay rent [2000-01-01 09:00] (Due)".to_string(),
                SidebarAction::Complete
            ))
        );
        assert!(app.detail_modal.is_none());
    }

    #[test]
    fn detail_modal_confirms_destructive_action_before_mutating() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Runner Notes".to_string()];
        app.handle_key("ctrl+m");
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime, &mut pending);

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        assert_eq!(runtime.last_mutation, None);

        apply_key_event(
            &mut app,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
            &mut runtime,
            &mut pending,
        );
        assert_eq!(
            runtime.last_mutation,
            Some((
                SidebarSection::Memories,
                "Runner Notes".to_string(),
                SidebarAction::Archive
            ))
        );
        assert!(app.detail_modal.is_none());
    }

    #[test]
    fn runtime_stream_updates_conversation_before_final_snapshot() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello runtime".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        app.focus = FocusTarget::Command(CommandPane::Sidebar);

        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("user: hello runtime")
        );
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        assert_eq!(app.status, "thinking...");
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("assistant: runtime response")
        );
        let mut finalized = false;
        while pending.is_some() {
            finalized = matches!(
                advance_prompt_stream(&mut app, &mut runtime, &mut pending),
                PromptAdvance::CompletedTurn
            );
        }
        assert!(finalized);
        assert_eq!(app.status, "submitted prompt: hello runtime");
        assert_eq!(app.focus, FocusTarget::Input);
    }

    #[test]
    fn startup_prompt_stream_runs_without_rendering_user_line() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime {
            startup_stream: Some(FakePromptStream {
                updates: vec![
                    PromptUpdate::Status("thinking...".to_string()),
                    PromptUpdate::AssistantDelta("welcome ".to_string()),
                    PromptUpdate::AssistantDelta("back".to_string()),
                ],
                finalized_snapshot: TuiSnapshot {
                    conversation_lines: vec!["assistant: welcome back".to_string()],
                    status: Some("loaded startup".to_string()),
                    ..TuiSnapshot::default()
                },
                cancelled_snapshot: TuiSnapshot::default(),
            }),
            context_messages: vec![TuiContextMessage {
                id: 22,
                role: "assistant".to_string(),
                content: "welcome back".to_string(),
            }],
            ..FakeRuntime::default()
        };

        let mut pending = start_startup_prompt_stream(&mut app, &mut runtime);
        assert_eq!(app.status, "thinking...");
        assert!(pending.is_some());
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        advance_prompt_stream(&mut app, &mut runtime, &mut pending);
        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("assistant: welcome back")
        );
        let mut finalized = false;
        while pending.is_some() {
            finalized = matches!(
                advance_prompt_stream(&mut app, &mut runtime, &mut pending),
                PromptAdvance::CompletedTurn
            );
        }
        assert!(finalized);
        assert_eq!(app.status, "loaded startup");
        assert_eq!(app.rendered_context_message_ids, HashSet::from([22]));
        assert!(
            !app.conversation_lines
                .iter()
                .any(|line| line.starts_with("user:"))
        );
    }

    #[test]
    fn completed_prompt_returns_restart_request() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello runtime".to_string();
        let mut runtime = FakeRuntime {
            pending_restart_request: Some("Restarted successfully. Ready to continue.".to_string()),
            ..FakeRuntime::default()
        };
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        let mut restart_request = None;
        while pending.is_some() {
            if let PromptAdvance::RestartRequested(resume_message) =
                advance_prompt_stream(&mut app, &mut runtime, &mut pending)
            {
                restart_request = Some(resume_message);
            }
        }

        assert_eq!(
            app.conversation_lines.last().map(String::as_str),
            Some("assistant: runtime response")
        );
        assert_eq!(app.status, "submitted prompt: hello runtime");
        assert!(runtime.pending_restart_request.is_none());
        assert_eq!(runtime.self_reflection_runs, 0);
        assert_eq!(
            restart_request.as_deref(),
            Some("Restarted successfully. Ready to continue.")
        );
        assert!(
            !app.conversation_lines
                .iter()
                .any(|line| line == "user: Restarted successfully. Ready to continue.")
        );
    }

    #[test]
    fn submit_is_blocked_while_stream_is_active_and_input_stays_editable() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        app.input = "draft".to_string();
        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);

        assert_eq!(app.input, "draft");
        assert_eq!(
            app.status,
            "Wait for the current task to finish before sending another message."
        );
        assert_eq!(runtime.submitted_prompts, vec!["hello".to_string()]);
    }

    #[test]
    fn mark_messages_rendered_after_chat_turn_skips_earlier_background_messages() {
        let mut app = TuiApp::bootstrap();
        let context_messages = vec![
            TuiContextMessage {
                id: 11,
                role: "assistant".to_string(),
                content: "background update".to_string(),
            },
            TuiContextMessage {
                id: 12,
                role: "user".to_string(),
                content: "hello".to_string(),
            },
            TuiContextMessage {
                id: 13,
                role: "assistant".to_string(),
                content: "foreground reply".to_string(),
            },
        ];

        app.mark_messages_rendered_after_chat_turn("hello", &HashSet::new(), &context_messages);

        assert!(!app.rendered_context_message_ids.contains(&11));
        assert!(app.rendered_context_message_ids.contains(&12));
        assert!(app.rendered_context_message_ids.contains(&13));
    }

    #[test]
    fn mark_messages_rendered_after_bootstrap_stream_marks_new_messages() {
        let mut app = TuiApp::bootstrap();
        let context_messages = vec![TuiContextMessage {
            id: 22,
            role: "assistant".to_string(),
            content: "welcome back".to_string(),
        }];

        app.mark_messages_rendered_after_bootstrap_stream(&HashSet::new(), &context_messages);

        assert_eq!(app.rendered_context_message_ids, HashSet::from([22]));
    }

    #[test]
    fn poll_context_updates_renders_trailing_unseen_messages() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["user: hello".to_string()],
            ..TuiSnapshot::default()
        });
        app.rendered_context_message_ids = HashSet::from([1]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 1,
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
                TuiContextMessage {
                    id: 2,
                    role: "assistant".to_string(),
                    content: "background update".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };

        poll_context_updates(&mut app, &mut runtime);

        assert_eq!(
            app.conversation_lines,
            vec![
                "user: hello".to_string(),
                "assistant: background update".to_string()
            ]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1, 2]));
    }

    #[test]
    fn poll_context_updates_marks_interleaved_messages_without_rendering_them() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["assistant: existing".to_string()],
            ..TuiSnapshot::default()
        });
        app.rendered_context_message_ids = HashSet::from([10, 12]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 10,
                    role: "assistant".to_string(),
                    content: "existing".to_string(),
                },
                TuiContextMessage {
                    id: 11,
                    role: "assistant".to_string(),
                    content: "unseen middle".to_string(),
                },
                TuiContextMessage {
                    id: 12,
                    role: "assistant".to_string(),
                    content: "already rendered later".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };

        poll_context_updates(&mut app, &mut runtime);

        assert_eq!(
            app.conversation_lines,
            vec!["assistant: existing".to_string()]
        );
        assert_eq!(
            app.rendered_context_message_ids,
            HashSet::from([10, 11, 12])
        );
    }

    #[test]
    fn poll_context_updates_renders_trailing_unseen_suffix_after_interleaved_gap() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["assistant: existing".to_string()],
            ..TuiSnapshot::default()
        });
        app.rendered_context_message_ids = HashSet::from([10, 12]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 10,
                    role: "assistant".to_string(),
                    content: "existing".to_string(),
                },
                TuiContextMessage {
                    id: 11,
                    role: "assistant".to_string(),
                    content: "unseen middle".to_string(),
                },
                TuiContextMessage {
                    id: 12,
                    role: "assistant".to_string(),
                    content: "already rendered later".to_string(),
                },
                TuiContextMessage {
                    id: 13,
                    role: "assistant".to_string(),
                    content: "new trailing".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };

        poll_context_updates(&mut app, &mut runtime);

        assert_eq!(
            app.conversation_lines,
            vec![
                "assistant: existing".to_string(),
                "assistant: new trailing".to_string()
            ]
        );
        assert_eq!(
            app.rendered_context_message_ids,
            HashSet::from([10, 11, 12, 13])
        );
    }

    #[test]
    fn maybe_poll_context_updates_waits_for_command_action_to_finish() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["assistant: existing".to_string()],
            ..TuiSnapshot::default()
        });
        app.command_active = true;
        app.rendered_context_message_ids = HashSet::from([1]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 1,
                    role: "assistant".to_string(),
                    content: "existing".to_string(),
                },
                TuiContextMessage {
                    id: 2,
                    role: "assistant".to_string(),
                    content: "background update".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };
        let mut last_context_poll = Instant::now() - Duration::from_secs(2);
        let first_poll = last_context_poll;

        maybe_poll_context_updates(
            &mut app,
            &mut runtime,
            true,
            &mut last_context_poll,
            Instant::now(),
        );

        assert_eq!(
            app.conversation_lines,
            vec!["assistant: existing".to_string()]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1]));
        assert_eq!(last_context_poll, first_poll);

        app.command_active = false;
        let second_poll = Instant::now();
        maybe_poll_context_updates(
            &mut app,
            &mut runtime,
            true,
            &mut last_context_poll,
            second_poll,
        );

        assert_eq!(
            app.conversation_lines,
            vec![
                "assistant: existing".to_string(),
                "assistant: background update".to_string()
            ]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1, 2]));
        assert_eq!(last_context_poll, second_poll);
    }

    #[test]
    fn maybe_poll_context_updates_waits_for_prompt_stream_to_finish() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["assistant: existing".to_string()],
            ..TuiSnapshot::default()
        });
        app.prompt_active = true;
        app.rendered_context_message_ids = HashSet::from([1]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 1,
                    role: "assistant".to_string(),
                    content: "existing".to_string(),
                },
                TuiContextMessage {
                    id: 2,
                    role: "assistant".to_string(),
                    content: "background update".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };
        let mut last_context_poll = Instant::now() - Duration::from_secs(2);
        let first_poll = last_context_poll;

        maybe_poll_context_updates(
            &mut app,
            &mut runtime,
            true,
            &mut last_context_poll,
            Instant::now(),
        );

        assert_eq!(
            app.conversation_lines,
            vec!["assistant: existing".to_string()]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1]));
        assert_eq!(last_context_poll, first_poll);

        app.prompt_active = false;
        let second_poll = Instant::now();
        maybe_poll_context_updates(
            &mut app,
            &mut runtime,
            true,
            &mut last_context_poll,
            second_poll,
        );

        assert_eq!(
            app.conversation_lines,
            vec![
                "assistant: existing".to_string(),
                "assistant: background update".to_string()
            ]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1, 2]));
        assert_eq!(last_context_poll, second_poll);
    }

    #[test]
    fn prompt_completion_makes_context_polling_immediately_ready() {
        let mut app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["assistant: existing".to_string()],
            ..TuiSnapshot::default()
        });
        app.rendered_context_message_ids = HashSet::from([1]);
        let mut runtime = FakeRuntime {
            context_messages: vec![
                TuiContextMessage {
                    id: 1,
                    role: "assistant".to_string(),
                    content: "existing".to_string(),
                },
                TuiContextMessage {
                    id: 2,
                    role: "assistant".to_string(),
                    content: "background update".to_string(),
                },
            ],
            ..FakeRuntime::default()
        };
        let pending_prompt = None;
        let mut context_poll_ready = false;
        let now = Instant::now();
        let mut last_context_poll = now;

        maybe_enable_context_polling_after_prompt_completion(
            &pending_prompt,
            &mut context_poll_ready,
            &mut last_context_poll,
            now,
        );
        maybe_poll_context_updates(
            &mut app,
            &mut runtime,
            context_poll_ready,
            &mut last_context_poll,
            now,
        );

        assert!(context_poll_ready);
        assert_eq!(
            app.conversation_lines,
            vec![
                "assistant: existing".to_string(),
                "assistant: background update".to_string()
            ]
        );
        assert_eq!(app.rendered_context_message_ids, HashSet::from([1, 2]));
        assert_eq!(last_context_poll, now);
    }

    #[test]
    fn deferred_context_refresh_runs_once_after_deadline() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime::default();
        let mut deferred_context_refresh_at = Some(Instant::now());

        maybe_run_deferred_context_refresh(
            &mut app,
            &mut runtime,
            false,
            false,
            Instant::now(),
            &mut deferred_context_refresh_at,
        );

        assert_eq!(runtime.refresh_context_calls, 1);
        assert!(deferred_context_refresh_at.is_none());
    }

    #[test]
    fn deferred_context_refresh_waits_for_command_action_to_finish() {
        let mut app = TuiApp::bootstrap();
        app.command_active = true;
        let mut runtime = FakeRuntime::default();
        let scheduled_at = Instant::now();
        let mut deferred_context_refresh_at = Some(scheduled_at);

        maybe_run_deferred_context_refresh(
            &mut app,
            &mut runtime,
            false,
            true,
            Instant::now(),
            &mut deferred_context_refresh_at,
        );

        assert_eq!(runtime.refresh_context_calls, 0);
        assert_eq!(deferred_context_refresh_at, Some(scheduled_at));

        app.command_active = false;
        maybe_run_deferred_context_refresh(
            &mut app,
            &mut runtime,
            false,
            false,
            Instant::now(),
            &mut deferred_context_refresh_at,
        );

        assert_eq!(runtime.refresh_context_calls, 1);
        assert!(deferred_context_refresh_at.is_none());
    }

    #[test]
    fn runtime_tick_defers_context_refresh_until_prompt_stream_finishes() {
        let mut app = TuiApp::bootstrap();
        let mut runtime = FakeRuntime::default();
        let mut pending = Some(PendingPrompt {
            submitted_prompt: Some("hello runtime".to_string()),
            schedule_self_reflection: false,
            before_ids: HashSet::new(),
            stream: Box::new(FakePromptStream {
                updates: vec![PromptUpdate::Status("thinking...".to_string())],
                finalized_snapshot: TuiSnapshot::default(),
                cancelled_snapshot: TuiSnapshot::default(),
            }),
        });
        let scheduled_at = Instant::now();
        let mut deferred_context_refresh_at = Some(scheduled_at);
        let mut previous_background_status = None;

        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            scheduled_at,
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert!(pending.is_some());
        assert!(app.prompt_active);
        assert_eq!(runtime.refresh_context_calls, 0);
        assert_eq!(deferred_context_refresh_at, Some(scheduled_at));

        pending = None;
        app.prompt_active = false;
        let after_prompt = Instant::now();
        let result = drive_runtime_tick(
            &mut app,
            &mut runtime,
            &mut pending,
            after_prompt,
            &mut deferred_context_refresh_at,
            &mut previous_background_status,
        );

        assert_eq!(result, None);
        assert_eq!(runtime.refresh_context_calls, 1);
        assert!(deferred_context_refresh_at.is_none());
    }

    #[test]
    fn cancel_prompt_preserves_draft_text_and_refreshes_snapshot() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        app.focus = FocusTarget::Command(CommandPane::Conversation);
        app.input = "draft".to_string();
        apply_intent_with_runtime(&mut app, UiIntent::CancelPrompt, &mut runtime, &mut pending);

        assert_eq!(app.input, "draft");
        assert_eq!(app.status, "Chat stream cancelled");
        assert_eq!(app.focus, FocusTarget::Input);
        assert!(pending.is_none());
        assert!(
            app.conversation_lines
                .iter()
                .any(|line| line.contains("assistant: cancelled"))
        );
    }

    #[test]
    fn cancel_prompt_does_not_run_deferred_self_reflection() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello".to_string();
        let mut runtime = FakeRuntime::default();
        let mut pending = None;

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime, &mut pending);
        apply_intent_with_runtime(&mut app, UiIntent::CancelPrompt, &mut runtime, &mut pending);

        assert!(pending.is_none());
        assert_eq!(runtime.self_reflection_runs, 0);
        assert_eq!(app.status, "Chat stream cancelled");
    }
}
