use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
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
use ratatui::widgets::{Block, Borders, Paragraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarSection {
    Memories,
    Agenda,
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
    pub status: String,
    pub input: String,
    pub sidebar_section: SidebarSection,
    pub focus: FocusTarget,
    pub last_command_pane: CommandPane,
    pub conversation_lines: Vec<String>,
    pub memory_titles: Vec<String>,
    pub agenda_titles: Vec<String>,
    pub selected_sidebar_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiExit {
    Quit,
    Continue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarAction {
    Archive,
    Complete,
    Delete,
}

pub trait TuiRuntime {
    fn submit_prompt(&mut self, prompt: &str) -> Result<TuiSnapshot, String>;
    fn open_sidebar_item(&mut self, section: SidebarSection, title: &str)
    -> Result<String, String>;
    fn mutate_sidebar_item(
        &mut self,
        section: SidebarSection,
        title: &str,
        action: SidebarAction,
    ) -> Result<TuiSnapshot, String>;
}

pub fn run() -> io::Result<()> {
    run_with_snapshot(TuiSnapshot::default())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TuiSnapshot {
    pub conversation_lines: Vec<String>,
    pub memory_titles: Vec<String>,
    pub agenda_titles: Vec<String>,
    pub status: Option<String>,
}

pub fn run_with_snapshot(snapshot: TuiSnapshot) -> io::Result<()> {
    run_with_snapshot_and_runtime(snapshot, &mut NoopRuntime)
}

pub fn run_with_snapshot_and_runtime<R: TuiRuntime>(
    snapshot: TuiSnapshot,
    runtime: &mut R,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = TuiApp::from_snapshot(snapshot);
    let result = run_event_loop(&mut terminal, &mut app, runtime);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
    runtime: &mut impl TuiRuntime,
) -> io::Result<()> {
    loop {
        terminal.draw(|frame| {
            app.render(frame.area(), frame.buffer_mut());
        })?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        match event::read()? {
            Event::Key(key) => {
                if apply_key_event(app, key, runtime) == TuiExit::Quit {
                    return Ok(());
                }
            }
            Event::Resize(_, _) => {
                app.status = "terminal resized".to_string();
            }
            _ => {}
        }
    }
}

fn apply_key_event(app: &mut TuiApp, key: KeyEvent, runtime: &mut impl TuiRuntime) -> TuiExit {
    if key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.status = "quit".to_string();
        return TuiExit::Quit;
    }

    if app.focus == FocusTarget::Input {
        match key.code {
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.input.push(ch);
                app.status = "editing prompt".to_string();
                return TuiExit::Continue;
            }
            KeyCode::Backspace => {
                app.input.pop();
                app.status = "editing prompt".to_string();
                return TuiExit::Continue;
            }
            _ => {}
        }
    }

    if let Some(token) = key_event_token(key) {
        let intent = app.handle_key(token);
        apply_intent_with_runtime(app, intent, runtime);
    }

    TuiExit::Continue
}

fn apply_intent_with_runtime(app: &mut TuiApp, intent: UiIntent, runtime: &mut impl TuiRuntime) {
    match intent {
        UiIntent::SubmitPrompt => {
            let submitted = app.input.trim().to_string();
            if submitted.is_empty() {
                app.status = "prompt was empty".to_string();
                return;
            }
            match runtime.submit_prompt(&submitted) {
                Ok(snapshot) => {
                    app.apply_snapshot(snapshot);
                    app.input.clear();
                    app.status = format!("submitted prompt: {submitted}");
                }
                Err(error) => {
                    app.status = format!("prompt failed: {error}");
                }
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
                    app.conversation_lines.push(detail);
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
            status: "bootstrap".to_string(),
            input: String::new(),
            sidebar_section: SidebarSection::Memories,
            focus: FocusTarget::Input,
            last_command_pane: CommandPane::Conversation,
            conversation_lines: vec!["Conversation history and streaming output".to_string()],
            memory_titles: Vec::new(),
            agenda_titles: Vec::new(),
            selected_sidebar_index: 0,
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
        if let Some(status) = snapshot.status {
            self.status = status;
        }
        let active_len = self.active_sidebar_items().len();
        if self.selected_sidebar_index >= active_len {
            self.selected_sidebar_index = active_len.saturating_sub(1);
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let vertical = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(area);
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(36)])
            .split(vertical[0]);

        Paragraph::new(self.conversation_lines.join("\n"))
            .block(
                Block::default()
                    .title(self.title.as_str())
                    .borders(Borders::ALL),
            )
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

        let footer = format!("{} | {}", self.status, self.footer_hints());
        Paragraph::new(Line::styled(
            footer,
            Style::default().add_modifier(Modifier::BOLD),
        ))
        .render(vertical[2], buf);
    }

    fn sidebar_header(&self) -> String {
        match self.sidebar_section {
            SidebarSection::Memories => "Memories [active] | Agenda".to_string(),
            SidebarSection::Agenda => "Memories | Agenda [active]".to_string(),
        }
    }

    fn sidebar_text(&self) -> String {
        let mut lines = vec![self.sidebar_header()];
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
        }
    }

    pub fn footer_hints(&self) -> &'static str {
        match self.focus {
            FocusTarget::Input => "Esc command mode  Ctrl+M memories  Ctrl+A agenda  Ctrl+D quit",
            FocusTarget::Command(_) => {
                "Tab switch pane  j/k move  Enter open  c complete  d archive/delete  i/a/Esc chat mode"
            }
            FocusTarget::Unknown => "Recovering focus",
        }
    }

    pub fn handle_key(&mut self, key: &str) -> UiIntent {
        if self.focus == FocusTarget::Unknown {
            self.focus = FocusTarget::Input;
            return UiIntent::Noop;
        }

        match key {
            "ctrl+m" => {
                self.sidebar_section = SidebarSection::Memories;
                self.focus = FocusTarget::Command(CommandPane::Sidebar);
                self.last_command_pane = CommandPane::Sidebar;
                UiIntent::Noop
            }
            "ctrl+a" => {
                self.sidebar_section = SidebarSection::Agenda;
                self.focus = FocusTarget::Command(CommandPane::Sidebar);
                self.last_command_pane = CommandPane::Sidebar;
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
            UiIntent::HistoryPrevious => {
                self.status = "prompt history previous".to_string();
            }
            UiIntent::HistoryNext => {
                self.status = "prompt history next".to_string();
            }
            UiIntent::CompleteInput => {
                self.status = "input completion requested".to_string();
            }
            UiIntent::MoveUp => {
                if self.selected_sidebar_index > 0 {
                    self.selected_sidebar_index -= 1;
                }
                self.status = "moved selection up".to_string();
            }
            UiIntent::MoveDown => {
                let len = self.active_sidebar_items().len();
                if self.selected_sidebar_index + 1 < len {
                    self.selected_sidebar_index += 1;
                }
                self.status = "moved selection down".to_string();
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
                UiIntent::Noop
            }
            FocusTarget::Unknown => {
                self.focus = FocusTarget::Input;
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
                self.sidebar_section = SidebarSection::Memories;
                self.focus = FocusTarget::Command(CommandPane::Sidebar);
                self.last_command_pane = CommandPane::Sidebar;
                UiIntent::Noop
            }
            "g" => {
                self.sidebar_section = SidebarSection::Agenda;
                self.focus = FocusTarget::Command(CommandPane::Sidebar);
                self.last_command_pane = CommandPane::Sidebar;
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
                if self.focus == FocusTarget::Command(CommandPane::Sidebar)
                    && self.sidebar_section == SidebarSection::Agenda
                {
                    UiIntent::CompleteSelected
                } else {
                    UiIntent::Noop
                }
            }
            "d" => {
                if self.focus == FocusTarget::Command(CommandPane::Sidebar) {
                    match self.sidebar_section {
                        SidebarSection::Memories => UiIntent::ArchiveSelected,
                        SidebarSection::Agenda => UiIntent::DeleteSelected,
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
}

struct NoopRuntime;

impl TuiRuntime for NoopRuntime {
    fn submit_prompt(&mut self, _prompt: &str) -> Result<TuiSnapshot, String> {
        Ok(TuiSnapshot::default())
    }

    fn open_sidebar_item(
        &mut self,
        _section: SidebarSection,
        _title: &str,
    ) -> Result<String, String> {
        Ok("sidebar detail unavailable".to_string())
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    use super::{
        CommandPane, FocusTarget, SidebarAction, SidebarSection, TuiApp, TuiExit, TuiRuntime,
        TuiSnapshot, UiIntent, apply_intent_with_runtime, apply_key_event, key_event_token,
    };

    #[derive(Default)]
    struct FakeRuntime {
        submitted_prompts: Vec<String>,
        last_opened: Option<(SidebarSection, String)>,
        last_mutation: Option<(SidebarSection, String, SidebarAction)>,
    }

    impl TuiRuntime for FakeRuntime {
        fn submit_prompt(&mut self, prompt: &str) -> Result<TuiSnapshot, String> {
            self.submitted_prompts.push(prompt.to_string());
            Ok(TuiSnapshot {
                conversation_lines: vec![
                    format!("user: {prompt}"),
                    "assistant: runtime response".to_string(),
                ],
                memory_titles: vec!["Fresh Memory".to_string()],
                agenda_titles: vec!["Fresh Agenda".to_string()],
                status: Some("runtime updated".to_string()),
            })
        }

        fn open_sidebar_item(
            &mut self,
            section: SidebarSection,
            title: &str,
        ) -> Result<String, String> {
            self.last_opened = Some((section, title.to_string()));
            Ok(format!("opened detail for {title}"))
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
                status: Some("mutation updated".to_string()),
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
    }

    #[test]
    fn render_contains_core_elroy_panes() {
        let app = TuiApp::bootstrap();
        let text = rendered_text(&app);

        assert!(text.contains("Elroy"));
        assert!(text.contains("Relevant Context"));
        assert!(text.contains("Memories [active] | Agenda"));
        assert!(text.contains("Input"));
        assert!(text.contains("bootstrap"));
        assert!(text.contains("Esc command mode"));
    }

    #[test]
    fn render_switches_sidebar_label_when_agenda_is_active() {
        let mut app = TuiApp::bootstrap();
        app.sidebar_section = SidebarSection::Agenda;
        let text = rendered_text(&app);

        assert!(text.contains("Memories | Agenda [active]"));
    }

    #[test]
    fn snapshot_render_uses_persisted_conversation_and_sidebar_data() {
        let app = TuiApp::from_snapshot(TuiSnapshot {
            conversation_lines: vec!["user: hello".to_string(), "assistant: hi".to_string()],
            memory_titles: vec!["Runner Notes".to_string()],
            agenda_titles: vec!["Doctor Visit".to_string()],
            status: Some("loaded snapshot".to_string()),
        });
        let text = rendered_text(&app);

        assert!(text.contains("user: hello"));
        assert!(text.contains("assistant: hi"));
        assert!(text.contains("> Runner Notes"));
        assert!(text.contains("loaded snapshot"));
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
    fn apply_key_event_appends_input_and_submits_prompt() {
        let mut app = TuiApp::bootstrap();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
                &mut FakeRuntime::default()
            ),
            TuiExit::Continue
        );
        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
                &mut FakeRuntime::default()
            ),
            TuiExit::Continue
        );
        assert_eq!(app.input, "hi");

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                &mut FakeRuntime::default()
            ),
            TuiExit::Continue
        );
        assert!(app.status.contains("submitted prompt: hi"));
        assert!(app.input.is_empty());
    }

    #[test]
    fn ctrl_d_requests_quit() {
        let mut app = TuiApp::bootstrap();

        assert_eq!(
            apply_key_event(
                &mut app,
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
                &mut FakeRuntime::default()
            ),
            TuiExit::Quit
        );
        assert_eq!(app.status, "quit");
    }

    #[test]
    fn runtime_submit_replaces_snapshot_data() {
        let mut app = TuiApp::bootstrap();
        app.input = "hello runtime".to_string();
        let mut runtime = FakeRuntime::default();

        apply_intent_with_runtime(&mut app, UiIntent::SubmitPrompt, &mut runtime);

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
    fn runtime_open_appends_detail_to_conversation() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Runner Notes".to_string()];
        app.handle_key("ctrl+m");
        let mut runtime = FakeRuntime::default();

        apply_intent_with_runtime(&mut app, UiIntent::OpenSelected, &mut runtime);

        assert_eq!(
            runtime.last_opened,
            Some((SidebarSection::Memories, "Runner Notes".to_string()))
        );
        assert!(
            app.conversation_lines
                .iter()
                .any(|line| line.contains("opened detail for Runner Notes"))
        );
    }

    #[test]
    fn runtime_mutation_refreshes_snapshot_data() {
        let mut app = TuiApp::bootstrap();
        app.memory_titles = vec!["Runner Notes".to_string()];
        app.handle_key("ctrl+m");
        let mut runtime = FakeRuntime::default();

        apply_intent_with_runtime(&mut app, UiIntent::ArchiveSelected, &mut runtime);

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
    }
}
