use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use elroy_config::AppConfig;
use elroy_llm::{
    CompletionRequest, ConversationMessage, LiveModelClient, LlmError, MessageRole, StreamEvent,
    ToolCall,
};
use elroy_tools::{ExecutableToolRegistry, ToolRegistry, ToolSpec};

static BACKGROUND_STATUSES: OnceLock<Mutex<Vec<(String, String)>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppSession {
    pub user_id: String,
    pub assistant_name: String,
}

impl AppSession {
    pub fn new(user_id: impl Into<String>, assistant_name: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            assistant_name: assistant_name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnContext {
    pub request_id: String,
    pub session: AppSession,
    pub config: AppConfig,
}

impl TurnContext {
    pub fn new(request_id: impl Into<String>, session: AppSession, config: AppConfig) -> Self {
        Self {
            request_id: request_id.into(),
            session,
            config,
        }
    }

    pub fn memory_dir(&self) -> &std::path::Path {
        &self.config.memory_dir
    }
}

pub fn set_background_status(key: impl Into<String>, message: impl Into<String>) {
    let statuses = BACKGROUND_STATUSES.get_or_init(|| Mutex::new(Vec::new()));
    let mut statuses = statuses.lock().expect("background status lock should work");
    let key = key.into();
    let message = message.into();
    if let Some((_, existing_message)) = statuses
        .iter_mut()
        .find(|(existing_key, _)| *existing_key == key)
    {
        *existing_message = message;
    } else {
        statuses.push((key, message));
    }
}

pub fn clear_background_status(key: &str) {
    let statuses = BACKGROUND_STATUSES.get_or_init(|| Mutex::new(Vec::new()));
    let mut statuses = statuses.lock().expect("background status lock should work");
    statuses.retain(|(existing_key, _)| existing_key != key);
}

pub fn get_background_status() -> Option<String> {
    let statuses = BACKGROUND_STATUSES.get_or_init(|| Mutex::new(Vec::new()));
    let statuses = statuses.lock().expect("background status lock should work");
    statuses.first().map(|(_, message)| message.clone())
}

pub struct ConversationRequest<'a> {
    pub user_message: &'a str,
    pub tools: &'a [ToolSpec],
    pub transcript: &'a [ConversationMessage],
    pub force_tool: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationOptions<'a> {
    pub role: MessageRole,
    pub force_tool: Option<&'a str>,
}

impl Default for ConversationOptions<'_> {
    fn default() -> Self {
        Self {
            role: MessageRole::User,
            force_tool: None,
        }
    }
}

pub trait ModelClient {
    fn next_events(
        &self,
        request: ConversationRequest<'_>,
    ) -> Result<Vec<StreamEvent>, ModelClientError>;
}

pub trait StreamingModelClient {
    fn stream_events(
        &self,
        request: ConversationRequest<'_>,
    ) -> Result<Box<dyn Iterator<Item = Result<StreamEvent, ModelClientError>>>, ModelClientError>;
}

pub trait ToolExecutor {
    fn execute(&self, call: &ToolCall) -> StreamEvent;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationOrchestrator {
    pub max_tool_rounds: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRun {
    pub events: Vec<StreamEvent>,
    pub transcript: Vec<ConversationMessage>,
}

pub struct TurnEventStream {
    model: Box<dyn StreamingModelClient>,
    tools: Vec<ToolSpec>,
    tool_executor: Box<dyn ToolExecutor>,
    transcript: Vec<ConversationMessage>,
    message: String,
    force_tool: Option<String>,
    max_tool_rounds: usize,
    current_round: usize,
    current_stream: Option<Box<dyn Iterator<Item = Result<StreamEvent, ModelClientError>>>>,
    current_round_saw_tool_call: bool,
    pending: VecDeque<Result<StreamEvent, ModelClientError>>,
    events: Vec<StreamEvent>,
    finished: bool,
}

#[derive(Debug)]
pub enum ModelClientError {
    Provider(LlmError),
}

impl std::fmt::Display for ModelClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ModelClientError {}

impl From<LlmError> for ModelClientError {
    fn from(value: LlmError) -> Self {
        Self::Provider(value)
    }
}

#[derive(Debug)]
pub struct LiveProviderModel {
    client: LiveModelClient,
    system_prompt: String,
}

impl LiveProviderModel {
    pub fn new(client: LiveModelClient, system_prompt: impl Into<String>) -> Self {
        Self {
            client,
            system_prompt: system_prompt.into(),
        }
    }
}

impl ModelClient for LiveProviderModel {
    fn next_events(
        &self,
        request: ConversationRequest<'_>,
    ) -> Result<Vec<StreamEvent>, ModelClientError> {
        let tools = ToolRegistry::new(request.tools.to_vec());
        self.client
            .query(CompletionRequest {
                system_prompt: &self.system_prompt,
                messages: request.transcript,
                tools: &tools,
                force_tool: request.force_tool,
            })
            .map_err(ModelClientError::from)
    }
}

impl StreamingModelClient for LiveProviderModel {
    fn stream_events(
        &self,
        request: ConversationRequest<'_>,
    ) -> Result<Box<dyn Iterator<Item = Result<StreamEvent, ModelClientError>>>, ModelClientError>
    {
        let tools = ToolRegistry::new(request.tools.to_vec());
        self.client
            .query_stream(CompletionRequest {
                system_prompt: &self.system_prompt,
                messages: request.transcript,
                tools: &tools,
                force_tool: request.force_tool,
            })
            .map(|stream| {
                Box::new(stream.map(|event| event.map_err(ModelClientError::from)))
                    as Box<dyn Iterator<Item = Result<StreamEvent, ModelClientError>>>
            })
            .map_err(ModelClientError::from)
    }
}

#[derive(Clone)]
pub struct LocalToolExecutor {
    registry: ExecutableToolRegistry,
}

impl LocalToolExecutor {
    pub fn new(registry: ExecutableToolRegistry) -> Self {
        Self { registry }
    }
}

impl ToolExecutor for LocalToolExecutor {
    fn execute(&self, call: &ToolCall) -> StreamEvent {
        let result = self.registry.invoke(&call.name, &call.arguments_json);
        StreamEvent::AssistantToolResult {
            content: result.content,
            is_error: result.is_error,
        }
    }
}

impl ConversationOrchestrator {
    pub fn new(max_tool_rounds: usize) -> Self {
        Self { max_tool_rounds }
    }

    pub fn run_turn(
        &self,
        model: &dyn ModelClient,
        tools: &[ToolSpec],
        tool_executor: &dyn ToolExecutor,
        user_message: &str,
    ) -> Result<TurnRun, ModelClientError> {
        self.run_turn_with_transcript_and_role(
            model,
            tools,
            tool_executor,
            &[],
            MessageRole::User,
            user_message,
        )
    }

    pub fn run_turn_with_transcript(
        &self,
        model: &dyn ModelClient,
        tools: &[ToolSpec],
        tool_executor: &dyn ToolExecutor,
        existing_transcript: &[ConversationMessage],
        user_message: &str,
    ) -> Result<TurnRun, ModelClientError> {
        self.run_turn_with_transcript_and_role(
            model,
            tools,
            tool_executor,
            existing_transcript,
            MessageRole::User,
            user_message,
        )
    }

    pub fn run_turn_with_transcript_and_role(
        &self,
        model: &dyn ModelClient,
        tools: &[ToolSpec],
        tool_executor: &dyn ToolExecutor,
        existing_transcript: &[ConversationMessage],
        role: MessageRole,
        message: &str,
    ) -> Result<TurnRun, ModelClientError> {
        self.run_turn_with_transcript_and_options(
            model,
            tools,
            tool_executor,
            existing_transcript,
            ConversationOptions {
                role,
                force_tool: None,
            },
            message,
        )
    }

    pub fn run_turn_with_transcript_and_options(
        &self,
        model: &dyn ModelClient,
        tools: &[ToolSpec],
        tool_executor: &dyn ToolExecutor,
        existing_transcript: &[ConversationMessage],
        options: ConversationOptions<'_>,
        message: &str,
    ) -> Result<TurnRun, ModelClientError> {
        let mut events = Vec::new();
        let mut transcript = existing_transcript.to_vec();
        transcript.push(ConversationMessage::new(options.role, message));

        for _ in 0..=self.max_tool_rounds {
            let model_events = model.next_events(ConversationRequest {
                user_message: message,
                tools,
                transcript: &transcript,
                force_tool: options.force_tool,
            })?;

            let mut saw_tool_call = false;

            for event in model_events {
                match event {
                    StreamEvent::ToolCallRequested(call) => {
                        saw_tool_call = true;
                        events.push(StreamEvent::ToolCallRequested(call.clone()));
                        transcript.push(ConversationMessage::assistant_with_tool_calls(
                            "",
                            vec![call.clone()],
                        ));

                        events.push(StreamEvent::StatusUpdate {
                            content: format!("running {}...", call.name),
                        });
                        let tool_event = tool_executor.execute(&call);
                        transcript.push(tool_event_to_message(&call, &tool_event));
                        events.push(tool_event);
                        events.push(StreamEvent::StatusUpdate {
                            content: "thinking...".to_string(),
                        });
                    }
                    StreamEvent::AssistantResponse { content } => {
                        transcript.push(ConversationMessage::new(MessageRole::Assistant, &content));
                        events.push(StreamEvent::AssistantResponse { content });
                    }
                    StreamEvent::AssistantInternalThought { content } => {
                        events.push(StreamEvent::AssistantInternalThought { content });
                    }
                    StreamEvent::AssistantToolResult { content, is_error } => {
                        transcript.push(ConversationMessage::tool_result(
                            "model-generated-tool-result",
                            &content,
                        ));
                        events.push(StreamEvent::AssistantToolResult { content, is_error });
                    }
                    StreamEvent::StatusUpdate { content } => {
                        events.push(StreamEvent::StatusUpdate { content });
                    }
                }
            }

            if !saw_tool_call {
                break;
            }
        }

        Ok(TurnRun { events, transcript })
    }

    pub fn stream_turn_with_transcript_and_options(
        &self,
        model: Box<dyn StreamingModelClient>,
        tools: Vec<ToolSpec>,
        tool_executor: Box<dyn ToolExecutor>,
        existing_transcript: &[ConversationMessage],
        options: ConversationOptions<'_>,
        message: &str,
    ) -> Result<TurnEventStream, ModelClientError> {
        let mut transcript = existing_transcript.to_vec();
        transcript.push(ConversationMessage::new(options.role, message));

        let mut stream = TurnEventStream {
            model,
            tools,
            tool_executor,
            transcript,
            message: message.to_string(),
            force_tool: options.force_tool.map(str::to_string),
            max_tool_rounds: self.max_tool_rounds,
            current_round: 0,
            current_stream: None,
            current_round_saw_tool_call: false,
            pending: VecDeque::new(),
            events: Vec::new(),
            finished: false,
        };
        stream.start_next_round()?;
        Ok(stream)
    }
}

impl TurnEventStream {
    fn start_next_round(&mut self) -> Result<(), ModelClientError> {
        if self.current_round > self.max_tool_rounds {
            self.finished = true;
            self.current_stream = None;
            return Ok(());
        }
        self.current_round_saw_tool_call = false;
        self.current_stream = Some(self.model.stream_events(ConversationRequest {
            user_message: &self.message,
            tools: &self.tools,
            transcript: &self.transcript,
            force_tool: self.force_tool.as_deref(),
        })?);
        Ok(())
    }

    fn handle_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::ToolCallRequested(call) => {
                self.current_round_saw_tool_call = true;
                self.transcript
                    .push(ConversationMessage::assistant_with_tool_calls(
                        "",
                        vec![call.clone()],
                    ));
                self.events
                    .push(StreamEvent::ToolCallRequested(call.clone()));
                self.pending
                    .push_back(Ok(StreamEvent::ToolCallRequested(call.clone())));
                self.events.push(StreamEvent::StatusUpdate {
                    content: format!("running {}...", call.name),
                });
                self.pending.push_back(Ok(StreamEvent::StatusUpdate {
                    content: format!("running {}...", call.name),
                }));

                let tool_event = self.tool_executor.execute(&call);
                self.transcript
                    .push(tool_event_to_message(&call, &tool_event));
                self.events.push(tool_event.clone());
                self.pending.push_back(Ok(tool_event));
                self.events.push(StreamEvent::StatusUpdate {
                    content: "thinking...".to_string(),
                });
                self.pending.push_back(Ok(StreamEvent::StatusUpdate {
                    content: "thinking...".to_string(),
                }));
            }
            StreamEvent::AssistantResponse { content } => {
                self.transcript
                    .push(ConversationMessage::new(MessageRole::Assistant, &content));
                self.events.push(StreamEvent::AssistantResponse {
                    content: content.clone(),
                });
                self.pending
                    .push_back(Ok(StreamEvent::AssistantResponse { content }));
            }
            StreamEvent::AssistantInternalThought { content } => {
                self.events.push(StreamEvent::AssistantInternalThought {
                    content: content.clone(),
                });
                self.pending
                    .push_back(Ok(StreamEvent::AssistantInternalThought { content }));
            }
            StreamEvent::AssistantToolResult { content, is_error } => {
                self.transcript.push(ConversationMessage::tool_result(
                    "model-generated-tool-result",
                    &content,
                ));
                self.events.push(StreamEvent::AssistantToolResult {
                    content: content.clone(),
                    is_error,
                });
                self.pending
                    .push_back(Ok(StreamEvent::AssistantToolResult { content, is_error }));
            }
            StreamEvent::StatusUpdate { content } => {
                self.events.push(StreamEvent::StatusUpdate {
                    content: content.clone(),
                });
                self.pending
                    .push_back(Ok(StreamEvent::StatusUpdate { content }));
            }
        }
    }

    pub fn finish(mut self) -> Result<TurnRun, ModelClientError> {
        while self.next().is_some() {}
        Ok(TurnRun {
            events: self.events,
            transcript: self.transcript,
        })
    }
}

impl Iterator for TurnEventStream {
    type Item = Result<StreamEvent, ModelClientError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }

            let next_from_model = match self.current_stream.as_mut() {
                Some(stream) => stream.next(),
                None => None,
            };

            match next_from_model {
                Some(Ok(event)) => {
                    self.handle_event(event);
                }
                Some(Err(error)) => {
                    self.finished = true;
                    return Some(Err(error));
                }
                None => {
                    if self.current_round_saw_tool_call && self.current_round < self.max_tool_rounds
                    {
                        self.current_round += 1;
                        if let Err(error) = self.start_next_round() {
                            self.finished = true;
                            return Some(Err(error));
                        }
                        continue;
                    }
                    self.finished = true;
                    return None;
                }
            }
        }
    }
}

pub fn validated_transcript(messages: &[ConversationMessage]) -> Vec<ConversationMessage> {
    let mut validated = Vec::new();

    for (index, original) in messages.iter().enumerate() {
        let mut message = original.clone();
        normalize_message(&mut message);

        if message.role == MessageRole::Assistant {
            if let Some(tool_calls) = &message.tool_calls {
                let following_tool_ids = contiguous_following_tool_ids(messages, index + 1);
                let repaired_calls = tool_calls
                    .iter()
                    .filter(|call| following_tool_ids.iter().any(|id| id == &call.id))
                    .cloned()
                    .collect::<Vec<_>>();
                message.tool_calls = if repaired_calls.is_empty() {
                    None
                } else {
                    Some(repaired_calls)
                };
            }
            validated.push(message);
            continue;
        }

        if message.role == MessageRole::Tool {
            let Some(tool_call_id) = message.tool_call_id.as_deref() else {
                continue;
            };
            if has_assistant_tool_call(&validated, tool_call_id) {
                validated.push(message);
            }
            continue;
        }

        validated.push(message);
    }

    validated
}

fn normalize_message(message: &mut ConversationMessage) {
    if !matches!(message.role, MessageRole::Assistant)
        || message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| calls.is_empty())
    {
        message.tool_calls = None;
    }

    if !matches!(message.role, MessageRole::Tool) {
        message.tool_call_id = None;
    }
}

fn contiguous_following_tool_ids(
    messages: &[ConversationMessage],
    start_index: usize,
) -> Vec<String> {
    let mut ids = Vec::new();
    for message in messages.iter().skip(start_index) {
        if message.role != MessageRole::Tool {
            break;
        }
        if let Some(tool_call_id) = message.tool_call_id.as_ref() {
            ids.push(tool_call_id.clone());
        }
    }
    ids
}

fn has_assistant_tool_call(messages: &[ConversationMessage], tool_call_id: &str) -> bool {
    messages.iter().any(|message| {
        message.role == MessageRole::Assistant
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| calls.iter().any(|call| call.id == tool_call_id))
    })
}

fn tool_event_to_message(call: &ToolCall, event: &StreamEvent) -> ConversationMessage {
    match event {
        StreamEvent::AssistantToolResult { content, .. } => {
            ConversationMessage::tool_result(call.id.clone(), content)
        }
        StreamEvent::AssistantResponse { content } => {
            ConversationMessage::tool_result(call.id.clone(), content)
        }
        StreamEvent::AssistantInternalThought { content } => {
            ConversationMessage::tool_result(call.id.clone(), content)
        }
        StreamEvent::StatusUpdate { content } => {
            ConversationMessage::tool_result(call.id.clone(), content)
        }
        StreamEvent::ToolCallRequested(_) => ConversationMessage::tool_result(
            call.id.clone(),
            "tool executor requested another tool",
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{
        AppSession, TurnContext, clear_background_status, get_background_status,
        set_background_status,
    };
    use elroy_config::AppConfig;
    use elroy_llm::{ConversationMessage, MessageRole, StreamEvent, ToolCall};
    use elroy_tools::{
        ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolSpec,
    };

    use super::{
        ConversationOrchestrator, ConversationRequest, ModelClient, StreamingModelClient,
        ToolExecutor, validated_transcript,
    };

    struct FakeModel {
        responses: RefCell<Vec<Vec<StreamEvent>>>,
    }

    impl FakeModel {
        fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
            Self {
                responses: RefCell::new(responses),
            }
        }
    }

    impl ModelClient for FakeModel {
        fn next_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, super::ModelClientError> {
            Ok(self.responses.borrow_mut().remove(0))
        }
    }

    impl StreamingModelClient for FakeModel {
        fn stream_events(
            &self,
            _request: ConversationRequest<'_>,
        ) -> Result<
            Box<dyn Iterator<Item = Result<StreamEvent, super::ModelClientError>>>,
            super::ModelClientError,
        > {
            let events = self.responses.borrow_mut().remove(0);
            Ok(Box::new(events.into_iter().map(Ok)))
        }
    }

    struct InspectingModel {
        force_tool: RefCell<Option<String>>,
    }

    impl InspectingModel {
        fn new() -> Self {
            Self {
                force_tool: RefCell::new(None),
            }
        }
    }

    impl ModelClient for InspectingModel {
        fn next_events(
            &self,
            request: ConversationRequest<'_>,
        ) -> Result<Vec<StreamEvent>, super::ModelClientError> {
            *self.force_tool.borrow_mut() = request.force_tool.map(str::to_string);
            Ok(vec![StreamEvent::AssistantResponse {
                content: "Forced tool acknowledged.".to_string(),
            }])
        }
    }

    struct FakeToolExecutor;

    impl ToolExecutor for FakeToolExecutor {
        fn execute(&self, call: &ToolCall) -> StreamEvent {
            StreamEvent::AssistantToolResult {
                content: format!("tool:{}:{}", call.name, call.arguments_json),
                is_error: false,
            }
        }
    }

    fn weather_tool() -> ToolSpec {
        ToolSpec::new(
            "get_weather",
            "Get the weather for a location.",
            JsonSchema::object(
                [("location", serde_json::json!({"type": "string"}))],
                ["location"],
            ),
        )
    }

    #[test]
    fn turn_context_exposes_memory_dir_from_config() {
        let session = AppSession::new("user-1", "Elroy");
        let config = AppConfig::defaults();
        let turn = TurnContext::new("req-1", session, config.clone());

        assert_eq!(turn.memory_dir(), config.memory_dir.as_path());
    }

    #[test]
    fn session_and_turn_have_separate_lifetimes() {
        let session = AppSession::new("user-1", "Elroy");
        let turn = TurnContext::new("req-1", session.clone(), AppConfig::defaults());

        assert_eq!(turn.session, session);
        assert_eq!(turn.request_id, "req-1");
    }

    #[test]
    fn orchestrator_runs_model_then_tools_then_followup_model_turn() {
        let model = FakeModel::new(vec![
            vec![
                StreamEvent::StatusUpdate {
                    content: "thinking".to_string(),
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
            ],
            vec![StreamEvent::AssistantResponse {
                content: "The weather in Paris is sunny.".to_string(),
            }],
        ]);
        let orchestrator = ConversationOrchestrator::new(2);
        let turn_run = orchestrator
            .run_turn(&model, &[weather_tool()], &FakeToolExecutor, "weather?")
            .expect("turn should succeed");

        assert_eq!(
            turn_run.events,
            vec![
                StreamEvent::StatusUpdate {
                    content: "thinking".to_string(),
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
                StreamEvent::StatusUpdate {
                    content: "running get_weather...".to_string(),
                },
                StreamEvent::AssistantToolResult {
                    content: "tool:get_weather:{\"location\":\"Paris\"}".to_string(),
                    is_error: false,
                },
                StreamEvent::StatusUpdate {
                    content: "thinking...".to_string(),
                },
                StreamEvent::AssistantResponse {
                    content: "The weather in Paris is sunny.".to_string(),
                },
            ]
        );
        assert_eq!(turn_run.transcript[0].role, MessageRole::User);
        assert_eq!(
            turn_run.transcript[1].tool_calls.as_ref().map(Vec::len),
            Some(1)
        );
        assert_eq!(turn_run.transcript[2].role, MessageRole::Tool);
        assert_eq!(turn_run.transcript[3].role, MessageRole::Assistant);
    }

    #[test]
    fn orchestrator_streams_model_then_tools_then_followup_model_turn() {
        let model = FakeModel::new(vec![
            vec![
                StreamEvent::StatusUpdate {
                    content: "thinking".to_string(),
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
            ],
            vec![StreamEvent::AssistantResponse {
                content: "The weather in Paris is sunny.".to_string(),
            }],
        ]);
        let orchestrator = ConversationOrchestrator::new(2);
        let stream = orchestrator
            .stream_turn_with_transcript_and_options(
                Box::new(model),
                vec![weather_tool()],
                Box::new(FakeToolExecutor),
                &[],
                super::ConversationOptions::default(),
                "weather?",
            )
            .expect("stream should start");

        let turn_run = stream.finish().expect("stream should finish");
        assert_eq!(
            turn_run.events,
            vec![
                StreamEvent::StatusUpdate {
                    content: "thinking".to_string(),
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
                StreamEvent::StatusUpdate {
                    content: "running get_weather...".to_string(),
                },
                StreamEvent::AssistantToolResult {
                    content: "tool:get_weather:{\"location\":\"Paris\"}".to_string(),
                    is_error: false,
                },
                StreamEvent::StatusUpdate {
                    content: "thinking...".to_string(),
                },
                StreamEvent::AssistantResponse {
                    content: "The weather in Paris is sunny.".to_string(),
                },
            ]
        );
        assert_eq!(turn_run.transcript[0].role, MessageRole::User);
        assert_eq!(
            turn_run.transcript[1].tool_calls.as_ref().map(Vec::len),
            Some(1)
        );
        assert_eq!(turn_run.transcript[2].role, MessageRole::Tool);
        assert_eq!(turn_run.transcript[3].role, MessageRole::Assistant);
    }

    #[test]
    fn streaming_turn_yields_tool_status_updates_in_order() {
        let model = FakeModel::new(vec![
            vec![StreamEvent::ToolCallRequested(ToolCall {
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                arguments_json: "{\"location\":\"Paris\"}".to_string(),
            })],
            vec![StreamEvent::AssistantResponse {
                content: "Done.".to_string(),
            }],
        ]);
        let orchestrator = ConversationOrchestrator::new(1);
        let mut stream = orchestrator
            .stream_turn_with_transcript_and_options(
                Box::new(model),
                vec![weather_tool()],
                Box::new(FakeToolExecutor),
                &[],
                super::ConversationOptions::default(),
                "weather?",
            )
            .expect("stream should start");

        assert!(matches!(
            stream.next(),
            Some(Ok(StreamEvent::ToolCallRequested(call))) if call.name == "get_weather"
        ));
        assert!(matches!(
            stream.next(),
            Some(Ok(StreamEvent::StatusUpdate { content })) if content == "running get_weather..."
        ));
        assert!(matches!(
            stream.next(),
            Some(Ok(StreamEvent::AssistantToolResult { content, is_error }))
                if content == "tool:get_weather:{\"location\":\"Paris\"}" && !is_error
        ));
        assert!(matches!(
            stream.next(),
            Some(Ok(StreamEvent::StatusUpdate { content })) if content == "thinking..."
        ));
        assert!(matches!(
            stream.next(),
            Some(Ok(StreamEvent::AssistantResponse { content })) if content == "Done."
        ));
    }

    #[test]
    fn orchestrator_returns_direct_response_without_tools() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Hello.".to_string(),
        }]]);
        let orchestrator = ConversationOrchestrator::new(1);
        let turn_run = orchestrator
            .run_turn(&model, &[weather_tool()], &FakeToolExecutor, "hi")
            .expect("turn should succeed");

        assert_eq!(
            turn_run.events,
            vec![StreamEvent::AssistantResponse {
                content: "Hello.".to_string(),
            }]
        );
        assert_eq!(turn_run.transcript.len(), 2);
        assert_eq!(turn_run.transcript[0].role, MessageRole::User);
        assert_eq!(turn_run.transcript[1].role, MessageRole::Assistant);
    }

    #[test]
    fn orchestrator_can_continue_from_existing_transcript() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Welcome back.".to_string(),
        }]]);
        let orchestrator = ConversationOrchestrator::new(1);
        let existing = vec![ConversationMessage::new(
            MessageRole::Assistant,
            "Earlier reply",
        )];

        let turn_run = orchestrator
            .run_turn_with_transcript(
                &model,
                &[weather_tool()],
                &FakeToolExecutor,
                &existing,
                "hi",
            )
            .expect("turn should succeed");

        assert_eq!(turn_run.transcript.len(), 3);
        assert_eq!(turn_run.transcript[0].role, MessageRole::Assistant);
        assert_eq!(turn_run.transcript[1].role, MessageRole::User);
        assert_eq!(turn_run.transcript[2].role, MessageRole::Assistant);
    }

    #[test]
    fn orchestrator_can_append_non_user_input_roles() {
        let model = FakeModel::new(vec![vec![StreamEvent::AssistantResponse {
            content: "Acknowledged.".to_string(),
        }]]);
        let orchestrator = ConversationOrchestrator::new(0);

        let turn_run = orchestrator
            .run_turn_with_transcript_and_role(
                &model,
                &[],
                &FakeToolExecutor,
                &[],
                MessageRole::System,
                "system bootstrap",
            )
            .expect("turn should succeed");

        assert_eq!(turn_run.transcript.len(), 2);
        assert_eq!(turn_run.transcript[0].role, MessageRole::System);
        assert_eq!(
            turn_run.transcript[0].content.as_deref(),
            Some("system bootstrap")
        );
        assert_eq!(turn_run.transcript[1].role, MessageRole::Assistant);
    }

    #[test]
    fn orchestrator_passes_force_tool_to_model_request() {
        let model = InspectingModel::new();
        let orchestrator = ConversationOrchestrator::new(0);

        let _ = orchestrator
            .run_turn_with_transcript_and_options(
                &model,
                &[weather_tool()],
                &FakeToolExecutor,
                &[],
                super::ConversationOptions {
                    role: MessageRole::User,
                    force_tool: Some("get_weather"),
                },
                "weather now",
            )
            .expect("turn should succeed");

        assert_eq!(model.force_tool.borrow().as_deref(), Some("get_weather"));
    }

    #[test]
    fn local_tool_executor_uses_executable_registry_results() {
        let registry =
            ExecutableToolRegistry::new(vec![ExecutableTool::new(weather_tool(), |_| {
                ToolExecutionResult::success("sunny")
            })]);
        let executor = super::LocalToolExecutor::new(registry);

        let result = executor.execute(&ToolCall {
            id: "call-1".to_string(),
            name: "get_weather".to_string(),
            arguments_json: "{\"location\":\"Paris\"}".to_string(),
        });

        assert_eq!(
            result,
            StreamEvent::AssistantToolResult {
                content: "sunny".to_string(),
                is_error: false,
            }
        );
    }

    #[test]
    fn validated_transcript_drops_orphan_tool_messages() {
        let validated = validated_transcript(&[
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::tool_result("missing-call", "{\"ok\":true}"),
        ]);

        assert_eq!(validated.len(), 1);
        assert_eq!(validated[0].role, MessageRole::User);
    }

    #[test]
    fn validated_transcript_repairs_missing_assistant_tool_calls() {
        let validated = validated_transcript(&[
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }],
            ),
            ConversationMessage::new(MessageRole::Assistant, "plain reply"),
        ]);

        assert_eq!(validated.len(), 3);
        assert_eq!(validated[1].role, MessageRole::Assistant);
        assert!(validated[1].tool_calls.is_none());
    }

    #[test]
    fn validated_transcript_keeps_multiple_matching_tool_results() {
        let validated = validated_transcript(&[
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![
                    ToolCall {
                        id: "call-1".to_string(),
                        name: "first_tool".to_string(),
                        arguments_json: "{}".to_string(),
                    },
                    ToolCall {
                        id: "call-2".to_string(),
                        name: "second_tool".to_string(),
                        arguments_json: "{}".to_string(),
                    },
                ],
            ),
            ConversationMessage::tool_result("call-1", "{\"first\":true}"),
            ConversationMessage::tool_result("call-2", "{\"second\":true}"),
        ]);

        assert_eq!(validated.len(), 4);
        assert_eq!(validated[2].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(validated[3].tool_call_id.as_deref(), Some("call-2"));
    }

    #[test]
    fn validated_transcript_drops_tool_result_with_wrong_tool_call_id() {
        let validated = validated_transcript(&[
            ConversationMessage::new(MessageRole::User, "hello"),
            ConversationMessage::assistant_with_tool_calls(
                "",
                vec![
                    ToolCall {
                        id: "call-1".to_string(),
                        name: "first_tool".to_string(),
                        arguments_json: "{}".to_string(),
                    },
                    ToolCall {
                        id: "call-2".to_string(),
                        name: "second_tool".to_string(),
                        arguments_json: "{}".to_string(),
                    },
                ],
            ),
            ConversationMessage::tool_result("call-1", "{\"first\":true}"),
            ConversationMessage::tool_result("wrong-id", "{\"wrong\":true}"),
        ]);

        assert_eq!(validated.len(), 3);
        assert_eq!(validated[2].tool_call_id.as_deref(), Some("call-1"));
        assert!(
            !validated
                .iter()
                .any(|message| message.tool_call_id.as_deref() == Some("wrong-id"))
        );
    }

    #[test]
    fn background_status_registry_tracks_first_active_message() {
        clear_background_status("memory-sync");
        clear_background_status("other");
        assert!(get_background_status().is_none());

        set_background_status("memory-sync", "syncing memories...");
        assert_eq!(
            get_background_status().as_deref(),
            Some("syncing memories...")
        );

        set_background_status("other", "other background task");
        assert_eq!(
            get_background_status().as_deref(),
            Some("syncing memories...")
        );

        clear_background_status("memory-sync");
        assert_eq!(
            get_background_status().as_deref(),
            Some("other background task")
        );

        clear_background_status("other");
        assert!(get_background_status().is_none());
    }
}
