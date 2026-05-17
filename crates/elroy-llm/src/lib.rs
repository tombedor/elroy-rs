use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::io::{BufRead, BufReader};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use elroy_tools::ToolRegistry;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ToolCall {
    pub fn arguments(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::from_str(&self.arguments_json)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub chat_model: Option<String>,
    pub id: Option<i64>,
    pub created_at_unix: i64,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

impl ConversationMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            chat_model: None,
            id: None,
            created_at_unix: unix_timestamp_now(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(content.into()),
            chat_model: None,
            id: None,
            created_at_unix: unix_timestamp_now(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            chat_model: None,
            id: None,
            created_at_unix: unix_timestamp_now(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.role != MessageRole::Assistant
            && self
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
        {
            return Err(format!(
                "only assistant messages can have tool calls, found {:?}",
                self.role
            ));
        }
        if self.role != MessageRole::Tool && self.tool_call_id.is_some() {
            return Err(format!(
                "only tool messages can have tool call ids, found {:?}",
                self.role
            ));
        }
        if self.role == MessageRole::Tool && self.tool_call_id.is_none() {
            return Err("tool messages must include a tool_call_id".to_string());
        }
        Ok(())
    }

    pub fn to_openai_message(&self) -> Value {
        match self.role {
            MessageRole::System => json!({"role": "system", "content": self.content}),
            MessageRole::User => json!({"role": "user", "content": self.content}),
            MessageRole::Assistant => {
                let mut message = json!({"role": "assistant", "content": self.content});
                if let Some(tool_calls) = &self.tool_calls {
                    message["tool_calls"] = Value::Array(
                        tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments_json,
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                message
            }
            MessageRole::Tool => json!({
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }),
        }
    }

    pub fn to_anthropic_message(&self) -> Value {
        match self.role {
            MessageRole::System => {
                json!({"role": "user", "content": [{"type": "text", "text": self.content}]})
            }
            MessageRole::User => {
                json!({"role": "user", "content": [{"type": "text", "text": self.content}]})
            }
            MessageRole::Assistant => {
                let mut content = Vec::new();
                if let Some(text) = &self.content
                    && !text.is_empty()
                {
                    content.push(json!({"type": "text", "text": text}));
                }
                if let Some(tool_calls) = &self.tool_calls {
                    for call in tool_calls {
                        let input = call.arguments().unwrap_or_else(|_| json!({}));
                        content.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": input,
                        }));
                    }
                }
                json!({"role": "assistant", "content": content})
            }
            MessageRole::Tool => json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": self.content.clone().unwrap_or_default(),
                }],
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamEvent {
    AssistantResponse { content: String },
    AssistantInternalThought { content: String },
    AssistantToolResult { content: String, is_error: bool },
    StatusUpdate { content: String },
    ToolCallRequested(ToolCall),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

impl PartialToolCall {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            arguments_json: String::new(),
        }
    }

    pub fn push_name_fragment(&mut self, fragment: &str) {
        self.name.push_str(fragment);
    }

    pub fn push_arguments_fragment(&mut self, fragment: &str) {
        self.arguments_json.push_str(fragment);
    }

    pub fn try_complete(&self) -> Option<ToolCall> {
        if self.name.is_empty() {
            return None;
        }

        serde_json::from_str::<serde_json::Value>(&self.arguments_json)
            .ok()
            .map(|_| ToolCall {
                id: self.id.clone(),
                name: self.name.clone(),
                arguments_json: self.arguments_json.clone(),
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub anthropic_api_version: Option<String>,
    pub timeout_seconds: u64,
    pub max_output_tokens: u32,
}

impl ProviderConfig {
    pub fn openai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            provider: Provider::OpenAi,
            model: model.into(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1/responses".to_string(),
            anthropic_api_version: None,
            timeout_seconds: 60,
            max_output_tokens: 2048,
        }
    }

    pub fn anthropic(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            provider: Provider::Anthropic,
            model: model.into(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1/messages".to_string(),
            anthropic_api_version: Some("2023-06-01".to_string()),
            timeout_seconds: 60,
            max_output_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProviderConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionRequest<'a> {
    pub system_prompt: &'a str,
    pub messages: &'a [ConversationMessage],
    pub tools: &'a ToolRegistry,
    pub force_tool: Option<&'a str>,
}

#[derive(Debug)]
pub struct LiveModelClient {
    http: Client,
    config: ProviderConfig,
}

#[derive(Debug)]
pub struct LiveEmbeddingClient {
    http: Client,
    config: EmbeddingProviderConfig,
}

impl LiveModelClient {
    pub fn new(config: ProviderConfig) -> Result<Self, LlmError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(LlmError::HttpClient)?;
        Ok(Self { http, config })
    }

    pub fn query(&self, request: CompletionRequest<'_>) -> Result<Vec<StreamEvent>, LlmError> {
        let response = match self.config.provider {
            Provider::OpenAi => self
                .http
                .post(&self.config.base_url)
                .bearer_auth(&self.config.api_key)
                .json(&build_openai_responses_request(
                    &self.config.model,
                    request.system_prompt,
                    request.messages,
                    request.tools,
                    self.config.max_output_tokens,
                    request.force_tool,
                ))
                .send()
                .map_err(LlmError::HttpRequest)?,
            Provider::Anthropic => self
                .http
                .post(&self.config.base_url)
                .header("x-api-key", &self.config.api_key)
                .header(
                    "anthropic-version",
                    self.config
                        .anthropic_api_version
                        .as_deref()
                        .unwrap_or("2023-06-01"),
                )
                .json(&build_anthropic_messages_request(
                    &self.config.model,
                    request.system_prompt,
                    request.messages,
                    request.tools,
                    self.config.max_output_tokens,
                    request.force_tool,
                ))
                .send()
                .map_err(LlmError::HttpRequest)?,
        };

        let status = response.status();
        let body = response.text().map_err(LlmError::ReadResponse)?;
        if !status.is_success() {
            return Err(LlmError::Api {
                provider: self.config.provider,
                status,
                body,
            });
        }

        let payload = serde_json::from_str::<Value>(&body).map_err(LlmError::ParseResponse)?;
        match self.config.provider {
            Provider::OpenAi => parse_openai_response(&payload),
            Provider::Anthropic => parse_anthropic_response(&payload),
        }
    }

    pub fn query_stream(
        &self,
        request: CompletionRequest<'_>,
    ) -> Result<Box<dyn Iterator<Item = Result<StreamEvent, LlmError>>>, LlmError> {
        let response = match self.config.provider {
            Provider::OpenAi => {
                let mut payload = build_openai_responses_request(
                    &self.config.model,
                    request.system_prompt,
                    request.messages,
                    request.tools,
                    self.config.max_output_tokens,
                    request.force_tool,
                );
                payload["stream"] = json!(true);
                self.http
                    .post(&self.config.base_url)
                    .bearer_auth(&self.config.api_key)
                    .json(&payload)
                    .send()
                    .map_err(LlmError::HttpRequest)?
            }
            Provider::Anthropic => {
                let mut payload = build_anthropic_messages_request(
                    &self.config.model,
                    request.system_prompt,
                    request.messages,
                    request.tools,
                    self.config.max_output_tokens,
                    request.force_tool,
                );
                payload["stream"] = json!(true);
                self.http
                    .post(&self.config.base_url)
                    .header("x-api-key", &self.config.api_key)
                    .header(
                        "anthropic-version",
                        self.config
                            .anthropic_api_version
                            .as_deref()
                            .unwrap_or("2023-06-01"),
                    )
                    .json(&payload)
                    .send()
                    .map_err(LlmError::HttpRequest)?
            }
        };

        let status = response.status();
        if !status.is_success() {
            let body = response.text().map_err(LlmError::ReadResponse)?;
            return Err(LlmError::Api {
                provider: self.config.provider,
                status,
                body,
            });
        }

        let reader = BufReader::new(response);
        match self.config.provider {
            Provider::OpenAi => Ok(Box::new(OpenAiStream::new(reader))),
            Provider::Anthropic => Ok(Box::new(AnthropicStream::new(reader))),
        }
    }
}

impl LiveEmbeddingClient {
    pub fn new(config: EmbeddingProviderConfig) -> Result<Self, LlmError> {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(LlmError::HttpClient)?;
        Ok(Self { http, config })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let response = self
            .http
            .post(&self.config.base_url)
            .bearer_auth(&self.config.api_key)
            .json(&json!({
                "model": self.config.model,
                "input": text,
            }))
            .send()
            .map_err(LlmError::HttpRequest)?;

        let status = response.status();
        let body = response.text().map_err(LlmError::ReadResponse)?;
        if !status.is_success() {
            return Err(LlmError::Api {
                provider: Provider::OpenAi,
                status,
                body,
            });
        }

        let payload = serde_json::from_str::<Value>(&body).map_err(LlmError::ParseResponse)?;
        parse_openai_embedding_response(&payload)
    }
}

#[derive(Debug)]
pub enum LlmError {
    HttpClient(reqwest::Error),
    HttpRequest(reqwest::Error),
    ReadResponse(reqwest::Error),
    ReadStream(std::io::Error),
    ParseResponse(serde_json::Error),
    Api {
        provider: Provider,
        status: StatusCode,
        body: String,
    },
}

impl Display for LlmError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HttpClient(error) => write!(f, "failed to construct HTTP client: {error}"),
            Self::HttpRequest(error) => write!(f, "HTTP request failed: {error}"),
            Self::ReadResponse(error) => write!(f, "failed to read HTTP response: {error}"),
            Self::ReadStream(error) => write!(f, "failed to read streaming response: {error}"),
            Self::ParseResponse(error) => write!(f, "failed to parse model response: {error}"),
            Self::Api {
                provider,
                status,
                body,
            } => write!(
                f,
                "{provider:?} API returned {status}: {}",
                trim_error_body(body)
            ),
        }
    }
}

impl std::error::Error for LlmError {}

fn trim_error_body(body: &str) -> String {
    const LIMIT: usize = 240;
    if body.len() <= LIMIT {
        body.to_string()
    } else {
        format!("{}...", &body[..LIMIT])
    }
}

fn parse_openai_embedding_response(payload: &Value) -> Result<Vec<f32>, LlmError> {
    let embedding = payload
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("embedding"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            LlmError::ParseResponse(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing embedding array",
            )))
        })?;

    embedding
        .iter()
        .map(|value| {
            value.as_f64().map(|number| number as f32).ok_or_else(|| {
                LlmError::ParseResponse(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "embedding value was not numeric",
                )))
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SseEvent {
    event_type: Option<String>,
    data: String,
}

struct SseEventReader<R> {
    reader: R,
    finished: bool,
}

impl<R> SseEventReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            finished: false,
        }
    }
}

impl<R: BufRead> Iterator for SseEventReader<R> {
    type Item = Result<SseEvent, LlmError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut event_type = None;
        let mut data_lines = Vec::new();

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    self.finished = true;
                    if data_lines.is_empty() && event_type.is_none() {
                        return None;
                    }
                    break;
                }
                Ok(_) => {}
                Err(error) => {
                    self.finished = true;
                    return Some(Err(LlmError::ReadStream(error)));
                }
            }

            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                if data_lines.is_empty() && event_type.is_none() {
                    continue;
                }
                break;
            }
            if let Some(value) = trimmed.strip_prefix("event:") {
                event_type = Some(value.trim().to_string());
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("data:") {
                data_lines.push(value.trim_start().to_string());
            }
        }

        let data = data_lines.join("\n");
        if data == "[DONE]" {
            self.finished = true;
            return None;
        }

        Some(Ok(SseEvent { event_type, data }))
    }
}

struct OpenAiStream<R> {
    events: SseEventReader<R>,
    queued: VecDeque<Result<StreamEvent, LlmError>>,
    emitted_message_fallbacks: HashSet<String>,
    partial_tool_calls: HashMap<String, PartialToolCall>,
    saw_text_delta: bool,
}

impl<R> OpenAiStream<R> {
    fn new(reader: R) -> Self {
        Self {
            events: SseEventReader::new(reader),
            queued: VecDeque::new(),
            emitted_message_fallbacks: HashSet::new(),
            partial_tool_calls: HashMap::new(),
            saw_text_delta: false,
        }
    }
}

impl<R: BufRead> Iterator for OpenAiStream<R> {
    type Item = Result<StreamEvent, LlmError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.queued.pop_front() {
                return Some(event);
            }

            let raw = self.events.next()?;
            let raw = match raw {
                Ok(raw) => raw,
                Err(error) => return Some(Err(error)),
            };

            let payload = match serde_json::from_str::<Value>(&raw.data) {
                Ok(payload) => payload,
                Err(error) => return Some(Err(LlmError::ParseResponse(error))),
            };

            match raw.event_type.as_deref() {
                Some("response.output_text.delta") => {
                    if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                        && !delta.is_empty()
                    {
                        self.saw_text_delta = true;
                        self.queued.push_back(Ok(StreamEvent::AssistantResponse {
                            content: delta.to_string(),
                        }));
                    }
                }
                Some("response.reasoning_summary_text.delta")
                | Some("response.reasoning.delta") => {
                    if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                        && !delta.is_empty()
                    {
                        self.queued
                            .push_back(Ok(StreamEvent::AssistantInternalThought {
                                content: delta.to_string(),
                            }));
                    }
                }
                Some("response.output_item.added") => {
                    if let Some(item) = payload.get("item")
                        && item.get("type").and_then(Value::as_str) == Some("function_call")
                    {
                        let id = item
                            .get("id")
                            .or_else(|| item.get("call_id"))
                            .and_then(Value::as_str)
                            .unwrap_or("openai-call")
                            .to_string();
                        let mut partial = PartialToolCall::new(id.clone());
                        if let Some(name) = item.get("name").and_then(Value::as_str) {
                            partial.push_name_fragment(name);
                        }
                        if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                            partial.push_arguments_fragment(arguments);
                        }
                        self.partial_tool_calls.insert(id, partial);
                    }
                }
                Some("response.function_call_arguments.delta") => {
                    let Some(id) = payload
                        .get("item_id")
                        .or_else(|| payload.get("call_id"))
                        .or_else(|| payload.get("id"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };
                    if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                        && let Some(partial) = self.partial_tool_calls.get_mut(id)
                    {
                        partial.push_arguments_fragment(delta);
                    }
                }
                Some("response.output_item.done") => {
                    let Some(item) = payload.get("item") else {
                        continue;
                    };
                    match item.get("type").and_then(Value::as_str) {
                        Some("function_call") => {
                            let call_id = item
                                .get("call_id")
                                .or_else(|| item.get("id"))
                                .and_then(Value::as_str)
                                .unwrap_or("openai-call");
                            let name = item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("tool_call");
                            let arguments_json = item
                                .get("arguments")
                                .map(|value| {
                                    value
                                        .as_str()
                                        .map(ToString::to_string)
                                        .unwrap_or_else(|| value.to_string())
                                })
                                .unwrap_or_else(|| {
                                    self.partial_tool_calls
                                        .remove(call_id)
                                        .map(|partial| partial.arguments_json)
                                        .unwrap_or_else(|| "{}".to_string())
                                });
                            self.partial_tool_calls.remove(call_id);
                            self.queued
                                .push_back(Ok(StreamEvent::ToolCallRequested(ToolCall {
                                    id: call_id.to_string(),
                                    name: name.to_string(),
                                    arguments_json,
                                })));
                        }
                        Some("message") if !self.saw_text_delta => {
                            let text = item
                                .get("content")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter_map(|block| block.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n");
                            if !text.is_empty()
                                && self.emitted_message_fallbacks.insert(text.clone())
                            {
                                self.queued.push_back(Ok(StreamEvent::AssistantResponse {
                                    content: text,
                                }));
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}

struct AnthropicStream<R> {
    events: SseEventReader<R>,
    queued: VecDeque<Result<StreamEvent, LlmError>>,
    partial_tool_calls: HashMap<usize, PartialToolCall>,
}

impl<R> AnthropicStream<R> {
    fn new(reader: R) -> Self {
        Self {
            events: SseEventReader::new(reader),
            queued: VecDeque::new(),
            partial_tool_calls: HashMap::new(),
        }
    }
}

impl<R: BufRead> Iterator for AnthropicStream<R> {
    type Item = Result<StreamEvent, LlmError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.queued.pop_front() {
                return Some(event);
            }

            let raw = self.events.next()?;
            let raw = match raw {
                Ok(raw) => raw,
                Err(error) => return Some(Err(error)),
            };
            let payload = match serde_json::from_str::<Value>(&raw.data) {
                Ok(payload) => payload,
                Err(error) => return Some(Err(LlmError::ParseResponse(error))),
            };

            match raw.event_type.as_deref() {
                Some("content_block_start") => {
                    let index = payload
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as usize;
                    let Some(block) = payload.get("content_block") else {
                        continue;
                    };
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("anthropic-call");
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool_call");
                        let mut partial = PartialToolCall::new(id);
                        partial.push_name_fragment(name);
                        if let Some(input) = block.get("input")
                            && !input.is_null()
                        {
                            partial.push_arguments_fragment(&input.to_string());
                        }
                        self.partial_tool_calls.insert(index, partial);
                    }
                }
                Some("content_block_delta") => {
                    let index = payload
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as usize;
                    let Some(delta) = payload.get("delta") else {
                        continue;
                    };
                    match delta.get("type").and_then(Value::as_str) {
                        Some("text_delta") => {
                            if let Some(text) = delta.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                self.queued.push_back(Ok(StreamEvent::AssistantResponse {
                                    content: text.to_string(),
                                }));
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(text) = delta.get("thinking").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                self.queued
                                    .push_back(Ok(StreamEvent::AssistantInternalThought {
                                        content: text.to_string(),
                                    }));
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(partial_json) =
                                delta.get("partial_json").and_then(Value::as_str)
                                && let Some(partial) = self.partial_tool_calls.get_mut(&index)
                            {
                                partial.push_arguments_fragment(partial_json);
                            }
                        }
                        _ => {}
                    }
                }
                Some("content_block_stop") => {
                    let index = payload
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as usize;
                    if let Some(partial) = self.partial_tool_calls.remove(&index)
                        && let Some(call) = partial.try_complete()
                    {
                        self.queued
                            .push_back(Ok(StreamEvent::ToolCallRequested(call)));
                    }
                }
                _ => {}
            }
        }
    }
}

fn unix_timestamp_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

pub fn build_openai_request(
    system_prompt: &str,
    messages: &[ConversationMessage],
    tools: &ToolRegistry,
    force_tool: Option<&str>,
) -> Value {
    let mut request_messages = vec![json!({"role": "system", "content": system_prompt})];
    request_messages.extend(
        messages
            .iter()
            .filter(|message| message.role != MessageRole::System)
            .map(ConversationMessage::to_openai_message),
    );

    let mut request = json!({
        "messages": request_messages,
        "tools": tools.openai_definitions(),
    });
    if let Some(force_tool) = force_tool {
        request["tool_choice"] = json!({"type": "function", "name": force_tool});
    }
    request
}

pub fn build_anthropic_request(
    system_prompt: &str,
    messages: &[ConversationMessage],
    tools: &ToolRegistry,
    force_tool: Option<&str>,
) -> Value {
    let mut request = json!({
        "system": system_prompt,
        "messages": messages
            .iter()
            .filter(|message| message.role != MessageRole::System)
            .map(ConversationMessage::to_anthropic_message)
            .collect::<Vec<_>>(),
        "tools": tools.anthropic_definitions(),
    });
    if let Some(force_tool) = force_tool {
        request["tool_choice"] = json!({"type": "tool", "name": force_tool});
    }
    request
}

pub fn build_openai_responses_request(
    model: &str,
    system_prompt: &str,
    messages: &[ConversationMessage],
    tools: &ToolRegistry,
    max_output_tokens: u32,
    force_tool: Option<&str>,
) -> Value {
    let mut input = Vec::new();
    if !system_prompt.is_empty() {
        input.push(json!({
            "role": "system",
            "content": system_prompt,
        }));
    }
    for message in messages
        .iter()
        .filter(|message| message.role != MessageRole::System)
    {
        input.extend(openai_input_items(message));
    }

    let mut request = json!({
        "model": model,
        "input": input,
        "tools": tools.openai_definitions(),
        "parallel_tool_calls": false,
        "max_output_tokens": max_output_tokens,
    });
    if let Some(force_tool) = force_tool {
        request["tool_choice"] = json!({"type": "function", "name": force_tool});
    }
    request
}

pub fn build_anthropic_messages_request(
    model: &str,
    system_prompt: &str,
    messages: &[ConversationMessage],
    tools: &ToolRegistry,
    max_output_tokens: u32,
    force_tool: Option<&str>,
) -> Value {
    let mut request = json!({
        "model": model,
        "max_tokens": max_output_tokens,
        "system": system_prompt,
        "messages": messages
            .iter()
            .filter(|message| message.role != MessageRole::System)
            .map(ConversationMessage::to_anthropic_message)
            .collect::<Vec<_>>(),
        "tools": tools.anthropic_definitions(),
    });
    if let Some(force_tool) = force_tool {
        request["tool_choice"] = json!({"type": "tool", "name": force_tool});
    }
    request
}

fn openai_input_items(message: &ConversationMessage) -> Vec<Value> {
    match message.role {
        MessageRole::System => vec![json!({
            "role": "system",
            "content": message.content.clone().unwrap_or_default(),
        })],
        MessageRole::User => vec![json!({
            "role": "user",
            "content": message.content.clone().unwrap_or_default(),
        })],
        MessageRole::Assistant => {
            let mut items = Vec::new();
            if let Some(content) = &message.content
                && !content.is_empty()
            {
                items.push(json!({
                    "role": "assistant",
                    "content": content,
                }));
            }
            if let Some(tool_calls) = &message.tool_calls {
                items.extend(tool_calls.iter().map(|call| {
                    json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments_json,
                    })
                }));
            }
            items
        }
        MessageRole::Tool => vec![json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content.clone().unwrap_or_default(),
        })],
    }
}

pub fn parse_openai_response(payload: &Value) -> Result<Vec<StreamEvent>, LlmError> {
    let mut events = Vec::new();

    if let Some(output_text) = payload.get("output_text").and_then(Value::as_str)
        && !output_text.is_empty()
    {
        events.push(StreamEvent::AssistantResponse {
            content: output_text.to_string(),
        });
    }

    if let Some(output) = payload.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    let text = item
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|block| {
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .map(ToString::to_string)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty()
                        && !events.iter().any(|event| {
                            matches!(event, StreamEvent::AssistantResponse { content } if content == &text)
                        })
                    {
                        events.push(StreamEvent::AssistantResponse { content: text });
                    }
                }
                Some("function_call") => {
                    let call_id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or("openai-call");
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("tool_call");
                    let arguments_json = item
                        .get("arguments")
                        .map(|value| {
                            value
                                .as_str()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| value.to_string())
                        })
                        .unwrap_or_else(|| "{}".to_string());
                    events.push(StreamEvent::ToolCallRequested(ToolCall {
                        id: call_id.to_string(),
                        name: name.to_string(),
                        arguments_json,
                    }));
                }
                Some("reasoning") => {
                    if let Some(summary) = item
                        .get("summary")
                        .and_then(Value::as_array)
                        .and_then(|parts| parts.first())
                        .and_then(|part| part.get("text"))
                        .and_then(Value::as_str)
                    {
                        events.push(StreamEvent::AssistantInternalThought {
                            content: summary.to_string(),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    Ok(events)
}

pub fn parse_anthropic_response(payload: &Value) -> Result<Vec<StreamEvent>, LlmError> {
    let mut events = Vec::new();

    if let Some(content) = payload.get("content").and_then(Value::as_array) {
        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        events.push(StreamEvent::AssistantResponse {
                            content: text.to_string(),
                        });
                    }
                }
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                        events.push(StreamEvent::AssistantInternalThought {
                            content: text.to_string(),
                        });
                    }
                }
                Some("tool_use") => {
                    let arguments_json = block
                        .get("input")
                        .map(Value::to_string)
                        .unwrap_or_else(|| "{}".to_string());
                    let call = ToolCall {
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("anthropic-call")
                            .to_string(),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool_call")
                            .to_string(),
                        arguments_json,
                    };
                    events.push(StreamEvent::ToolCallRequested(call));
                }
                _ => {}
            }
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use elroy_tools::{JsonSchema, ToolRegistry, ToolSpec};
    use mockito::{Matcher, Server};
    use serde_json::json;

    use super::{
        CompletionRequest, ConversationMessage, EmbeddingProviderConfig, LiveEmbeddingClient,
        LiveModelClient, LlmError, MessageRole, PartialToolCall, Provider, ProviderConfig,
        StreamEvent, ToolCall, build_anthropic_messages_request, build_anthropic_request,
        build_openai_request, build_openai_responses_request, parse_anthropic_response,
        parse_openai_response,
    };

    fn weather_registry() -> ToolRegistry {
        ToolRegistry::new(vec![ToolSpec::new(
            "get_weather",
            "Get the weather for a location.",
            JsonSchema::object([("location", json!({"type": "string"}))], ["location"]),
        )])
    }

    #[test]
    fn partial_tool_call_completes_once_arguments_are_valid_json() {
        let mut partial = PartialToolCall::new("call-1");
        partial.push_name_fragment("get_weather");
        partial.push_arguments_fragment("{\"location\":");
        assert_eq!(partial.try_complete(), None);

        partial.push_arguments_fragment("\"Paris\"}");
        let completed = partial.try_complete().expect("tool call should complete");

        assert_eq!(completed.name, "get_weather");
        assert_eq!(
            completed.arguments().expect("json should parse"),
            json!({"location": "Paris"})
        );
    }

    #[test]
    fn partial_tool_call_requires_name_before_completion() {
        let mut partial = PartialToolCall::new("call-2");
        partial.push_arguments_fragment("{\"location\":\"Paris\"}");

        assert_eq!(partial.try_complete(), None);
    }

    #[test]
    fn stream_event_can_wrap_tool_call_requests() {
        let event = StreamEvent::ToolCallRequested(ToolCall {
            id: "call-3".to_string(),
            name: "get_weather".to_string(),
            arguments_json: "{\"location\":\"Paris\"}".to_string(),
        });

        match event {
            StreamEvent::ToolCallRequested(call) => {
                assert_eq!(call.id, "call-3");
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("expected tool call request event"),
        }
    }

    #[test]
    fn openai_request_uses_system_prompt_and_ignores_transcript_system_messages() {
        let request = build_openai_request(
            "You are Elroy.",
            &[
                ConversationMessage::new(MessageRole::System, "Old system"),
                ConversationMessage::new(MessageRole::User, "Hello"),
            ],
            &weather_registry(),
            None,
        );

        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["messages"][1]["role"], "user");
        assert_eq!(request["messages"].as_array().map(Vec::len), Some(2));
        assert_eq!(request["tools"][0]["type"], "function");
    }

    #[test]
    fn anthropic_request_maps_tool_results_into_user_content_blocks() {
        let request = build_anthropic_request(
            "You are Elroy.",
            &[ConversationMessage::tool_result("call-1", "{\"temp\":25}")],
            &weather_registry(),
            None,
        );

        assert_eq!(request["system"], "You are Elroy.");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(request["tools"][0]["name"], "get_weather");
    }

    #[test]
    fn anthropic_request_maps_assistant_tool_calls_to_tool_use_blocks() {
        let request = build_anthropic_request(
            "You are Elroy.",
            &[ConversationMessage::assistant_with_tool_calls(
                "Thinking",
                vec![ToolCall {
                    id: "call-9".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }],
            )],
            &weather_registry(),
            None,
        );

        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(request["messages"][0]["content"][1]["type"], "tool_use");
        assert_eq!(request["messages"][0]["content"][1]["name"], "get_weather");
    }

    #[test]
    fn context_message_validation_matches_python_role_rules() {
        let mut invalid = ConversationMessage::new(MessageRole::User, "hello");
        invalid.tool_calls = Some(vec![ToolCall {
            id: "call-1".to_string(),
            name: "get_weather".to_string(),
            arguments_json: "{\"location\":\"Paris\"}".to_string(),
        }]);
        assert!(invalid.validate().is_err());

        let mut missing_tool_id = ConversationMessage::new(MessageRole::Tool, "done");
        missing_tool_id.tool_call_id = None;
        assert!(missing_tool_id.validate().is_err());

        let valid = ConversationMessage::tool_result("call-1", "done");
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn anthropic_tool_use_message_does_not_emit_empty_text_block() {
        let mut message = ConversationMessage::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call-9".to_string(),
                name: "get_weather".to_string(),
                arguments_json: "{\"location\":\"Paris\"}".to_string(),
            }],
        );
        message.content = None;

        let request =
            build_anthropic_request("You are Elroy.", &[message], &weather_registry(), None);

        assert_eq!(
            request["messages"][0]["content"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(request["messages"][0]["content"][0]["type"], "tool_use");
    }

    #[test]
    fn openai_responses_request_uses_input_items_and_function_call_output_items() {
        let request = build_openai_responses_request(
            "gpt-5.4",
            "You are Elroy.",
            &[
                ConversationMessage::new(MessageRole::System, "Old system"),
                ConversationMessage::new(MessageRole::User, "Hello"),
                ConversationMessage::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "call-1".to_string(),
                        name: "get_weather".to_string(),
                        arguments_json: "{\"location\":\"Paris\"}".to_string(),
                    }],
                ),
                ConversationMessage::tool_result("call-1", "{\"temp\":25}"),
            ],
            &weather_registry(),
            1024,
            None,
        );

        assert_eq!(request["model"], "gpt-5.4");
        assert_eq!(request["input"][0]["role"], "system");
        assert_eq!(request["input"][1]["role"], "user");
        assert_eq!(request["input"][2]["type"], "function_call");
        assert_eq!(request["input"][3]["type"], "function_call_output");
        assert_eq!(request["input"].as_array().map(Vec::len), Some(4));
        assert_eq!(request["max_output_tokens"], 1024);
    }

    #[test]
    fn openai_responses_request_can_force_specific_tool() {
        let request = build_openai_responses_request(
            "gpt-5.4",
            "You are Elroy.",
            &[ConversationMessage::new(MessageRole::User, "Hello")],
            &weather_registry(),
            1024,
            Some("get_weather"),
        );

        assert_eq!(request["tool_choice"]["type"], "function");
        assert_eq!(request["tool_choice"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_messages_request_includes_model_and_max_tokens() {
        let request = build_anthropic_messages_request(
            "claude-sonnet-4-20250514",
            "You are Elroy.",
            &[ConversationMessage::new(MessageRole::User, "Hello")],
            &weather_registry(),
            2048,
            None,
        );

        assert_eq!(request["model"], "claude-sonnet-4-20250514");
        assert_eq!(request["max_tokens"], 2048);
        assert_eq!(request["system"], "You are Elroy.");
    }

    #[test]
    fn anthropic_messages_request_can_force_specific_tool() {
        let request = build_anthropic_messages_request(
            "claude-sonnet-4-20250514",
            "You are Elroy.",
            &[ConversationMessage::new(MessageRole::User, "Hello")],
            &weather_registry(),
            2048,
            Some("get_weather"),
        );

        assert_eq!(request["tool_choice"]["type"], "tool");
        assert_eq!(request["tool_choice"]["name"], "get_weather");
    }

    #[test]
    fn parse_openai_response_extracts_text_and_function_calls() {
        let payload = json!({
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Paris\"}"
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "It is sunny."}
                    ]
                }
            ]
        });

        let events = parse_openai_response(&payload).expect("response should parse");

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::ToolCallRequested(_)));
        assert!(matches!(events[1], StreamEvent::AssistantResponse { .. }));
    }

    #[test]
    fn parse_anthropic_response_extracts_text_and_tool_use_blocks() {
        let payload = json!({
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "Paris"}}
            ]
        });

        let events = parse_anthropic_response(&payload).expect("response should parse");

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::AssistantResponse { .. }));
        assert!(matches!(events[1], StreamEvent::ToolCallRequested(_)));
    }

    #[test]
    fn live_openai_client_sends_authorization_header_and_parses_output() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer openai-key")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_body(
                json!({
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello from OpenAI."}]
                    }]
                })
                .to_string(),
            )
            .create();

        let client = LiveModelClient::new(ProviderConfig {
            provider: Provider::OpenAi,
            model: "gpt-5.4".to_string(),
            api_key: "openai-key".to_string(),
            base_url: format!("{}/responses", server.url()),
            anthropic_api_version: None,
            timeout_seconds: 5,
            max_output_tokens: 512,
        })
        .expect("client should construct");

        let events = client
            .query(CompletionRequest {
                system_prompt: "You are Elroy.",
                messages: &[ConversationMessage::new(MessageRole::User, "Hello")],
                tools: &weather_registry(),
                force_tool: None,
            })
            .expect("request should succeed");

        mock.assert();
        assert_eq!(
            events,
            vec![StreamEvent::AssistantResponse {
                content: "Hello from OpenAI.".to_string()
            }]
        );
    }

    #[test]
    fn live_openai_embedding_client_sends_authorization_header_and_parses_embedding() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/embeddings")
            .match_header("authorization", "Bearer openai-key")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_body(
                json!({
                    "data": [{
                        "embedding": [0.1, 0.2, 0.3]
                    }]
                })
                .to_string(),
            )
            .create();

        let client = LiveEmbeddingClient::new(EmbeddingProviderConfig {
            model: "text-embedding-3-small".to_string(),
            api_key: "openai-key".to_string(),
            base_url: format!("{}/embeddings", server.url()),
            timeout_seconds: 5,
        })
        .expect("client should construct");

        let embedding = client
            .embed("What belongs in my workout kit?")
            .expect("embed ok");

        mock.assert();
        assert_eq!(embedding, vec![0.1_f32, 0.2_f32, 0.3_f32]);
    }

    #[test]
    fn live_anthropic_client_sends_required_headers_and_parses_tool_use() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "anthropic-key")
            .match_header("anthropic-version", "2023-06-01")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_body(
                json!({
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }]
                })
                .to_string(),
            )
            .create();

        let client = LiveModelClient::new(ProviderConfig {
            provider: Provider::Anthropic,
            model: "claude-sonnet-4-20250514".to_string(),
            api_key: "anthropic-key".to_string(),
            base_url: format!("{}/messages", server.url()),
            anthropic_api_version: Some("2023-06-01".to_string()),
            timeout_seconds: 5,
            max_output_tokens: 512,
        })
        .expect("client should construct");

        let events = client
            .query(CompletionRequest {
                system_prompt: "You are Elroy.",
                messages: &[ConversationMessage::new(MessageRole::User, "Hello")],
                tools: &weather_registry(),
                force_tool: None,
            })
            .expect("request should succeed");

        mock.assert();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::ToolCallRequested(_)));
    }

    #[test]
    fn live_client_surfaces_non_success_responses() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/responses")
            .with_status(401)
            .with_body("{\"error\":\"bad key\"}")
            .create();

        let client = LiveModelClient::new(ProviderConfig {
            provider: Provider::OpenAi,
            model: "gpt-5.4".to_string(),
            api_key: "bad-key".to_string(),
            base_url: format!("{}/responses", server.url()),
            anthropic_api_version: None,
            timeout_seconds: 5,
            max_output_tokens: 512,
        })
        .expect("client should construct");

        let error = client
            .query(CompletionRequest {
                system_prompt: "You are Elroy.",
                messages: &[ConversationMessage::new(MessageRole::User, "Hello")],
                tools: &weather_registry(),
                force_tool: None,
            })
            .expect_err("request should fail");

        mock.assert();
        assert!(matches!(
            error,
            LlmError::Api {
                provider: Provider::OpenAi,
                ..
            }
        ));
    }

    #[test]
    fn live_openai_client_streams_text_and_tool_calls() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/responses")
            .match_header("authorization", "Bearer openai-key")
            .match_body(Matcher::PartialJson(json!({"stream": true})))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(
                concat!(
                    "event: response.output_text.delta\n",
                    "data: {\"delta\":\"Hello\"}\n\n",
                    "event: response.output_text.delta\n",
                    "data: {\"delta\":\" there\"}\n\n",
                    "event: response.output_item.done\n",
                    "data: {\"item\":{\"type\":\"function_call\",\"call_id\":\"call-1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}\n\n",
                    "data: [DONE]\n\n"
                ),
            )
            .create();

        let client = LiveModelClient::new(ProviderConfig {
            provider: Provider::OpenAi,
            model: "gpt-5.4".to_string(),
            api_key: "openai-key".to_string(),
            base_url: format!("{}/responses", server.url()),
            anthropic_api_version: None,
            timeout_seconds: 5,
            max_output_tokens: 512,
        })
        .expect("client should construct");

        let events = client
            .query_stream(CompletionRequest {
                system_prompt: "You are Elroy.",
                messages: &[ConversationMessage::new(MessageRole::User, "Hello")],
                tools: &weather_registry(),
                force_tool: None,
            })
            .expect("stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("stream should parse");

        mock.assert();
        assert_eq!(
            events,
            vec![
                StreamEvent::AssistantResponse {
                    content: "Hello".to_string()
                },
                StreamEvent::AssistantResponse {
                    content: " there".to_string()
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "call-1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
            ]
        );
    }

    #[test]
    fn live_anthropic_client_streams_text_and_tool_calls() {
        let mut server = Server::new();
        let mock = server
            .mock("POST", "/messages")
            .match_header("x-api-key", "anthropic-key")
            .match_body(Matcher::PartialJson(json!({"stream": true})))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(
                concat!(
                    "event: content_block_delta\n",
                    "data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check.\"}}\n\n",
                    "event: content_block_start\n",
                    "data: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}\n\n",
                    "event: content_block_stop\n",
                    "data: {\"index\":1}\n\n"
                ),
            )
            .create();

        let client = LiveModelClient::new(ProviderConfig {
            provider: Provider::Anthropic,
            model: "claude-sonnet-4-20250514".to_string(),
            api_key: "anthropic-key".to_string(),
            base_url: format!("{}/messages", server.url()),
            anthropic_api_version: Some("2023-06-01".to_string()),
            timeout_seconds: 5,
            max_output_tokens: 512,
        })
        .expect("client should construct");

        let events = client
            .query_stream(CompletionRequest {
                system_prompt: "You are Elroy.",
                messages: &[ConversationMessage::new(MessageRole::User, "Hello")],
                tools: &weather_registry(),
                force_tool: None,
            })
            .expect("stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("stream should parse");

        mock.assert();
        assert_eq!(
            events,
            vec![
                StreamEvent::AssistantResponse {
                    content: "Let me check.".to_string()
                },
                StreamEvent::ToolCallRequested(ToolCall {
                    id: "toolu_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments_json: "{\"location\":\"Paris\"}".to_string(),
                }),
            ]
        );
    }
}
