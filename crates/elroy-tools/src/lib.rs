use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum JsonSchema {
    Object {
        properties: Map<String, Value>,
        required: Vec<String>,
        additional_properties: bool,
    },
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: JsonSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    pub fn openai_definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.as_json_schema(),
                "strict": true,
            }
        })
    }

    pub fn anthropic_definition(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.as_json_schema(),
        })
    }
}

impl JsonSchema {
    pub fn object(
        properties: impl IntoIterator<Item = (impl Into<String>, Value)>,
        required: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let properties = properties
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect::<Map<_, _>>();
        let required = required.into_iter().map(Into::into).collect::<Vec<_>>();

        Self::Object {
            properties,
            required,
            additional_properties: false,
        }
    }

    pub fn as_json_schema(&self) -> Value {
        match self {
            Self::Object {
                properties,
                required,
                additional_properties,
            } => json!({
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": additional_properties,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRegistry {
    specs: Vec<ToolSpec>,
}

impl ToolRegistry {
    pub fn new(specs: Vec<ToolSpec>) -> Self {
        Self { specs }
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn openai_definitions(&self) -> Vec<Value> {
        self.specs.iter().map(ToolSpec::openai_definition).collect()
    }

    pub fn anthropic_definitions(&self) -> Vec<Value> {
        self.specs
            .iter()
            .map(ToolSpec::anthropic_definition)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecutionResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolExecutionResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

#[derive(Clone)]
pub struct ExecutableTool {
    spec: ToolSpec,
    executor: Arc<dyn Fn(Value) -> ToolExecutionResult + Send + Sync>,
}

impl ExecutableTool {
    pub fn new(
        spec: ToolSpec,
        executor: impl Fn(Value) -> ToolExecutionResult + Send + Sync + 'static,
    ) -> Self {
        Self {
            spec,
            executor: Arc::new(executor),
        }
    }

    pub fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    pub fn execute(&self, arguments: Value) -> ToolExecutionResult {
        (self.executor)(arguments)
    }
}

#[derive(Clone)]
pub struct ExecutableToolRegistry {
    tools: Vec<ExecutableTool>,
}

impl ExecutableToolRegistry {
    pub fn new(tools: Vec<ExecutableTool>) -> Self {
        Self { tools }
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|tool| tool.spec.clone()).collect()
    }

    pub fn invoke(&self, name: &str, arguments_json: &str) -> ToolExecutionResult {
        let Some(tool) = self.tools.iter().find(|tool| tool.spec.name == name) else {
            return ToolExecutionResult::error(format!("unknown tool: {name}"));
        };

        let arguments = match serde_json::from_str::<Value>(arguments_json) {
            Ok(arguments) => arguments,
            Err(error) => {
                return ToolExecutionResult::error(format!(
                    "invalid arguments for tool {name}: {error}"
                ));
            }
        };

        tool.execute(arguments)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ExecutableTool, ExecutableToolRegistry, JsonSchema, ToolExecutionResult, ToolRegistry,
        ToolSpec,
    };

    fn weather_tool() -> ToolSpec {
        ToolSpec::new(
            "get_weather",
            "Get the weather for a location.",
            JsonSchema::object(
                [(
                    "location",
                    json!({
                        "type": "string",
                        "description": "City and region, such as San Francisco, CA",
                    }),
                )],
                ["location"],
            ),
        )
    }

    #[test]
    fn openai_definition_uses_function_wrapper_and_strict_schema() {
        let tool = weather_tool();
        let definition = tool.openai_definition();

        assert_eq!(definition["type"], "function");
        assert_eq!(definition["function"]["name"], "get_weather");
        assert_eq!(definition["function"]["strict"], true);
        assert_eq!(
            definition["function"]["parameters"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn anthropic_definition_uses_input_schema_shape() {
        let tool = weather_tool();
        let definition = tool.anthropic_definition();

        assert_eq!(definition["name"], "get_weather");
        assert_eq!(definition["input_schema"]["type"], "object");
        assert_eq!(definition["input_schema"]["required"], json!(["location"]));
    }

    #[test]
    fn registry_projects_definitions_for_each_provider() {
        let registry = ToolRegistry::new(vec![weather_tool()]);

        assert_eq!(registry.specs().len(), 1);
        assert_eq!(registry.openai_definitions().len(), 1);
        assert_eq!(registry.anthropic_definitions().len(), 1);
    }

    #[test]
    fn executable_tool_registry_invokes_registered_tool() {
        let registry =
            ExecutableToolRegistry::new(vec![ExecutableTool::new(weather_tool(), |arguments| {
                let location = arguments["location"].as_str().unwrap_or("unknown");
                ToolExecutionResult::success(format!("sunny in {location}"))
            })]);

        let result = registry.invoke("get_weather", "{\"location\":\"Paris\"}");

        assert_eq!(result, ToolExecutionResult::success("sunny in Paris"));
        assert_eq!(registry.specs().len(), 1);
    }

    #[test]
    fn executable_tool_registry_rejects_unknown_tools_and_bad_json() {
        let registry = ExecutableToolRegistry::new(vec![]);

        let unknown = registry.invoke("missing_tool", "{}");
        assert!(unknown.is_error);
        assert!(unknown.content.contains("unknown tool"));

        let registry =
            ExecutableToolRegistry::new(vec![ExecutableTool::new(weather_tool(), |_| {
                ToolExecutionResult::success("ok")
            })]);
        let invalid = registry.invoke("get_weather", "{not json}");
        assert!(invalid.is_error);
        assert!(invalid.content.contains("invalid arguments"));
    }
}
