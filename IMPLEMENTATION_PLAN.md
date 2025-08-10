# Elroy-RS V0 Implementation Plan

Based on analysis of REQUIREMENTS.md and the original elroy Python codebase, this document outlines the detailed implementation plan for elroy-rs V0.

All functionality should work and pass tests before proceeding to subsequent steps. Do NOT include any mock data in real implementation of app.

## Architecture Overview

**Core Components:**
- **Database Layer**: Diesel ORM with PostgreSQL, separate models for Memory, Reminder, ContextMessage, etc.
- **LLM Layer**: Async clients for Anthropic (`anthropic-sdk-rust`) and OpenAI (`async-openai`) APIs
- **API Layer**: HTTP endpoints for chat and memo processing
- **CLI Layer**: Rich terminal UI using `ratatui` for interactive chat
- **Tool System**: Plugin architecture supporting synthetic tool calls

## Implementation Stages

### Stage 1: Foundation (Database & Core Models)
**Tasks:**
- Set up Cargo.toml with required dependencies
- Create Diesel schema for core entities (users, memories, reminders, context_messages)
- Implement database models with proper relationships
- Set up migration system
- Create database connection management

**Tests:**
- Database connection and migration tests
- Model serialization/deserialization tests
- Basic CRUD operations for all entities

### Stage 2: LLM Integration
**Tasks:**
- Implement unified LLM client trait supporting both Anthropic and OpenAI
- Add streaming response support
- Create tool call parsing and accumulation logic
- Implement embedding generation for semantic search

**Tests:**
- LLM client connection tests (mocked)
- Tool call parsing tests with sample responses
- Streaming response handling tests

### Stage 3: Core Chat Functionality
**Tasks:**
- Implement chat message processing loop
- Create context message validation and management
- Add tool execution framework
- Build synthetic tool call system (key difference from Python version)

**Tests:**
- Chat message processing with various input types
- Context message validation edge cases
- Tool call execution success/failure scenarios

### Stage 4: Memory & Reminder System
**Tasks:**
- Implement `ingest_memo` tool that classifies text into memories/reminders
- Create memory and reminder CRUD operations
- Add semantic search for memory retrieval
- Build reminder scheduling and due reminder detection

**Tests:**
- Memo ingestion with various text inputs
- Memory semantic search accuracy
- Reminder trigger time calculations and due detection

### Stage 5: API Endpoints
**Tasks:**
- Create HTTP server with required endpoints:
  - `POST /chat` - Process chat messages
  - `GET /get_current_messages` - Get current conversation context
  - `POST /ingest_memo` - Process memo text
  - `GET /get_current_memories` - List user memories
  - `GET /get_due_timed_reminders` - List due reminders
  - `POST /create_reminder` - Create new reminder
- Add request/response models matching Python API
- Implement proper error handling and validation

**Tests:**
- API endpoint integration tests
- Request/response serialization tests
- Error handling for malformed requests

### Stage 6: CLI Interface
**Tasks:**
- Build ratatui-based terminal interface
- Implement chat display with message history
- Add command handling (clap integration)
- Create interactive memory/reminder browsing

**Tests:**
- CLI command parsing tests
- UI component rendering tests (where possible)
- Integration tests for CLI workflows

### Stage 7: Advanced Features
**Tasks:**
- Implement synthetic tool calls (adding tool messages to chat history)
- Add chat history persistence and management
- Create proper async task handling
- Optimize database queries and caching

**Tests:**
- Synthetic tool call integration tests
- Chat history persistence tests
- Performance tests for database operations

## Key Technical Decisions

### Synthetic Tool Calls
Unlike the Python version that manipulates context via system calls, elroy-rs will add proper tool messages to chat history following OpenAI API specs. This means:
- When a tool is called, both the assistant message with tool_calls and the subsequent tool message with results are added to context
- This provides better compatibility with LLM API specifications
- Chat history accurately reflects the full conversation including tool interactions

### Database Design
- Use PostgreSQL with Diesel ORM
- Avoid pgvector for V0 (as specified in requirements)
- Implement semantic similarity in-memory or via external service
- Core tables: users, memories, reminders, context_messages, function_calls

### Async Architecture
- Leverage Tokio for async I/O, especially for LLM API calls and database operations
- Use connection pooling for database access
- Implement proper error handling with context propagation

### Error Handling
- Use `anyhow` for application errors and `thiserror` for custom error types
- Implement recoverable tool errors that don't crash the conversation
- Proper logging throughout the application

## Required Dependencies

```toml
[dependencies]
# CLI
clap = { version = "4.0", features = ["derive"] }
ratatui = "0.26"
crossterm = "0.27"

# Database
diesel = { version = "2.0", features = ["postgres", "uuid", "chrono", "serde_json"] }
diesel_migrations = "2.0"

# LLM APIs
anthropic-sdk-rust = "0.1"
async-openai = "0.17"

# Async runtime
tokio = { version = "1.0", features = ["full"] }
tokio-stream = "0.1"

# HTTP server
axum = "0.7"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Utils
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

## Project Structure

```
src/
├── main.rs                 # CLI entry point
├── lib.rs                  # Library root
├── api/                    # HTTP API endpoints
│   ├── mod.rs
│   ├── chat.rs
│   ├── memo.rs
│   └── models.rs          # Request/response types
├── cli/                    # Terminal UI
│   ├── mod.rs
│   ├── app.rs            # Main TUI application
│   └── components/       # UI components
├── core/                   # Core business logic
│   ├── mod.rs
│   ├── context.rs        # ElroyContext equivalent
│   ├── session.rs        # Session management
│   └── config.rs         # Configuration
├── db/                     # Database layer
│   ├── mod.rs
│   ├── models.rs         # Diesel models
│   ├── schema.rs         # Generated schema
│   └── migrations/       # SQL migrations
├── llm/                    # LLM clients
│   ├── mod.rs
│   ├── client.rs         # Unified LLM trait
│   ├── anthropic.rs      # Anthropic client
│   ├── openai.rs         # OpenAI client
│   └── streaming.rs      # Stream parsing
├── memory/                 # Memory management
│   ├── mod.rs
│   ├── operations.rs     # Memory CRUD
│   └── search.rs         # Semantic search
├── reminder/               # Reminder management
│   ├── mod.rs
│   └── operations.rs     # Reminder CRUD
└── tools/                  # Tool system
    ├── mod.rs
    ├── registry.rs       # Tool registration
    ├── memo.rs          # ingest_memo tool
    └── synthetic.rs      # Synthetic tool calls
```

## Success Criteria for V0

✅ **Basic chat functionality** - Users can send messages and receive LLM responses
✅ **Tool support** - Assistant can call `ingest_memo` tool during conversations
✅ **API endpoints** - All required endpoints work correctly matching Python API
✅ **Memory/Reminder management** - Users can create, retrieve, and manage memories and reminders
✅ **Database persistence** - All data properly stored and retrieved from PostgreSQL
✅ **CLI interface** - Functional terminal UI for basic interactions
✅ **Chat history** - Maintain conversation context across interactions
✅ **Synthetic tool calls** - Proper tool message handling in chat history

## Excluded from V0 (as specified)
- ❌ Async context refreshing
- ❌ Memory consolidation
- ❌ Advanced memory management features
- ❌ pgvector integration

## Testing Strategy

### Unit Tests
- Database model tests
- LLM client tests (mocked)
- Tool execution tests
- Memory/reminder operation tests

### Integration Tests
- API endpoint tests
- End-to-end chat flow tests
- Database integration tests
- CLI workflow tests

### Test Data
- Sample chat conversations
- Test memories and reminders
- Mock LLM responses for consistent testing

This plan provides a comprehensive roadmap for implementing elroy-rs V0 while maintaining compatibility with the existing elroy Python API and focusing on the core requirements.
