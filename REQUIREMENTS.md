# Elroy RS

A rust port of the elroy repo.

The elroy repo can be found at /Users/tombedor/Development/elroy

# Dependencies

`anthropic-sdk-rust`: lib for interacting with Anthropic API's

`async-openai`: for interacting with opeani API

`diesel`: for managing postgres. Do NOT use pg-vector, we will not use postgres for handling vector similarity comparisons

`clap`: for CLI setup

`ratatui`: rich-text like terminal UI

# API

Interactions with the backend happen via API. Operations include:

- chat: processes a chat message to a llm completion API

- inget_memo: process user text, create a memory or reminder

# LLM Tools

The AI chat loop should support tool calls:
- reminder management (as per elroy)
- memory management (as per elroy capabilities)

# Features:
- maintain chat history
- add "synthetic" tool calls: this should add a tool message to the chat history, along with the preceding assistant message that's required as per the openai api spec. This is in contrast to Elroy's implementation, which manipulates data in the context window via additional system calls.

# V0:
- Basic chat should be supported
- ingest_memo tool should be supported
- chat and memo endpoints should be supported
- listing of memories and reminders should be supported

Do NOT implement:
- async context refreshing
- memory consolidation
