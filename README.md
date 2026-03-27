# CodePilot - AI-Powered Autonomous Coding Agent

**UNCC ITCS 6156 — Generative AI, Spring 2025 | Group Project 2**

CodePilot is an autonomous CLI coding agent built with **LangGraph**, **LangChain**, and **MCP (Model Context Protocol)**. It accepts natural language coding instructions, generates execution plans, and implements them by calling filesystem tools — all with built-in quality gates, interactive feedback loops, and multi-provider LLM support.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [LangGraph Workflow](#langgraph-workflow)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Provider Abstraction](#provider-abstraction)
- [MCP Integration](#mcp-integration)
- [RAG Pipeline](#rag-pipeline)
- [Execution Modes](#execution-modes)
- [Key Design Decisions](#key-design-decisions)
- [Team](#team)

---

## Features

- **Natural Language to Code** — Describe what you want in plain English; CodePilot writes, modifies, and organizes files automatically.
- **Intelligent Routing** — LLM classifies tasks as *simple* (direct execution) or *complex* (plan-first with approval gates).
- **Multi-Step Planning** — Complex tasks go through plan generation, LLM-based quality scoring, user clarification Q&A, and explicit approval before execution.
- **LLM-as-Judge** — Plans are scored (0.0–1.0) by a strict LLM judge that catches ambiguity, wrong assumptions, missing steps, and incorrect ordering.
- **Implementation Verification** — After execution, a code judge verifies the implementation matches the plan; incorrect results trigger automatic retries.
- **MCP Tool Calling** — All file operations (read, write, edit, search) go through the Model Context Protocol for standardized, auditable tool use.
- **RAG-Enhanced Context** — A custom Python documentation retrieval server (BM25 + vector fusion) provides relevant API references during planning.
- **Multi-Provider LLM Support** — Seamlessly routes between OpenAI, Groq, and Ollama based on task complexity and API key availability.
- **Confirm / Auto Modes** — Choose whether to approve each tool call individually or let the agent run autonomously.
- **Session Persistence** — Checkpointing via LangGraph allows pause/resume across sessions.
- **Rich CLI** — Interactive terminal interface with syntax highlighting, progress tracking, and status panels.

---

## Architecture Overview

```
                         +------------------+
                         |    User (CLI)    |
                         +--------+---------+
                                  |
                                  v
                    +-------------+-------------+
                    |   Query Reconstruction    |
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |     Context Updator       |  <-- Scans project via MCP
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |      Super Router         |  <-- LLM classifies: simple/complex
                    +------+------------+-------+
                           |            |
                    simple |            | complex
                           |            |
                           v            v
                    +------+--+  +------+--------+
                    |Implement|  |   Plan Node    | <-- RAG + LLM generation + LLM judge
                    +------+--+  +------+---------+
                           |            |
                           |     score < threshold?
                           |       /        \
                           |      v          v
                           | Clarification  User Approval
                           |   (Q&A)        (approve/reject)
                           |      \          /
                           |       v        v
                           |     +----------+
                           |     | Re-plan  |
                           |     +----------+
                           |            |
                           +<-----------+ (approved)
                           |
                    +------v------+
                    |  Code Judge  |  <-- LLM verifies implementation
                    +------+------+
                           |
                    correct? -----> END
                    incorrect? ---> Retry (up to 5x)
```

---

## LangGraph Workflow

The agent is built as a **LangGraph state machine** with 9 nodes and 4 conditional edges:

### Nodes

| Node | Description |
|------|-------------|
| `query_reconstruction` | Rewrites raw user input into a clear, actionable instruction |
| `comparator` | Compares previous and current codebase snapshots to detect changes |
| `context_updator` | Scans the target project directory (file tree, tech stack, dependencies) |
| `super_router` | LLM classifies task complexity: `simple` or `complex` |
| `plan_node` | Generates step-by-step execution plan, queries RAG, scores plan with LLM judge |
| `user_clarification` | Generates targeted Q&A when plan score is below threshold |
| `user_plan_approval` | Displays plan to user for explicit approval or rejection |
| `implement` | Runs LLM tool-calling loop over MCP tools (max 25 steps per iteration) |
| `code_judge` | LLM verifies implementation correctness; fallback to rule-based check |

### Conditional Edges

| Edge | Condition | Routes |
|------|-----------|--------|
| `has_prev_context` | Previous context exists? | `comparator` or `context_updator` |
| `route_type` | Simple or complex task? | `implement` or `plan_node` |
| `plan_score` | Plan score >= 0.85 or iterations >= 4? | `user_plan_approval` or `user_clarification` |
| `implementation_correct` | Judge says correct or iterations >= 5? | `END` or `implement` (retry) |

### Loop Safety

- **Plan loop**: Max 4 iterations before forcing user approval
- **Implement loop**: Max 5 iterations before terminating
- **Tool-calling loop**: Max 25 LLM steps per implement invocation

---

## Project Structure

```
CodePilot_UNCC/
├── main.py                          # Entry point
├── config.yaml                      # Agent configuration
├── .env                             # API keys (OPENAI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY)
├── requirements.txt                 # Python dependencies
├── Group Project 2 - Design.pdf     # Design documentation
│
├── core/                            # Core infrastructure
│   ├── state.py                     # AgentState TypedDict — shared state schema
│   ├── graph.py                     # LangGraph definition (nodes, edges, conditionals)
│   ├── config.py                    # YAML config loader with path resolution
│   ├── checkpoint.py                # Persistent checkpointing (pickle-based)
│   └── async_utils.py              # Safe async-to-sync bridge
│
├── nodes/                           # LangGraph node implementations
│   ├── query_reconstruction.py      # Restructures user query
│   ├── super_router.py              # Task complexity classifier
│   ├── comparator.py                # Codebase diff detection
│   ├── context_updator.py           # Project scanner (MCP + local fallback)
│   ├── plan_node.py                 # Plan generation + RAG + scoring
│   ├── user_clarification.py        # Interactive Q&A for ambiguities
│   ├── user_plan_approval.py        # Plan display + approval collection
│   ├── implement.py                 # MCP tool-calling executor
│   └── code_judge.py                # Implementation verification
│
├── tools/                           # Utility modules used by nodes
│   ├── plan_generator.py            # LLM-based plan generation
│   ├── plan_verifier.py             # LLM-based plan quality scoring
│   ├── clarification_generator.py   # Generates clarification questions
│   ├── suggestion_generator.py      # Generates multiple-choice options
│   ├── codebase_learner.py          # Project directory scanner
│   ├── tool_calling_node.py         # LLM tool-calling loop orchestrator
│   └── mcp_tooling.py              # MCP tool wrappers for LangChain
│
├── providers/                       # LLM provider abstraction
│   └── provider.py                  # Routes to Ollama / Groq / OpenAI
│
├── mcp_client/                      # MCP client
│   └── client.py                    # Connects to MCP servers, routes tool calls
│
├── mcp_servers/                     # Custom MCP servers
│   └── rag_server/                  # Python docs RAG server
│       ├── server.py                # FastMCP entry point (query_python_docs tool)
│       ├── indexer.py               # Document chunking + ChromaDB + BM25 indexing
│       ├── retriever.py             # Fusion retrieval (BM25 + vector)
│       └── fusion.py                # Reciprocal Rank Fusion algorithm
│
├── prompts/                         # Prompt templates
│   ├── prompt_renderer.py           # POML template renderer
│   └── *.poml                       # POML templates (plan, judge, router, etc.)
│
├── ui/                              # User interface
│   └── cli.py                       # Rich-based interactive CLI
│
└── data/                            # RAG data store
    ├── docs/                        # Python documentation files
    └── bm25/                        # Persisted BM25 index artifacts
```

---

## Setup & Installation

### Prerequisites

- **Python 3.12+**
- **Node.js & npm** (for MCP filesystem and Tavily servers)
- **Ollama** (local LLM — install from [ollama.com](https://ollama.com))

### Step 1: Clone and navigate

```bash
cd CodePilot_UNCC
```

### Step 2: Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install MCP servers (npm)

```bash
npm install -g @modelcontextprotocol/server-filesystem
npm install -g tavily-mcp
```

### Step 5: Pull Ollama model

```bash
ollama pull llama3.1:8b
```

### Step 6: Set up environment variables

Create a `.env` file in `CodePilot_UNCC/`:

```env
# Required for web search fallback in RAG
TAVILY_API_KEY=your_tavily_key

# Optional — cloud LLM providers (at least one recommended for complex tasks)
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

### Step 7: Create project directory

```bash
mkdir -p ../project
```

### Step 8: Run

```bash
python main.py
```

Or specify a custom project directory:

```bash
python main.py /path/to/your/project
```

---

## Configuration

`config.yaml` controls agent behavior:

```yaml
provider:
  name: openai              # ollama | groq | openai
  model: gpt-4o-mini        # model to use for the configured provider

project_path: ../project    # target project directory (relative to CodePilot_UNCC/)

execution_mode: auto        # confirm | auto

checkpointing:
  enabled: true
  path: .codepilot/checkpoints.pkl

mcp_servers:
  filesystem:
    command: npx
    args: ["@modelcontextprotocol/server-filesystem", "."]
  tavily:
    command: npx
    args: ["tavily-mcp"]
  rag:
    command: ./venv/bin/python
    args: ["mcp_servers/rag_server/server.py"]
```

---

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/clear` | Clear the screen |
| `/mode confirm` | Switch to confirm mode (approve each tool call) |
| `/mode auto` | Switch to auto mode (tools execute automatically) |
| `/exit` | Quit the agent |

### Example: Simple Task

```
> create a python script that sorts a list of numbers using merge sort
```

Flow: `query_reconstruction` -> `context_updator` -> `super_router (simple)` -> `implement` -> `code_judge` -> `END`

### Example: Complex Task

```
> Build a Flask REST API for a task manager with CRUD endpoints, input validation, and error handling
```

Flow: `query_reconstruction` -> `context_updator` -> `super_router (complex)` -> `plan_node` -> (clarification Q&A if needed) -> `user_plan_approval` -> `implement` -> `code_judge` -> `END`

---

## Provider Abstraction

CodePilot supports three LLM providers with automatic routing:

```
Priority: OpenAI > Groq > Ollama (local fallback)
```

| Task Type | Provider Used |
|-----------|--------------|
| Pre-routing (query reconstruction, routing) | Ollama (fast, local) |
| Simple tasks (full pipeline) | Ollama |
| Complex tasks (planning, implementation, judging) | Best available cloud (OpenAI > Groq) |
| Rate limit hit | Automatic fallback to Ollama |
| No API keys set | Ollama for everything |

**Default models:**
- Ollama: `llama3.1:8b`
- Groq: `llama-3.3-70b-versatile`
- OpenAI: `gpt-4o-mini`

---

## MCP Integration

CodePilot uses MCP for all tool interactions:

### Filesystem Server
- `read_file`, `write_file`, `edit_file`, `create_directory`, `list_directory`, `move_file`
- Sandboxed to the project directory for safety

### Tavily Server
- `tavily_search` — Web search fallback when RAG doesn't have relevant documentation

### RAG Server (Custom)
- `query_python_docs` — Retrieves relevant Python documentation chunks
- Uses fusion retrieval: BM25 (keyword) + ChromaDB (vector embeddings)
- Reciprocal Rank Fusion for result merging

### Tool Calling Architecture

The LLM uses a single generic tool `mcp_call(tool_name, arguments)` which routes to the appropriate MCP server:

```
LLM -> mcp_call("write_file", {"path": "/project/app.py", "content": "..."})
    -> MCPClient -> filesystem MCP server -> writes file
    -> returns result to LLM
```

---

## RAG Pipeline

The RAG server provides context-aware documentation retrieval:

1. **Indexing** (`indexer.py`): Documents are chunked semantically, embedded with `sentence-transformers`, and stored in ChromaDB. BM25 artifacts are pickled for fast keyword search.

2. **Retrieval** (`retriever.py`): Queries hit both BM25 (keyword) and ChromaDB (vector) indexes in parallel.

3. **Fusion** (`fusion.py`): Results are merged using Reciprocal Rank Fusion (RRF) to combine keyword relevance with semantic similarity.

4. **Fallback**: If RAG returns no results, `plan_node` falls back to Tavily web search.

---

## Execution Modes

### Auto Mode (`execution_mode: auto`)
- All tool calls execute immediately without user intervention
- Best for trusted, well-planned tasks

### Confirm Mode (`execution_mode: confirm`)
- Each tool call pauses and prompts the user:
  ```
  Tool call: write_file({"path": "/project/app.py", ...})
  Approve? (y/n)
  ```
- `y` / Enter — approve and execute
- `n` — reject (tool is skipped, LLM adapts)
- Switch at runtime: `/mode confirm` or `/mode auto`

---

## Key Design Decisions

1. **LangGraph over raw LangChain** — State machine architecture enables complex conditional routing with loop safety, checkpointing, and clear node boundaries.

2. **MCP over direct file I/O** — Standardized tool protocol provides auditability, sandboxing, and extensibility (adding new tool servers requires zero code changes to the core agent).

3. **LLM-as-Judge for plan quality** — Catches ambiguity and wrong assumptions before execution, preventing wasted compute on bad plans.

4. **Dual retrieval (BM25 + vector)** — Keyword search catches exact API names; vector search catches semantic intent. Fusion gives best of both.

5. **Multi-provider with automatic fallback** — Graceful degradation: cloud rate limit -> Ollama fallback. No API keys -> full local operation.

6. **Inline prompts over POML templates** — Dynamic content (execution logs, code snippets) containing `<`, `>`, `/` characters broke POML's XML parser. Critical nodes use inline f-string prompts for reliability.

7. **Queue-based confirm mode** — Background async thread posts confirmation requests via `queue.Queue`; main CLI thread responds. Avoids `prompt_toolkit` / `asyncio` conflicts across threads.

---

