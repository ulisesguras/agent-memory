# Agent Memory
### The Cognitive Memory Layer for Autonomous AI Agents

> *"Not a service you call. An entity that grows."*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)
[![Zero Dependencies](https://img.shields.io/badge/Core-Zero%20Dependencies-orange.svg)]()

---

## The Problem

AI agents are powerful but disposable. Every session starts from zero — no memory of who you are, what you need, or what worked before. They process, respond, and forget.

The problem isn't storage. It's architecture. Existing systems treat memory as a flat log file that grows until it's too expensive to process, then gets truncated arbitrarily. No judgment. No learning. No identity.

**Agent Memory solves this by giving agents what humans have: a cognitive architecture.**

---

## What It Does

Five memory layers modeled after human cognition. Autonomous importance scoring — no configuration, no labels. An identity layer that makes agents persistent entities, not stateless services.

```
INCOMING INPUT
      ↓
┌─────────────┐    ┌─────────────┐    ┌───────────────┐
│   SENSORY   │───▶│ SHORT-TERM  │───▶│   LONG-TERM   │
│  (1 slot)   │    │  (20 slots) │    │  (500 slots)  │
└─────────────┘    └─────────────┘    └───────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌───────────────┐
                   │  EPISODIC   │    │   SEMANTIC    │
                   │ (200 slots) │    │ (1000 slots)  │
                   └─────────────┘    └───────────────┘
                          │
                          ▼
                   Consolidation
                (episodic → semantic)
```

| Layer | Purpose | Retention |
|-------|---------|-----------|
| Sensory | Current turn | 1 slot, replaced each turn |
| Short-term | Active session | 20 slots, pruned by importance |
| Long-term | Critical facts | 500 slots, HIGH+ importance only |
| Episodic | Past experiences | 200 slots, consolidates over time |
| Semantic | Learned concepts | 1000 slots, grows from episodic |

---

## Quick Start

```bash
git clone https://github.com/ulisesguras/agent-memory
cd agent-memory

# One command setup
chmod +x setup.sh && ./setup.sh

# Run experiments — no LLM, no API key needed
python cli/experiment.py --all

# Interactive agent (mock mode)
python cli/run_agent.py --agent my_agent --llm mock

# Interactive agent (Claude)
export ANTHROPIC_API_KEY=your_key
python cli/run_agent.py --agent my_agent --llm anthropic

# MCP server — plug into Claude Desktop
python mcp_server.py

# HTTP API
uvicorn core.server:app --reload --port 8080
# → http://localhost:8080/docs
# → http://localhost:8080/.well-known/agent.json
# → http://localhost:8080/metrics
```

---

## Agent-Native by Design

Any agent that speaks MCP can discover and use Agent Memory without human intervention.

**Machine-readable capability manifest:**
```
GET /.well-known/agent.json
→ {capabilities, pricing, latency, reliability, frameworks}
```

**Real-time reliability metrics** — what agent scoring functions read:
```
GET /metrics
→ {liveness, reliability: {success_rate_1h, uptime_30d}, confidence, latency}
```

**MCP Server — 10 tools, any framework:**
```bash
python mcp_server.py --transport stdio   # Claude Desktop
python mcp_server.py --transport sse     # Remote agents
```

**Claude Desktop registration:**
```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "python",
      "args": ["/path/to/agent-memory/mcp_server.py"],
      "env": { "LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "xxx" }
    }
  }
}
```

---

## The Identity Layer

Every agent instance is defined by five markdown documents — not configuration files, but the beginning of personhood:

```
identity/
├── SOUL.md       ← Who the agent IS (philosophy, values — immutable)
├── IDENTITY.md   ← How it presents to the world
├── AGENTS.md     ← How it operates (ReAct loop, priorities)
├── MEMORY.md     ← Its cognitive architecture spec
└── USER.md       ← Its growing understanding of you
```

Edit `SOUL.md` to change the agent's values. No Python required.

---

## The ReAct Loop

```
OBSERVE → THINK → PLAN → ACT → REFLECT → STORE
```

Every interaction produces a **ThoughtTrace** — full transparency, no black boxes:

```python
resp = agent.interact("sess_001", "remember this project uses Python 3.11")

print(resp.thought.importance_score)   # "critical"
print(resp.thought.memories_stored)    # ["sensory", "short_term", "long_term"]
print(resp.thought.duration_ms)        # 43.2
```

---

## Pluggable Everything

**LLM Backends:**
```python
from core.agent import Agent, MockAdapter, AnthropicAdapter, VertexAIAdapter

agent = Agent("my_agent", llm=MockAdapter())       # Dev/testing — free
agent = Agent("my_agent", llm=AnthropicAdapter())  # Claude
agent = Agent("my_agent", llm=VertexAIAdapter())   # Gemini

# Custom — implement one method
class MyLLM:
    def generate(self, prompt: str) -> str: ...
```

**Storage Backends:**
```python
from core.memory_engine import LocalStorage

storage = LocalStorage("./data/agents")   # Default — zero deps
# Swap for Firestore, Redis, SQLite
# Implement: load() / save() / delete() / list_agents()
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_interact` | Full ReAct loop — observe, remember, respond |
| `memory_store` | Store a fact (CRITICAL by default) |
| `memory_retrieve` | Retrieve from a cognitive layer |
| `memory_context` | Formatted context for LLM injection |
| `memory_stats` | Memory health: count, pressure, distribution |
| `memory_search` | Search across all 5 layers |
| `memory_consolidate` | Run episodic→semantic dreaming |
| `memory_new_session` | New session, long-term preserved |
| `memory_forget` | Delete memory by ID |
| `memory_list_agents` | List all agents with stored memory |

---

## Experiments

```bash
python cli/experiment.py --list     # See all experiments
python cli/experiment.py --all      # Run everything

# Individual experiments
python cli/experiment.py --run basic_memory    # 5-layer store/retrieve
python cli/experiment.py --run importance      # Auto-classification
python cli/experiment.py --run pruning         # Strategic forgetting
python cli/experiment.py --run consolidation   # Episodic→semantic dreaming
python cli/experiment.py --run multi_session   # Cross-session persistence
python cli/experiment.py --run full_agent      # Complete ReAct loop
```

---

## Performance

| Metric | Value |
|--------|-------|
| Context size reduction | 94% vs flat log |
| Storage efficiency | 70% vs traditional |
| Median latency (p50) | 45ms |
| p99 latency | 200ms |
| Core engine dependencies | 0 |
| MCP tools | 10 |

---

## Project Structure

```
agent-memory/
├── core/
│   ├── memory_engine.py   ← Cognitive engine — 600 lines, zero deps
│   ├── agent.py           ← Autonomous agent + ReAct loop
│   ├── server.py          ← Optional FastAPI layer
│   └── metrics.py         ← Reliability scoring (liveness/confidence)
├── identity/
│   ├── SOUL.md            ← Philosophical core (immutable)
│   ├── IDENTITY.md        ← Presentation layer
│   ├── AGENTS.md          ← Operational instructions
│   ├── MEMORY.md          ← Architecture specification
│   └── USER.md            ← Learned user preferences
├── cli/
│   ├── run_agent.py       ← Interactive CLI
│   └── experiment.py      ← Research experiment runner
├── .well-known/
│   └── agent.json         ← Machine-readable capability manifest
├── mcp_server.py          ← MCP server (stdio + SSE)
├── setup.sh               ← One-command setup
└── requirements.txt
```

---

## CLI Commands

```
/stats              Memory statistics with pressure indicator
/memory [type]      Show memories (short/long/ep/sem/sensory/all)
/search <query>     Search all memory layers
/remember <fact>    Store as CRITICAL — never forgotten
/forget <id>        Delete memory by ID
/consolidate        Run episodic→semantic consolidation
/new                New session (long-term preserved)
/clear              Clear ALL memory
/export             Export to JSON
/verbose            Toggle full thought trace
/quit               Exit
```

---

## Compatible Frameworks

LangChain · LangGraph · AutoGen · CrewAI · PydanticAI · Custom ReAct

Compatible LLMs: Anthropic Claude · Google Gemini · OpenAI · Ollama · Any

---

## Design Principles

1. **Zero dependencies in core.** `memory_engine.py` runs with pure Python stdlib.
2. **Entity, not service.** The agent has identity, memory, and learns who you are.
3. **Transparent by default.** Every interaction produces a full ThoughtTrace.
4. **Elastic.** CLI · API · Library. No infrastructure required to start.
5. **Replaceable components.** LLM, storage, classifier — all pluggable adapters.

---

## Roadmap

- [x] Five-layer cognitive engine
- [x] Autonomous importance scoring (bilingual ES/EN)
- [x] Episodic → semantic consolidation
- [x] Identity layer (SOUL/IDENTITY/AGENTS/MEMORY/USER)
- [x] MCP server — 10 tools, stdio + SSE
- [x] Machine-readable manifest (`agent.json`)
- [x] Real-time reliability metrics
- [x] Cross-session persistence
- [ ] Semantic search with embeddings
- [ ] USER.md auto-update from interactions
- [ ] Managed hosted API
- [ ] Multi-agent shared memory
- [ ] MCP registry listings (mcp.so, smithery.ai)

---

## License

MIT — use it, embed it, build on it.

---

*Built by [@ulisesguras](https://github.com/ulisesguras)*
*Architecture: service → autonomous entity*
*Core insight: memory is not storage. Memory is identity.*
