# MEMORY.md — My Living Memory State

> This file is a template. In production, it is generated and updated autonomously.
> Each agent instance has its own MEMORY.md derived from this structure.

---

## Memory Architecture

Inspired by human cognitive neuroscience. Five layers. Each with different retention, decay, and retrieval behavior.

```
┌─────────────────────────────────────────────────────────┐
│                    INCOMING INPUT                        │
│                         ↓                               │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ SENSORY  │───▶│ SHORT-TERM  │───▶│   LONG-TERM   │  │
│  │ (seconds)│    │  (session)  │    │  (permanent)  │  │
│  └──────────┘    └─────────────┘    └───────────────┘  │
│                         │                   │           │
│                         ▼                   ▼           │
│                  ┌─────────────┐    ┌───────────────┐  │
│                  │  EPISODIC   │    │   SEMANTIC    │  │
│                  │(experiences)│    │  (concepts)   │  │
│                  └─────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Memory Types

### 1. Sensory Memory
```yaml
type: sensory
purpose: "Immediate working context — what's happening right now"
retention: "Current turn only"
max_items: 1
decay: "Cleared on each new turn"
examples:
  - "Current user message"
  - "Last tool call result"
  - "Active task state"
```

### 2. Short-Term Memory
```yaml
type: short_term
purpose: "Recent interaction context — the current session"
retention: "Duration of session"
max_items: 20
decay: "Pruned when limit reached, by importance score"
examples:
  - "Last 5 exchanges in this conversation"
  - "Current task being worked on"
  - "Decisions made this session"
```

### 3. Long-Term Memory
```yaml
type: long_term
purpose: "Important facts that must persist across sessions"
retention: "Indefinite (unless explicitly deleted)"
max_items: 500
decay: "Only via explicit pruning of low-importance items"
importance_threshold: "HIGH or CRITICAL only"
examples:
  - "User's name is Alex"
  - "This project uses Python 3.11"
  - "User prefers concise responses"
  - "API key pattern: stored securely"
```

### 4. Episodic Memory
```yaml
type: episodic
purpose: "Specific past experiences and events"
retention: "Long-term but subject to consolidation"
max_items: 200
decay: "Consolidates into semantic memory over time"
examples:
  - "On 2026-01-15, debugged authentication error in main.py"
  - "User was frustrated with verbose responses last Tuesday"
  - "Successfully deployed to Cloud Run after 3 attempts"
```

### 5. Semantic Memory
```yaml
type: semantic
purpose: "Learned concepts, patterns, and generalized knowledge"
retention: "Indefinite"
max_items: 1000
decay: "Updated/overwritten as understanding improves"
examples:
  - "User typically works in Python"
  - "This codebase follows FastAPI patterns"
  - "User's timezone is UTC-3"
  - "Deployment usually fails due to missing env vars"
```

---

## Importance Scoring

```
CRITICAL → Never forget. User explicitly marked or system-classified as essential.
HIGH     → Important fact. Keep until explicitly pruned.
MEDIUM   → Useful context. May be pruned if memory pressure high.
LOW      → Ephemeral detail. Pruned first when space needed.
```

### Auto-classification triggers:
- Keywords: "remember", "important", "never forget", "critical" → CRITICAL
- Keywords: "key", "main", "always" → HIGH  
- Interaction length > 200 chars → MEDIUM
- Default → LOW
- Correction of previous error → HIGH (agent learning)

---

## Memory Operations

```python
# Conceptual API — see core/memory_engine.py for implementation

store(type, content, importance)      # Write to layer
retrieve(type, query, limit)          # Read from layer  
prune(session_id, threshold)          # Forget low-importance
consolidate(episodic → semantic)      # Dream-like synthesis
search_semantic(query, top_k)         # Vector similarity search
```

---

## Current State (Live)
```yaml
# This section is auto-populated at runtime
agent_id: ""
session_id: ""
total_memories: 0
last_updated: ""
memory_pressure: 0.0   # 0.0 = empty, 1.0 = at capacity
consolidation_pending: false
```

---

*This is a living document. Updated autonomously by the agent during operation.*
