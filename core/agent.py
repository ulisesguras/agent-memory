"""
core/agent.py

The Agent — autonomous, memory-driven, self-directing.

This is not a wrapper around an LLM. This is an entity that:
  1. Reads its identity from SOUL/IDENTITY/AGENTS/USER markdown files
  2. Uses MemoryEngine to recall context before every response
  3. Follows the OBSERVE → THINK → PLAN → ACT → REFLECT → STORE loop
  4. Updates USER.md as it learns preferences
  5. Consolidates episodic → semantic memory periodically

Designed for experiments, prototyping, and elastic memory research.
Not hardcoded. Not a SaaS product. A cognitive agent you can run locally.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from core.memory_engine import (
    MemoryEngine, MemoryType, Importance, Memory, LocalStorage, ImportanceClassifier
)


# ─────────────────────────────────────────
# THOUGHT TRACE (Transparent Reasoning)
# ─────────────────────────────────────────

@dataclass
class ThoughtTrace:
    """
    The agent's internal reasoning log for one interaction.
    Transparent by default — you can see what the agent was thinking.
    """
    timestamp:        str   = field(default_factory=lambda: datetime.now().isoformat())
    observation:      str   = ""
    memory_loaded:    dict  = field(default_factory=dict)
    plan:             str   = ""
    action_taken:     str   = ""
    importance_score: str   = ""
    memories_stored:  list  = field(default_factory=list)
    consolidation:    list  = field(default_factory=list)
    duration_ms:      float = 0

    def render(self) -> str:
        ts = datetime.fromisoformat(self.timestamp).strftime("%H:%M:%S")
        lines = [
            f"[{ts}] ── AGENT THOUGHT TRACE ──────────────────────",
            f"  OBSERVE  : {self.observation}",
        ]
        if self.memory_loaded:
            counts = {k: len(v) for k, v in self.memory_loaded.items() if v}
            lines.append(f"  MEMORY   : {counts}")
        lines.append(f"  PLAN     : {self.plan}")
        lines.append(f"  ACT      : {self.action_taken}")
        lines.append(f"  REFLECT  : importance={self.importance_score}")
        if self.memories_stored:
            lines.append(f"  STORED   : {len(self.memories_stored)} memory entries")
        if self.consolidation:
            lines.append(f"  DREAM    : {len(self.consolidation)} episodic→semantic")
        lines.append(f"  DURATION : {self.duration_ms:.0f}ms")
        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)


# ─────────────────────────────────────────
# AGENT RESPONSE
# ─────────────────────────────────────────

@dataclass
class AgentResponse:
    """Complete output from a single agent interaction."""
    agent_id:    str
    session_id:  str
    text:        str
    thought:     ThoughtTrace
    memory_stats: dict

    def to_dict(self) -> dict:
        return {
            "agent_id":    self.agent_id,
            "session_id":  self.session_id,
            "text":        self.text,
            "thought":     self.thought.__dict__,
            "memory_stats": self.memory_stats,
        }


# ─────────────────────────────────────────
# LLM ADAPTER (pluggable)
# ─────────────────────────────────────────

class LLMAdapter:
    """
    Abstract LLM interface. Swap backends without changing the agent.
    Implement `generate(prompt) -> str` for your provider.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def get_name(self) -> str:
        return "base"


class VertexAIAdapter(LLMAdapter):
    """Google Vertex AI / Gemini backend."""
    def __init__(self, model: str = "gemini-2.0-flash-exp", project: str = None, location: str = "us-central1"):
        import vertexai
        from vertexai.generative_models import GenerativeModel
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "dynamicstorage")
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model)
        self._name = model

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    def get_name(self) -> str:
        return self._name


class AnthropicAdapter(LLMAdapter):
    """Anthropic Claude backend."""
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def generate(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def get_name(self) -> str:
        return self._model


class MockAdapter(LLMAdapter):
    """
    No-LLM adapter for testing and development.
    Returns a structured mock response that echoes back context.
    """
    def generate(self, prompt: str) -> str:
        # Extract user input from prompt for the mock
        if "USER INPUT:" in prompt:
            user_part = prompt.split("USER INPUT:")[-1].split("INSTRUCTIONS:")[0].strip()
        else:
            user_part = prompt[-200:]
        return (
            f"[MOCK AGENT RESPONSE]\n"
            f"I received: \"{user_part[:100]}...\"\n"
            f"Memory context was loaded. I would now respond based on it.\n"
            f"(Replace MockAdapter with VertexAIAdapter or AnthropicAdapter for real responses.)"
        )

    def get_name(self) -> str:
        return "mock"


# ─────────────────────────────────────────
# IDENTITY LOADER
# ─────────────────────────────────────────

def load_identity(identity_dir: str = "./identity") -> dict:
    """
    Load the agent's identity documents (SOUL, IDENTITY, AGENTS, USER).
    Returns a dict with their contents for prompt construction.
    """
    docs = {}
    path = Path(identity_dir)
    for fname in ["SOUL.md", "IDENTITY.md", "AGENTS.md", "USER.md"]:
        fpath = path / fname
        if fpath.exists():
            docs[fname.replace(".md", "").lower()] = fpath.read_text()
        else:
            docs[fname.replace(".md", "").lower()] = ""
    return docs


# ─────────────────────────────────────────
# THE AGENT
# ─────────────────────────────────────────

class Agent:
    """
    An autonomous, memory-driven AI agent.

    Not a chatbot wrapper. An entity that:
    - Remembers across sessions
    - Learns who you are
    - Acts on the ReAct loop
    - Manages its own memory autonomously

    Usage:
        agent = Agent(agent_id="my_agent", llm=VertexAIAdapter())
        response = agent.interact("session_001", "What did we discuss last time?")
        print(response.text)
        print(response.thought.render())
    """

    CONSOLIDATION_INTERVAL = 10  # Consolidate every N interactions

    def __init__(
        self,
        agent_id:      str,
        agent_type:    str = "general",
        llm:           Optional[LLMAdapter] = None,
        identity_dir:  str = "./identity",
        data_dir:      str = "./data/agents",
    ):
        self.agent_id    = agent_id
        self.agent_type  = agent_type
        self.llm         = llm or MockAdapter()
        self.identity    = load_identity(identity_dir)

        self.memory = MemoryEngine(
            agent_id  = agent_id,
            storage   = LocalStorage(data_dir),
            classifier= ImportanceClassifier(),
        )

        self._interaction_count = 0

    # ── Core Interaction Loop ─────────────────────────────

    def interact(self, session_id: str, user_input: str, explicit_importance: Optional[Importance] = None) -> AgentResponse:
        """
        The main interaction entry point.
        Executes: OBSERVE → THINK → PLAN → ACT → REFLECT → STORE
        """
        t0 = time.time()
        trace = ThoughtTrace()

        # ── 1. OBSERVE ──────────────────────────────────────
        trace.observation = f"New input from session '{session_id}': {user_input[:80]}..."

        # Store as sensory memory (replaces previous sensory)
        self.memory.store(MemoryType.SENSORY, user_input, importance=Importance.LOW)

        # ── 2. THINK — Load memory context ──────────────────
        ctx = self.memory.retrieve_context(limit_per_layer=3)
        trace.memory_loaded = {k: [m.content[:60] for m in v] for k, v in ctx.items()}

        # ── 3. PLAN — Build prompt ───────────────────────────
        prompt = self._build_prompt(user_input, session_id)
        trace.plan = f"Build contextual prompt for agent_type='{self.agent_type}', using {sum(len(v) for v in ctx.values())} memory items"

        # ── 4. ACT — Generate response ───────────────────────
        ai_response = self.llm.generate(prompt)
        trace.action_taken = f"Generated response ({len(ai_response)} chars) via {self.llm.get_name()}"

        # ── 5. REFLECT — Score importance ────────────────────
        importance = self.memory.classifier.classify(
            f"{user_input} {ai_response}",
            explicit_importance
        )
        trace.importance_score = importance.value

        # ── 6. STORE — Commit to memory ──────────────────────
        stored = []

        # Always store in short-term
        stm = self.memory.store(
            MemoryType.SHORT_TERM,
            f"User: {user_input}\nAgent: {ai_response}",
            importance=importance,
            context={"session_id": session_id},
        )
        stored.append(stm)

        # Promote to episodic if substantive
        if importance in (Importance.HIGH, Importance.CRITICAL):
            ep = self.memory.store(
                MemoryType.EPISODIC,
                f"[{datetime.now().strftime('%Y-%m-%d')}] {user_input[:150]}",
                importance=importance,
                tags=["interaction"],
                context={"session_id": session_id},
            )
            stored.append(ep)

        # Promote to long-term if critical
        if importance == Importance.CRITICAL:
            lt = self.memory.store(
                MemoryType.LONG_TERM,
                f"KEY FACT: {ai_response[:200]}",
                importance=Importance.CRITICAL,
                tags=["promoted"],
            )
            stored.append(lt)

        trace.memories_stored = [m.id for m in stored]

        # ── Periodic consolidation ─────────────────────────
        self._interaction_count += 1
        if self._interaction_count % self.CONSOLIDATION_INTERVAL == 0:
            new_semantic = self.memory.consolidate()
            trace.consolidation = [m.id for m in new_semantic]

        trace.duration_ms = (time.time() - t0) * 1000

        return AgentResponse(
            agent_id     = self.agent_id,
            session_id   = session_id,
            text         = ai_response,
            thought      = trace,
            memory_stats = self.memory.stats(),
        )

    # ── Prompt Construction ───────────────────────────────

    def _build_prompt(self, user_input: str, session_id: str) -> str:
        """
        Builds the full prompt with identity + memory context.
        The identity layer is what separates this from a generic LLM call.
        """
        soul_excerpt = self._extract_soul_essence()
        memory_context = self.memory.format_context_for_prompt(limit_per_layer=3)
        agent_role = self._get_role_description()

        return f"""{agent_role}

━━━ YOUR IDENTITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{soul_excerpt}

━━━ YOUR MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{memory_context}

━━━ OPERATING PRINCIPLES ━━━━━━━━━━━━━━━━━━━━━━━
- Use memory context before asking clarifying questions
- Lead with the answer, then explain
- Be direct. No padding or performative enthusiasm
- If you're uncertain, say so clearly
- Match the language and formality of the user
- Session: {session_id}

━━━ USER INPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_input}

━━━ RESPONSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def _get_role_description(self) -> str:
        roles = {
            "general":    "You are an adaptive AI agent with dynamic memory and autonomous reasoning.",
            "analyst":    "You are a data analyst agent. You find patterns, validate assumptions, and produce structured insights.",
            "researcher": "You are a research agent. You synthesize information from multiple sources into coherent knowledge.",
            "assistant":  "You are a contextual assistant. You remember what matters and act on it.",
            "specialist": "You are a domain specialist. Deep knowledge, precise answers, no generalities.",
        }
        return roles.get(self.agent_type, roles["general"])

    def _extract_soul_essence(self) -> str:
        """Extract key principles from SOUL.md for prompt injection."""
        soul = self.identity.get("soul", "")
        if not soul:
            return "You are an autonomous agent with memory and judgment."
        # Extract just the philosophy section for token efficiency
        lines = soul.split("\n")
        relevant = []
        in_section = False
        for line in lines:
            if "## Philosophy" in line or "## Temperament" in line:
                in_section = True
            elif line.startswith("## ") and in_section:
                in_section = False
            if in_section and line.strip() and not line.startswith("#"):
                relevant.append(line.strip())
        return "\n".join(relevant[:10]) if relevant else "Curious. Persistent. Direct. Honest."

    # ── Memory Management API ─────────────────────────────

    def remember(self, fact: str, memory_type: MemoryType = MemoryType.LONG_TERM) -> Memory:
        """Explicitly store a fact with CRITICAL importance."""
        return self.memory.store(memory_type, fact, importance=Importance.CRITICAL, tags=["explicit"])

    def forget(self, memory_id: str) -> bool:
        """Forget a specific memory by ID."""
        return self.memory.forget(memory_id)

    def search_memory(self, query: str, top_k: int = 5) -> list:
        """Search memory for relevant entries."""
        return self.memory.search(query, top_k=top_k)

    def new_session(self, clear_sensory: bool = True):
        """Start a fresh session while preserving long-term memory."""
        self.memory.clear_session(protect_long_term=True)

    def stats(self) -> dict:
        """Memory statistics."""
        return self.memory.stats()

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, type={self.agent_type}, llm={self.llm.get_name()}, {self.memory})"
