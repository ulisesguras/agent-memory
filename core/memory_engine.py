"""
core/memory_engine.py

The Memory Engine — autonomous memory management for AI agents.
This is not a service layer. This is a cognitive architecture.

Each agent has a MemoryEngine instance that manages all five memory types,
runs the importance classifier, handles pruning, and performs consolidation.
"""

import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


# ─────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────

class MemoryType(str, Enum):
    SENSORY    = "sensory"      # Current turn only — immediate context
    SHORT_TERM = "short_term"   # Session-scoped — recent exchanges
    LONG_TERM  = "long_term"    # Persistent — important facts
    EPISODIC   = "episodic"     # Past experiences — what happened
    SEMANTIC   = "semantic"     # Learned concepts — what was extracted

class Importance(str, Enum):
    CRITICAL = "critical"   # Score: 4 — Never prune
    HIGH     = "high"       # Score: 3 — Prune only under extreme pressure
    MEDIUM   = "medium"     # Score: 2 — Prune when needed
    LOW      = "low"        # Score: 1 — Prune first


IMPORTANCE_SCORE = {
    Importance.CRITICAL: 4,
    Importance.HIGH:     3,
    Importance.MEDIUM:   2,
    Importance.LOW:      1,
}

# Memory layer capacity limits
LAYER_LIMITS = {
    MemoryType.SENSORY:    1,
    MemoryType.SHORT_TERM: 20,
    MemoryType.LONG_TERM:  500,
    MemoryType.EPISODIC:   200,
    MemoryType.SEMANTIC:   1000,
}

# Long-term layer threshold — only HIGH+ goes here automatically
LONG_TERM_THRESHOLD = Importance.HIGH


# ─────────────────────────────────────────
# MEMORY UNIT
# ─────────────────────────────────────────

@dataclass
class Memory:
    """
    The atomic unit of agent memory.
    Not just a dict. A typed, scored, timestamped cognitive entry.
    """
    id:           str
    type:         MemoryType
    content:      str
    importance:   Importance       = Importance.MEDIUM
    access_count: int              = 0
    created_at:   str              = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str]   = None
    tags:         list             = field(default_factory=list)
    context:      dict             = field(default_factory=dict)

    def touch(self):
        """Record that this memory was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    def score(self) -> float:
        """
        Composite retention score.
        Higher = more valuable to keep.
        Used by pruner to decide what to forget.
        """
        importance_weight = IMPORTANCE_SCORE.get(self.importance, 1)
        recency_bonus = 0
        if self.last_accessed:
            hours_since = (datetime.now() - datetime.fromisoformat(self.last_accessed)).total_seconds() / 3600
            recency_bonus = max(0, 1 - (hours_since / 72))  # Decays over 72h
        return (importance_weight * 2) + self.access_count + recency_bonus

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        d["type"]       = MemoryType(d["type"])
        d["importance"] = Importance(d["importance"])
        return cls(**d)


# ─────────────────────────────────────────
# IMPORTANCE CLASSIFIER
# ─────────────────────────────────────────

class ImportanceClassifier:
    """
    Autonomous importance scoring.
    The agent decides what matters — user doesn't have to tag everything.
    
    Extensible: override `classify()` to plug in an LLM-based scorer.
    """

    CRITICAL_SIGNALS = [
        "remember this", "never forget", "critical", "crucial", "vital",
        "always", "must", "required", "do not", "don't ever",
    ]
    HIGH_SIGNALS = [
        "important", "key", "significant", "main", "primary",
        "prefer", "always use", "my name is", "i am", "we are",
    ]
    LOW_SIGNALS = [
        "thanks", "ok", "sure", "got it", "understood", "yes", "no",
        "hello", "hi", "bye",
    ]

    def classify(self, content: str, explicit_override: Optional[Importance] = None) -> Importance:
        """
        Classify importance of a memory content string.
        explicit_override lets the caller force a level (e.g., user said "remember this").
        """
        if explicit_override:
            return explicit_override

        text = content.lower()

        if any(sig in text for sig in self.CRITICAL_SIGNALS):
            return Importance.CRITICAL

        if any(sig in text for sig in self.HIGH_SIGNALS):
            return Importance.HIGH

        if any(sig in text for sig in self.LOW_SIGNALS):
            return Importance.LOW

        # Length heuristic — longer interactions tend to be more substantive
        if len(content) > 500:
            return Importance.HIGH
        if len(content) > 150:
            return Importance.MEDIUM

        return Importance.LOW


# ─────────────────────────────────────────
# STORAGE BACKEND
# ─────────────────────────────────────────

class LocalStorage:
    """
    File-based storage backend.
    Saves each agent's memory as a JSON file.
    Swap this for Firestore, Redis, SQLite, etc. without touching MemoryEngine.
    """

    def __init__(self, data_dir: str = "./data/agents"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return self.data_dir / f"{agent_id}.json"

    def load(self, agent_id: str) -> dict:
        path = self._path(agent_id)
        if not path.exists():
            return {"memories": [], "meta": {}}
        with open(path) as f:
            return json.load(f)

    def save(self, agent_id: str, data: dict):
        with open(self._path(agent_id), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def delete(self, agent_id: str):
        path = self._path(agent_id)
        if path.exists():
            path.unlink()

    def list_agents(self) -> list[str]:
        return [p.stem for p in self.data_dir.glob("*.json")]


# ─────────────────────────────────────────
# MEMORY ENGINE
# ─────────────────────────────────────────

class MemoryEngine:
    """
    The cognitive core of an autonomous agent.
    
    Manages five memory layers, autonomously classifies importance,
    prunes strategically, and consolidates episodic → semantic memory.
    
    This is not a CRUD layer. This is a cognitive architecture.
    
    Usage:
        engine = MemoryEngine(agent_id="my_agent")
        engine.store(MemoryType.SHORT_TERM, "User said they prefer Python 3.11")
        memories = engine.retrieve(MemoryType.SHORT_TERM, limit=5)
        engine.prune()
    """

    def __init__(
        self,
        agent_id: str,
        storage: Optional[LocalStorage] = None,
        classifier: Optional[ImportanceClassifier] = None,
    ):
        self.agent_id   = agent_id
        self.storage    = storage or LocalStorage()
        self.classifier = classifier or ImportanceClassifier()
        self._memories: list[Memory] = []
        self._meta: dict = {}
        self._load()

    # ── Persistence ──────────────────────────────────────

    def _load(self):
        data = self.storage.load(self.agent_id)
        self._memories = [Memory.from_dict(m) for m in data.get("memories", [])]
        self._meta = data.get("meta", {
            "agent_id": self.agent_id,
            "created_at": datetime.now().isoformat(),
            "total_interactions": 0,
        })

    def _save(self):
        self.storage.save(self.agent_id, {
            "memories": [m.to_dict() for m in self._memories],
            "meta": self._meta,
        })

    # ── Core Operations ───────────────────────────────────

    def store(
        self,
        memory_type: MemoryType,
        content: str,
        importance: Optional[Importance] = None,
        tags: list = None,
        context: dict = None,
        explicit_importance: Optional[Importance] = None,
    ) -> Memory:
        """
        Store a memory. Auto-classifies importance if not provided.
        
        For SENSORY memory: replaces the single existing sensory entry.
        For SHORT_TERM: appends and prunes if over limit.
        For LONG_TERM: only stores if importance >= HIGH.
        """
        resolved_importance = importance or self.classifier.classify(content, explicit_importance)

        # LONG_TERM has a quality gate
        if memory_type == MemoryType.LONG_TERM:
            if IMPORTANCE_SCORE.get(resolved_importance, 0) < IMPORTANCE_SCORE[LONG_TERM_THRESHOLD]:
                # Redirect LOW/MEDIUM to SHORT_TERM instead
                memory_type = MemoryType.SHORT_TERM

        mem = Memory(
            id          = str(uuid.uuid4()),
            type        = memory_type,
            content     = content,
            importance  = resolved_importance,
            tags        = tags or [],
            context     = context or {},
        )

        # SENSORY: single slot — replace existing
        if memory_type == MemoryType.SENSORY:
            self._memories = [m for m in self._memories if m.type != MemoryType.SENSORY]

        self._memories.append(mem)
        self._prune_layer(memory_type)
        self._save()
        return mem

    def retrieve(
        self,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        query: Optional[str] = None,
    ) -> list[Memory]:
        """
        Retrieve memories by type. Optional simple text filter.
        Returns sorted by score (most valuable first).
        """
        results = self._memories

        if memory_type:
            results = [m for m in results if m.type == memory_type]

        if query:
            q = query.lower()
            results = [m for m in results if q in m.content.lower()]

        # Sort by composite score
        results = sorted(results, key=lambda m: m.score(), reverse=True)

        # Touch accessed memories (update recency)
        for m in results[:limit]:
            m.touch()

        self._save()
        return results[:limit]

    def retrieve_context(self, limit_per_layer: int = 3) -> dict[str, list[Memory]]:
        """
        Retrieve a balanced context snapshot across all memory layers.
        Used by the agent when building its prompt context.
        """
        return {
            "sensory":    self.retrieve(MemoryType.SENSORY,    limit=1),
            "short_term": self.retrieve(MemoryType.SHORT_TERM, limit=limit_per_layer),
            "long_term":  self.retrieve(MemoryType.LONG_TERM,  limit=limit_per_layer),
            "episodic":   self.retrieve(MemoryType.EPISODIC,   limit=limit_per_layer),
            "semantic":   self.retrieve(MemoryType.SEMANTIC,   limit=limit_per_layer),
        }

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """
        Search across all memory types for relevant content.
        Simple substring for now. Swap for embedding search when needed.
        """
        q = query.lower()
        results = [m for m in self._memories if q in m.content.lower()]
        return sorted(results, key=lambda m: m.score(), reverse=True)[:top_k]

    def forget(self, memory_id: str) -> bool:
        """Explicitly forget a specific memory by ID."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != memory_id]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def clear_type(self, memory_type: MemoryType, protect_critical: bool = True):
        """Clear all memories of a given type. Optionally protect CRITICAL ones."""
        def should_keep(m: Memory) -> bool:
            if m.type != memory_type:
                return True
            if protect_critical and m.importance == Importance.CRITICAL:
                return True
            return False
        self._memories = [m for m in self._memories if should_keep(m)]
        self._save()

    def clear_session(self, protect_long_term: bool = True):
        """
        Clear session-scoped memories (sensory + short_term).
        Long-term, episodic, and semantic survive by default.
        """
        session_types = {MemoryType.SENSORY, MemoryType.SHORT_TERM}
        if protect_long_term:
            self._memories = [m for m in self._memories if m.type not in session_types]
        else:
            self._memories = []
        self._save()

    # ── Pruning ───────────────────────────────────────────

    def _prune_layer(self, memory_type: MemoryType):
        """
        If a layer exceeds its limit, prune the lowest-scoring memories.
        CRITICAL memories are always protected.
        """
        layer = [m for m in self._memories if m.type == memory_type]
        limit = LAYER_LIMITS.get(memory_type, 100)

        if len(layer) <= limit:
            return

        # Protect critical
        protected = [m for m in layer if m.importance == Importance.CRITICAL]
        candidates = [m for m in layer if m.importance != Importance.CRITICAL]

        # Keep top N by score
        keep = limit - len(protected)
        survivors = sorted(candidates, key=lambda m: m.score(), reverse=True)[:max(0, keep)]

        keep_ids = {m.id for m in protected + survivors}
        self._memories = [m for m in self._memories if m.type != memory_type or m.id in keep_ids]

    def prune(self):
        """Run pruning across all layers."""
        for mt in MemoryType:
            self._prune_layer(mt)
        self._save()

    # ── Consolidation (Dream-like processing) ─────────────

    def consolidate(self) -> list[Memory]:
        """
        Episodic → Semantic consolidation.
        
        Takes highly-accessed episodic memories and extracts
        generalized concepts into semantic memory.
        
        This mimics how humans consolidate experiences into knowledge during sleep.
        Returns list of newly created semantic memories.
        """
        new_semantic: list[Memory] = []

        candidates = [
            m for m in self._memories
            if m.type == MemoryType.EPISODIC
            and m.access_count >= 3
            and IMPORTANCE_SCORE.get(m.importance, 0) >= 2
        ]

        for ep in candidates:
            # Create a semantic memory from this episodic one
            concept = f"[Consolidated from experience] {ep.content[:300]}"

            # Check if we already have something similar (simple check)
            existing = self.search(ep.content[:50])
            similar_semantic = [m for m in existing if m.type == MemoryType.SEMANTIC]
            if similar_semantic:
                continue  # Already consolidated

            sem = self.store(
                memory_type=MemoryType.SEMANTIC,
                content=concept,
                importance=Importance.HIGH,
                tags=["consolidated", "from_episodic"],
                context={"source_episodic_id": ep.id},
            )
            new_semantic.append(sem)

        return new_semantic

    # ── Introspection ────────────────────────────────────

    def stats(self) -> dict:
        """Memory statistics snapshot."""
        counts = {mt.value: 0 for mt in MemoryType}
        importance_dist = {imp.value: 0 for imp in Importance}

        for m in self._memories:
            counts[m.type.value] += 1
            importance_dist[m.importance.value] += 1

        total = len(self._memories)
        capacity = sum(LAYER_LIMITS.values())

        return {
            "agent_id":         self.agent_id,
            "total_memories":   total,
            "memory_pressure":  round(total / capacity, 3),
            "by_type":          counts,
            "by_importance":    importance_dist,
            "total_capacity":   capacity,
        }

    def format_context_for_prompt(self, limit_per_layer: int = 3) -> str:
        """
        Format memory context for inclusion in an LLM prompt.
        Returns a structured string the agent uses to recall context.
        """
        ctx = self.retrieve_context(limit_per_layer=limit_per_layer)
        lines = []

        label_map = {
            "sensory":    "[ CURRENT CONTEXT ]",
            "short_term": "[ RECENT INTERACTIONS ]",
            "long_term":  "[ IMPORTANT FACTS ]",
            "episodic":   "[ PAST EXPERIENCES ]",
            "semantic":   "[ LEARNED KNOWLEDGE ]",
        }

        for layer, memories in ctx.items():
            if memories:
                lines.append(label_map[layer])
                for m in memories:
                    lines.append(f"  • {m.content}")
                lines.append("")

        return "\n".join(lines) if lines else "[ No memory context available ]"

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryEngine(agent={self.agent_id}, "
            f"memories={s['total_memories']}, "
            f"pressure={s['memory_pressure']:.1%})"
        )
