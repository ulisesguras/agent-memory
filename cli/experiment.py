"""
cli/experiment.py

Experiment runner for Agent Memory research.
Run predefined experiments to observe memory behavior.

Usage:
    python cli/experiment.py --run basic_memory
    python cli/experiment.py --run importance_test
    python cli/experiment.py --run consolidation
    python cli/experiment.py --run multi_session
    python cli/experiment.py --list
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import Agent, MockAdapter
from core.memory_engine import MemoryType, Importance


class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GRAY   = "\033[90m"


def section(title: str):
    print(f"\n{C.CYAN}{C.BOLD}══ {title} {'═' * (50 - len(title))}{C.RESET}")


def step(n: int, text: str):
    print(f"\n{C.YELLOW}[Step {n}]{C.RESET} {text}")


def result(key: str, value):
    print(f"  {C.GREEN}✓{C.RESET} {key}: {C.BOLD}{value}{C.RESET}")


def show_memory_snapshot(agent: Agent):
    stats = agent.stats()
    print(f"\n  Memory state: {stats['total_memories']} total, pressure={stats['memory_pressure']:.1%}")
    for t, c in stats["by_type"].items():
        if c > 0:
            bar = "█" * min(c, 20)
            print(f"    {t:<12} {bar}  ({c})")


# ─────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────

def experiment_basic_memory():
    """
    Test basic store/retrieve across all memory types.
    """
    section("EXPERIMENT: Basic Memory Store/Retrieve")

    agent = Agent(agent_id="exp_basic", llm=MockAdapter(), data_dir="/tmp/agent_exp")
    agent.memory._memories = []  # Start clean

    step(1, "Store one entry per memory type")
    types_data = [
        (MemoryType.SENSORY,    "User just said hello",              Importance.LOW),
        (MemoryType.SHORT_TERM, "User is debugging a Python error",  Importance.MEDIUM),
        (MemoryType.LONG_TERM,  "User's name is Alex",               Importance.CRITICAL),
        (MemoryType.EPISODIC,   "Fixed authentication bug on Jan 15", Importance.HIGH),
        (MemoryType.SEMANTIC,   "User prefers Python over JavaScript", Importance.HIGH),
    ]

    for mt, content, imp in types_data:
        m = agent.memory.store(mt, content, importance=imp)
        result(mt.value, f"stored (id={m.id[:8]})")

    show_memory_snapshot(agent)

    step(2, "Retrieve each type")
    for mt, _, _ in types_data:
        mems = agent.memory.retrieve(mt, limit=1)
        if mems:
            result(mt.value, f'"{mems[0].content[:50]}"')

    step(3, "Search across all memory")
    results = agent.memory.search("Python", top_k=3)
    result("search 'Python'", f"{len(results)} matches found")
    for r in results:
        print(f"    [{r.type.value}] {r.content[:60]}")

    step(4, "Format context for prompt")
    ctx = agent.memory.format_context_for_prompt()
    print(f"\n{C.GRAY}{ctx}{C.RESET}")

    print(f"\n{C.GREEN}✓ Basic memory experiment complete{C.RESET}")


def experiment_importance():
    """
    Test the importance classifier with various inputs.
    """
    section("EXPERIMENT: Importance Classification")

    agent = Agent(agent_id="exp_importance", llm=MockAdapter(), data_dir="/tmp/agent_exp")
    agent.memory._memories = []

    classifier = agent.memory.classifier

    test_cases = [
        ("Remember this: the API key is abc123",    Importance.CRITICAL),
        ("This is the most important thing I'll say", Importance.CRITICAL),
        ("The main framework we use is FastAPI",    Importance.HIGH),
        ("My name is Maria and I work in Buenos Aires", Importance.HIGH),
        ("We're analyzing the quarterly results which show a 23% improvement in user retention", Importance.HIGH),
        ("okay thanks",                             Importance.LOW),
        ("yes",                                     Importance.LOW),
        ("Could you explain how memory pruning works?", Importance.MEDIUM),
    ]

    step(1, "Test automatic classification")
    correct = 0
    for text, expected in test_cases:
        got = classifier.classify(text)
        match = got == expected
        correct += match
        status = f"{C.GREEN}✓{C.RESET}" if match else f"{C.RED}✗{C.RESET}"
        print(f"  {status} [{got.value:<8}] expected={expected.value:<8}  \"{text[:50]}\"")

    print(f"\n  Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")

    step(2, "Test long-term quality gate")
    # LOW importance should be redirected to short-term
    m = agent.memory.store(MemoryType.LONG_TERM, "okay thanks", importance=Importance.LOW)
    result("LOW→LONG_TERM redirected to", m.type.value)

    # HIGH importance should pass through
    m2 = agent.memory.store(MemoryType.LONG_TERM, "User always uses async/await pattern", importance=Importance.HIGH)
    result("HIGH→LONG_TERM stored as", m2.type.value)

    print(f"\n{C.GREEN}✓ Importance experiment complete{C.RESET}")


def experiment_pruning():
    """
    Test memory pruning behavior under capacity pressure.
    """
    section("EXPERIMENT: Pruning Under Pressure")

    agent = Agent(agent_id="exp_prune", llm=MockAdapter(), data_dir="/tmp/agent_exp")
    agent.memory._memories = []

    step(1, "Fill short-term memory beyond limit (limit=20)")
    for i in range(25):
        imp = Importance.CRITICAL if i == 0 else (
            Importance.HIGH if i < 5 else (
                Importance.MEDIUM if i < 15 else Importance.LOW
            )
        )
        agent.memory.store(MemoryType.SHORT_TERM, f"Interaction {i}: some content here", importance=imp)

    stats = agent.stats()
    result("short_term count after 25 stores", stats["by_type"]["short_term"])
    result("critical memories protected", stats["by_importance"]["critical"])

    step(2, "Verify CRITICAL memories survived pruning")
    criticals = [m for m in agent.memory._memories if m.importance == Importance.CRITICAL]
    result("critical memories present", len(criticals))
    if criticals:
        print(f"    Survivor: \"{criticals[0].content}\"")

    step(3, "Check what was pruned (should be LOWs first)")
    lows = [m for m in agent.memory._memories if m.importance == Importance.LOW]
    result("low-importance memories remaining", len(lows))

    print(f"\n{C.GREEN}✓ Pruning experiment complete{C.RESET}")


def experiment_consolidation():
    """
    Test episodic → semantic consolidation.
    """
    section("EXPERIMENT: Episodic → Semantic Consolidation (Dreaming)")

    agent = Agent(agent_id="exp_consolidate", llm=MockAdapter(), data_dir="/tmp/agent_exp")
    agent.memory._memories = []

    step(1, "Store episodic memories and simulate repeated access")
    for i in range(5):
        m = agent.memory.store(
            MemoryType.EPISODIC,
            f"User always starts sessions by asking about their Python project",
            importance=Importance.HIGH,
            tags=["pattern_detected"]
        )
        # Simulate accesses
        for _ in range(3):
            m.touch()
        agent.memory._save()

    step(2, "Run consolidation")
    before_sem = agent.stats()["by_type"]["semantic"]
    new_sem = agent.memory.consolidate()
    after_sem = agent.stats()["by_type"]["semantic"]

    result("semantic memories before", before_sem)
    result("semantic memories after", after_sem)
    result("new semantic memories created", len(new_sem))

    if new_sem:
        print(f"\n  New semantic memory content:")
        print(f"  {C.GRAY}\"{new_sem[0].content[:100]}\"{C.RESET}")

    step(3, "Re-run consolidation (should not duplicate)")
    new_sem2 = agent.memory.consolidate()
    result("duplicate consolidation prevented", len(new_sem2) == 0)

    print(f"\n{C.GREEN}✓ Consolidation experiment complete{C.RESET}")


def experiment_multi_session():
    """
    Test memory persistence across sessions.
    """
    section("EXPERIMENT: Cross-Session Memory Persistence")

    agent_id = "exp_multi_session"
    data_dir = "/tmp/agent_exp"

    step(1, "Session A — store important facts")
    agent_a = Agent(agent_id=agent_id, llm=MockAdapter(), data_dir=data_dir)
    agent_a.memory._memories = []

    agent_a.memory.store(MemoryType.LONG_TERM, "User's project is called DynamicStorage", importance=Importance.CRITICAL)
    agent_a.memory.store(MemoryType.LONG_TERM, "User prefers Python 3.11", importance=Importance.HIGH)
    agent_a.memory.store(MemoryType.SHORT_TERM, "We just discussed FastAPI architecture", importance=Importance.MEDIUM)

    result("memories stored in session A", agent_a.stats()["total_memories"])

    step(2, "End session A — clear session memory (simulate close)")
    agent_a.new_session()  # Clears sensory + short_term
    result("memories after session end", agent_a.stats()["total_memories"])
    result("long_term survived", agent_a.stats()["by_type"]["long_term"])

    step(3, "Session B — new agent instance, same agent_id (simulates restart)")
    agent_b = Agent(agent_id=agent_id, llm=MockAdapter(), data_dir=data_dir)
    stats_b = agent_b.stats()
    result("memories loaded in session B", stats_b["total_memories"])
    result("long_term memories available", stats_b["by_type"]["long_term"])

    step(4, "Verify the agent remembers across sessions")
    context = agent_b.memory.format_context_for_prompt()
    remembers_project = "DynamicStorage" in context
    remembers_python = "Python" in context
    result("remembers project name", remembers_project)
    result("remembers Python preference", remembers_python)

    step(5, "Show context the agent will use")
    print(f"\n{C.GRAY}{context}{C.RESET}")

    print(f"\n{C.GREEN}✓ Multi-session experiment complete{C.RESET}")


def experiment_full_agent():
    """
    Run a full agent interaction loop with thought traces.
    """
    section("EXPERIMENT: Full Agent ReAct Loop")

    agent = Agent(agent_id="exp_full", llm=MockAdapter(), data_dir="/tmp/agent_exp")
    agent.memory._memories = []

    conversations = [
        ("sess_001", "My name is Sofia and I'm building a memory system for AI agents"),
        ("sess_001", "Remember this is critical: the project deadline is March 15, 2026"),
        ("sess_001", "What do you remember about me so far?"),
        ("sess_001", "I prefer concise responses without bullet points"),
        ("sess_002", "I'm back. What's the most important thing you remember?"),
    ]

    for i, (session_id, user_input) in enumerate(conversations, 1):
        step(i, f"[{session_id}] \"{user_input}\"")
        response = agent.interact(session_id, user_input)
        print(f"\n  {C.GREEN}Response:{C.RESET} {response.text[:200]}")
        print(response.thought.render())

        if i == 4:  # After session 001
            agent.new_session()
            print(f"\n  {C.YELLOW}→ Session ended. Starting session 002...{C.RESET}")

    step(len(conversations) + 1, "Final memory snapshot")
    show_memory_snapshot(agent)

    print(f"\n{C.GREEN}✓ Full agent experiment complete{C.RESET}")


# ─────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────

EXPERIMENTS = {
    "basic_memory":    (experiment_basic_memory,    "Store/retrieve across all 5 memory types"),
    "importance":      (experiment_importance,       "Automatic importance classification"),
    "pruning":         (experiment_pruning,          "Memory pruning under capacity pressure"),
    "consolidation":   (experiment_consolidation,    "Episodic→semantic dreaming consolidation"),
    "multi_session":   (experiment_multi_session,    "Cross-session memory persistence"),
    "full_agent":      (experiment_full_agent,       "Complete ReAct loop with thought traces"),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Memory Experiments")
    parser.add_argument("--run",  default=None, help="Experiment name to run")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--all",  action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if args.list or (not args.run and not args.all):
        print(f"\n{C.CYAN}{C.BOLD}Available Experiments:{C.RESET}\n")
        for name, (_, desc) in EXPERIMENTS.items():
            print(f"  {C.YELLOW}{name:<20}{C.RESET}  {desc}")
        print(f"\n{C.GRAY}Usage: python cli/experiment.py --run <name>{C.RESET}\n")
        return

    to_run = list(EXPERIMENTS.keys()) if args.all else [args.run]

    for exp_name in to_run:
        if exp_name not in EXPERIMENTS:
            print(f"{C.RED}Unknown experiment: {exp_name}{C.RESET}")
            continue
        fn, _ = EXPERIMENTS[exp_name]
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"\n{C.GRAY}Elapsed: {elapsed:.2f}s{C.RESET}\n")


if __name__ == "__main__":
    main()
