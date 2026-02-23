"""
cli/run_agent.py

Interactive CLI for running an Agent Memory instance.
No web server needed. Run locally, experiment freely.

Usage:
    python cli/run_agent.py --agent my_agent --type general --llm mock
    python cli/run_agent.py --agent researcher_001 --type researcher --llm vertex
    python cli/run_agent.py --agent analyst --type analyst --llm anthropic

Commands during session:
    /stats          — Show memory statistics
    /memory [type]  — Show memories (type: all|short|long|episodic|semantic)
    /search <query> — Search memory
    /remember <fact>— Store with CRITICAL importance
    /forget <id>    — Delete a memory by ID
    /consolidate    — Run episodic→semantic consolidation
    /new            — Start new session (preserves long-term memory)
    /clear          — Clear all memory for this agent
    /export         — Export all memory to JSON
    /quit           — Exit
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import Agent, MockAdapter, VertexAIAdapter, AnthropicAdapter
from core.memory_engine import MemoryType, Importance


# ─────────────────────────────────────────
# COLOR OUTPUT
# ─────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GRAY   = "\033[90m"
    PURPLE = "\033[95m"
    BLUE   = "\033[94m"

def banner(agent_id: str, agent_type: str, llm_name: str):
    print(f"""
{C.CYAN}{C.BOLD}
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝ 
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝  
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║   
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝  
{C.RESET}
{C.GRAY}Autonomous Agent Memory System — Experimental Platform{C.RESET}
{C.PURPLE}Agent:{C.RESET} {agent_id}  {C.PURPLE}Type:{C.RESET} {agent_type}  {C.PURPLE}LLM:{C.RESET} {llm_name}
{C.GRAY}Type /help for commands{C.RESET}
""")


def print_thought(trace, verbose: bool = False):
    if verbose:
        print(f"\n{C.GRAY}{trace.render()}{C.RESET}")
    else:
        ts = trace.timestamp[11:19]
        mem_count = sum(len(v) for v in trace.memory_loaded.values()) if trace.memory_loaded else 0
        print(f"{C.GRAY}[{ts}] memory={mem_count} importance={trace.importance_score} stored={len(trace.memories_stored)} {trace.duration_ms:.0f}ms{C.RESET}")


def print_stats(stats: dict):
    print(f"\n{C.CYAN}━━━ MEMORY STATS ━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print(f"  Agent:    {stats['agent_id']}")
    print(f"  Total:    {stats['total_memories']} / {stats['total_capacity']}")
    print(f"  Pressure: {stats['memory_pressure']:.1%}")
    print(f"\n  {C.BOLD}By Type:{C.RESET}")
    for t, c in stats["by_type"].items():
        bar = "█" * c + "░" * max(0, 10 - min(c, 10))
        print(f"    {t:<12} {bar}  {c}")
    print(f"\n  {C.BOLD}By Importance:{C.RESET}")
    for imp, c in stats["by_importance"].items():
        color = {
            "critical": C.RED,
            "high": C.YELLOW,
            "medium": C.GREEN,
            "low": C.GRAY,
        }.get(imp, C.RESET)
        print(f"    {color}{imp:<10}{C.RESET} {c}")
    print()


def print_memories(memories: list, label: str = ""):
    if label:
        print(f"\n{C.CYAN}━━━ {label.upper()} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    if not memories:
        print(f"  {C.GRAY}(no memories){C.RESET}")
        return
    for m in memories:
        imp_color = {
            "critical": C.RED,
            "high": C.YELLOW,
            "medium": C.GREEN,
            "low": C.GRAY,
        }.get(m.importance.value, C.RESET)
        ts = m.created_at[:10] if m.created_at else "?"
        content_preview = m.content[:80].replace("\n", " ")
        print(f"  {imp_color}[{m.importance.value}]{C.RESET} {C.PURPLE}({m.type.value}){C.RESET} {C.GRAY}{ts}{C.RESET}")
        print(f"    {content_preview}...")
        print(f"    {C.GRAY}id={m.id[:8]}  accesses={m.access_count}{C.RESET}")
    print()


# ─────────────────────────────────────────
# COMMAND HANDLER
# ─────────────────────────────────────────

def handle_command(cmd: str, agent: Agent, session_id: str, verbose: bool) -> bool:
    """Returns True to continue, False to quit."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/quit" or command == "/exit":
        print(f"\n{C.YELLOW}Session ended. Long-term memory preserved.{C.RESET}")
        return False

    elif command == "/stats":
        print_stats(agent.stats())

    elif command == "/memory":
        type_map = {
            "short": MemoryType.SHORT_TERM,
            "long":  MemoryType.LONG_TERM,
            "ep":    MemoryType.EPISODIC,
            "epis":  MemoryType.EPISODIC,
            "sem":   MemoryType.SEMANTIC,
            "sensory": MemoryType.SENSORY,
        }
        if arg and arg in type_map:
            mems = agent.memory.retrieve(type_map[arg], limit=20)
            print_memories(mems, f"{arg} memory")
        else:
            for mt in MemoryType:
                mems = agent.memory.retrieve(mt, limit=5)
                print_memories(mems, mt.value)

    elif command == "/search":
        if not arg:
            print(f"{C.RED}Usage: /search <query>{C.RESET}")
        else:
            results = agent.search_memory(arg, top_k=5)
            print_memories(results, f"search: '{arg}'")

    elif command == "/remember":
        if not arg:
            print(f"{C.RED}Usage: /remember <fact>{C.RESET}")
        else:
            m = agent.remember(arg)
            print(f"{C.GREEN}✓ Stored as CRITICAL: {m.id[:8]}{C.RESET}")

    elif command == "/forget":
        if not arg:
            print(f"{C.RED}Usage: /forget <memory_id>{C.RESET}")
        else:
            success = agent.forget(arg)
            if success:
                print(f"{C.GREEN}✓ Memory {arg[:8]} forgotten{C.RESET}")
            else:
                print(f"{C.RED}✗ Memory not found{C.RESET}")

    elif command == "/consolidate":
        new_sem = agent.memory.consolidate()
        if new_sem:
            print(f"{C.GREEN}✓ Consolidated {len(new_sem)} episodic → semantic memories{C.RESET}")
        else:
            print(f"{C.YELLOW}Nothing to consolidate yet (need episodic with access_count ≥ 3){C.RESET}")

    elif command == "/new":
        agent.new_session()
        print(f"{C.GREEN}✓ New session started. Long-term memory preserved.{C.RESET}")

    elif command == "/clear":
        confirm = input(f"{C.RED}Clear ALL memory for agent '{agent.agent_id}'? (yes/no): {C.RESET}")
        if confirm.lower() == "yes":
            agent.memory._memories = []
            agent.memory._save()
            print(f"{C.GREEN}✓ All memory cleared{C.RESET}")
        else:
            print("Cancelled.")

    elif command == "/export":
        path = f"./data/exports/{agent.agent_id}_export.json"
        Path("./data/exports").mkdir(parents=True, exist_ok=True)
        data = {
            "agent_id": agent.agent_id,
            "exported_at": __import__("datetime").datetime.now().isoformat(),
            "memories": [m.to_dict() for m in agent.memory._memories],
            "stats": agent.stats(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"{C.GREEN}✓ Exported to {path}{C.RESET}")

    elif command == "/verbose":
        verbose = not verbose
        state = "ON" if verbose else "OFF"
        print(f"{C.YELLOW}Verbose mode: {state}{C.RESET}")

    elif command == "/help":
        print(f"""
{C.CYAN}━━━ COMMANDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}
  /stats              Memory statistics
  /memory [type]      Show memories (short/long/ep/sem/sensory)
  /search <query>     Search all memory
  /remember <fact>    Store as CRITICAL importance
  /forget <id>        Delete memory by ID (first 8 chars ok)
  /consolidate        Run episodic→semantic consolidation
  /new                New session (preserves long-term)
  /clear              Clear ALL memory (asks confirmation)
  /export             Export memory to JSON
  /verbose            Toggle thought trace detail
  /quit               Exit
{C.GRAY}Just type anything else to chat with the agent.{C.RESET}
""")

    return True


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def build_llm(llm_name: str, model: str = None) -> object:
    if llm_name == "vertex":
        return VertexAIAdapter(model=model or "gemini-2.0-flash-exp")
    elif llm_name == "anthropic":
        return AnthropicAdapter(model=model or "claude-sonnet-4-20250514")
    else:
        return MockAdapter()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory — Interactive CLI")
    parser.add_argument("--agent",   default="default_agent", help="Agent ID")
    parser.add_argument("--type",    default="general",       help="Agent type (general|analyst|researcher|assistant|specialist)")
    parser.add_argument("--llm",     default="mock",          help="LLM backend (mock|vertex|anthropic)")
    parser.add_argument("--model",   default=None,            help="Specific model name")
    parser.add_argument("--session", default=None,            help="Session ID (auto-generated if not set)")
    parser.add_argument("--verbose", action="store_true",     help="Show full thought trace")
    parser.add_argument("--identity",default="./identity",    help="Path to identity files directory")
    parser.add_argument("--data",    default="./data/agents", help="Path to agent data directory")
    args = parser.parse_args()

    import uuid as _uuid
    session_id = args.session or f"session_{_uuid.uuid4().hex[:8]}"
    llm = build_llm(args.llm, args.model)

    agent = Agent(
        agent_id     = args.agent,
        agent_type   = args.type,
        llm          = llm,
        identity_dir = args.identity,
        data_dir     = args.data,
    )

    banner(args.agent, args.type, llm.get_name())
    print(f"{C.GRAY}Session: {session_id}{C.RESET}")
    print_stats(agent.stats())

    verbose = args.verbose

    while True:
        try:
            user_input = input(f"\n{C.BOLD}{C.BLUE}You:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.YELLOW}Goodbye. Memory saved.{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(user_input, agent, session_id, verbose)
            if not should_continue:
                break
            continue

        # Normal interaction
        response = agent.interact(session_id, user_input)

        print(f"\n{C.BOLD}{C.GREEN}Agent:{C.RESET} {response.text}")
        print_thought(response.thought, verbose=verbose)


if __name__ == "__main__":
    main()
