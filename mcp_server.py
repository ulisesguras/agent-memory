"""
mcp_server.py — Agent Memory as an MCP Server

Makes Agent Memory discoverable and usable by ANY agent that speaks MCP:
Claude Desktop, GPT, Gemini, LangChain, PydanticAI, custom ReAct agents, etc.

Architecture:
  Any MCP-compatible agent
        ↓  (MCP protocol)
  mcp_server.py       (this file — the MCP interface)
        ↓
  core/agent.py + core/memory_engine.py  (the cognitive engine)

Usage:
  pip install mcp
  python mcp_server.py                    # stdio (Claude Desktop)
  python mcp_server.py --transport sse    # HTTP/SSE (remote agents)

Register in Claude Desktop (claude_desktop_config.json):
  {
    "mcpServers": {
      "agent-memory": {
        "command": "python",
        "args": ["/path/to/agent_memory/mcp_server.py"],
        "env": { "LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "xxx" }
      }
    }
  }
"""

import sys
import json
import asyncio
import argparse
import os
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import Agent, MockAdapter, VertexAIAdapter, AnthropicAdapter
from core.memory_engine import MemoryType, Importance, LocalStorage

MCP_AVAILABLE = False
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    import mcp.types as mcp_types
    MCP_AVAILABLE = True
except ImportError:
    mcp_types = None

# ─── CONFIG ─────────────────────────────────────────
DATA_DIR     = os.environ.get("AGENT_MEMORY_DATA_DIR", "./data/agents")
IDENTITY_DIR = os.environ.get("AGENT_MEMORY_IDENTITY_DIR", "./identity")
LLM_BACKEND  = os.environ.get("LLM_BACKEND", "mock")
LLM_MODEL    = os.environ.get("LLM_MODEL", None)

_agents: dict[str, Agent] = {}

def build_llm():
    if LLM_BACKEND == "vertex":   return VertexAIAdapter(model=LLM_MODEL or "gemini-2.0-flash-exp")
    if LLM_BACKEND == "anthropic": return AnthropicAdapter(model=LLM_MODEL or "claude-sonnet-4-20250514")
    return MockAdapter()

def get_agent(agent_id: str, agent_type: str = "general") -> Agent:
    if agent_id not in _agents:
        _agents[agent_id] = Agent(
            agent_id=agent_id, agent_type=agent_type,
            llm=build_llm(), identity_dir=IDENTITY_DIR, data_dir=DATA_DIR,
        )
    return _agents[agent_id]

# ─── TOOL CATALOG (works with or without MCP SDK) ───
TOOL_CATALOG = [
    {"name": "memory_interact",
     "description": "Full ReAct loop: observe, load all 5 memory layers, generate response, score importance, store. Returns response + thought trace.",
     "inputSchema": {"type": "object", "required": ["agent_id", "session_id", "user_input"],
         "properties": {"agent_id": {"type": "string"}, "session_id": {"type": "string"},
                        "user_input": {"type": "string"}, "agent_type": {"type": "string", "default": "general"},
                        "importance": {"type": "string", "enum": ["critical","high","medium","low"]}}}},
    {"name": "memory_store",
     "description": "Explicitly store a fact. CRITICAL importance by default — never forgotten unless deleted.",
     "inputSchema": {"type": "object", "required": ["agent_id", "content"],
         "properties": {"agent_id": {"type": "string"}, "content": {"type": "string"},
                        "memory_type": {"type": "string", "default": "long_term"},
                        "importance": {"type": "string", "default": "critical"},
                        "tags": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "memory_retrieve",
     "description": "Retrieve memories from a specific cognitive layer. Optional text filter.",
     "inputSchema": {"type": "object", "required": ["agent_id"],
         "properties": {"agent_id": {"type": "string"}, "memory_type": {"type": "string"},
                        "query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}}},
    {"name": "memory_context",
     "description": "Get formatted memory context snapshot ready for LLM prompt injection.",
     "inputSchema": {"type": "object", "required": ["agent_id"],
         "properties": {"agent_id": {"type": "string"}, "limit_per_layer": {"type": "integer", "default": 3}}}},
    {"name": "memory_stats",
     "description": "Memory health: total count, pressure (0-1), distribution by type and importance.",
     "inputSchema": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}}}},
    {"name": "memory_search",
     "description": "Search across ALL five memory layers for relevant content.",
     "inputSchema": {"type": "object", "required": ["agent_id", "query"],
         "properties": {"agent_id": {"type": "string"}, "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5}}}},
    {"name": "memory_consolidate",
     "description": "Run episodic→semantic consolidation. Extracts learned concepts from repeated experiences (dream-like processing).",
     "inputSchema": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}}}},
    {"name": "memory_new_session",
     "description": "Start new session. Clears sensory+short-term. Long-term, episodic, semantic memory preserved.",
     "inputSchema": {"type": "object", "required": ["agent_id"], "properties": {"agent_id": {"type": "string"}}}},
    {"name": "memory_forget",
     "description": "Delete a specific memory by ID.",
     "inputSchema": {"type": "object", "required": ["agent_id", "memory_id"],
         "properties": {"agent_id": {"type": "string"}, "memory_id": {"type": "string"}}}},
    {"name": "memory_list_agents",
     "description": "List all agents with stored memory, their count and pressure.",
     "inputSchema": {"type": "object", "properties": {}}},
]

TOOLS = []
if MCP_AVAILABLE:
    TOOLS = [mcp_types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
             for t in TOOL_CATALOG]

# ─── TOOL HANDLERS ──────────────────────────────────

async def handle_tool(name: str, arguments: dict) -> list:
    def ok(data: Any):
        if MCP_AVAILABLE:
            return [mcp_types.TextContent(type="text", text=json.dumps(data, indent=2, default=str))]
        return [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]

    def err(msg: str):
        return ok({"error": msg})

    try:
        if name == "memory_interact":
            agent = get_agent(arguments["agent_id"], arguments.get("agent_type", "general"))
            explicit = Importance(arguments["importance"]) if "importance" in arguments else None
            resp = agent.interact(session_id=arguments["session_id"], user_input=arguments["user_input"],
                                   explicit_importance=explicit)
            return ok({"agent_id": resp.agent_id, "session_id": resp.session_id, "response": resp.text,
                       "thought": {"importance": resp.thought.importance_score,
                                   "memories_stored": len(resp.thought.memories_stored),
                                   "duration_ms": resp.thought.duration_ms},
                       "memory_stats": resp.memory_stats})

        elif name == "memory_store":
            agent = get_agent(arguments["agent_id"])
            mt  = MemoryType(arguments.get("memory_type", "long_term"))
            imp = Importance(arguments.get("importance", "critical"))
            mem = agent.memory.store(mt, arguments["content"], importance=imp,
                                      tags=arguments.get("tags", ["explicit", "mcp"]))
            return ok({"status": "stored", "memory_id": mem.id, "type": mem.type.value, "importance": mem.importance.value})

        elif name == "memory_retrieve":
            agent = get_agent(arguments["agent_id"])
            mt    = MemoryType(arguments["memory_type"]) if "memory_type" in arguments else None
            mems  = agent.memory.retrieve(memory_type=mt, query=arguments.get("query"),
                                           limit=arguments.get("limit", 10))
            return ok({"count": len(mems), "memories": [
                {"id": m.id, "type": m.type.value, "importance": m.importance.value,
                 "content": m.content, "access_count": m.access_count, "tags": m.tags}
                for m in mems]})

        elif name == "memory_context":
            agent = get_agent(arguments["agent_id"])
            lpl   = arguments.get("limit_per_layer", 3)
            ctx   = agent.memory.retrieve_context(limit_per_layer=lpl)
            return ok({"context_text": agent.memory.format_context_for_prompt(lpl),
                       "layers": {layer: [{"content": m.content, "importance": m.importance.value} for m in mems]
                                  for layer, mems in ctx.items()}})

        elif name == "memory_stats":
            return ok(get_agent(arguments["agent_id"]).stats())

        elif name == "memory_search":
            agent   = get_agent(arguments["agent_id"])
            results = agent.search_memory(arguments["query"], top_k=arguments.get("top_k", 5))
            return ok({"query": arguments["query"], "count": len(results),
                       "results": [{"id": m.id, "type": m.type.value, "importance": m.importance.value,
                                    "content": m.content} for m in results]})

        elif name == "memory_consolidate":
            agent    = get_agent(arguments["agent_id"])
            new_mems = agent.memory.consolidate()
            return ok({"consolidated": len(new_mems),
                       "new_semantic": [{"id": m.id, "content": m.content[:100]} for m in new_mems]})

        elif name == "memory_new_session":
            get_agent(arguments["agent_id"]).new_session()
            return ok({"status": "new_session_started", "long_term_preserved": True})

        elif name == "memory_forget":
            success = get_agent(arguments["agent_id"]).forget(arguments["memory_id"])
            return ok({"success": success, "memory_id": arguments["memory_id"]})

        elif name == "memory_list_agents":
            storage = LocalStorage(DATA_DIR)
            result  = []
            for aid in storage.list_agents():
                s = get_agent(aid).stats()
                result.append({"agent_id": aid, "total_memories": s["total_memories"],
                                "memory_pressure": s["memory_pressure"]})
            return ok({"agents": result, "count": len(result)})

        else:
            return err(f"Unknown tool: {name}")

    except Exception as e:
        return err(f"Tool execution failed: {str(e)}")


# ─── MCP SERVER ─────────────────────────────────────

def create_server():
    server = Server("agent-memory")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        return await handle_tool(name, arguments)

    @server.list_resources()
    async def list_resources():
        return [
            mcp_types.Resource(uri="agent-memory://manifest", name="Agent Memory Manifest",
                               description="Machine-readable capability manifest", mimeType="application/json"),
            mcp_types.Resource(uri="agent-memory://docs", name="Documentation",
                               description="Architecture and usage", mimeType="text/markdown"),
            mcp_types.Resource(uri="agent-memory://tools", name="Tool Catalog",
                               description="All available tools with schemas", mimeType="application/json"),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if uri == "agent-memory://manifest":
            p = Path(__file__).parent / ".well-known" / "agent.json"
            return p.read_text() if p.exists() else json.dumps({"error": "not found"})
        elif uri == "agent-memory://docs":
            p = Path(__file__).parent / "README.md"
            return p.read_text() if p.exists() else "Documentation not found."
        elif uri == "agent-memory://tools":
            return json.dumps(TOOL_CATALOG, indent=2)
        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ─── FALLBACK (no MCP SDK) ──────────────────────────

def run_without_mcp():
    print("\n Agent Memory MCP Server — Tool Catalog")
    print("Install MCP SDK first: pip install mcp\n")
    print(f"{'Tool':<25} Description")
    print("─" * 70)
    for t in TOOL_CATALOG:
        print(f"  {t['name']:<23} {t['description'][:50]}...")
    print("\nResources:")
    print("  agent-memory://manifest   Machine-readable capability manifest")
    print("  agent-memory://docs       Architecture documentation")
    print("  agent-memory://tools      Tool catalog with JSON schemas")
    print("\nClaude Desktop config (claude_desktop_config.json):")
    print(json.dumps({"mcpServers": {"agent-memory": {
        "command": "python",
        "args": [str(Path(__file__).absolute())],
        "env": {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "your_key"}
    }}}, indent=2))


# ─── MAIN ───────────────────────────────────────────

async def run_stdio():
    server = create_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="agent-memory", server_version="3.0.0",
            capabilities=server.get_capabilities(notification_options=None, experimental_options={}),
        ))

async def run_sse(host="0.0.0.0", port=8081):
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount

    server = create_server()
    sse    = SseServerTransport("/messages")

    async def handle_sse(req):
        async with sse.connect_sse(req.scope, req.receive, req._send) as streams:
            await server.run(streams[0], streams[1], InitializationOptions(
                server_name="agent-memory", server_version="3.0.0",
                capabilities=server.get_capabilities(notification_options=None, experimental_options={}),
            ))

    async def handle_msg(req):
        await sse.handle_post_message(req.scope, req.receive, req._send)

    starlette_app = Starlette(routes=[Route("/sse", handle_sse), Mount("/messages", handle_msg)])
    print(f"\n Agent Memory MCP Server — SSE mode")
    print(f" Listening: http://{host}:{port}/sse")
    print(f" Tools: {len(TOOLS)}\n")
    await uvicorn.Server(uvicorn.Config(starlette_app, host=host, port=port, log_level="info")).serve()

def main():
    if not MCP_AVAILABLE:
        run_without_mcp()
        return

    parser = argparse.ArgumentParser(description="Agent Memory MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if args.transport == "sse":
        asyncio.run(run_sse(args.host, args.port))
    else:
        asyncio.run(run_stdio())

if __name__ == "__main__":
    main()
