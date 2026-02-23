"""
core/server.py  (v3.1 — Agent Discovery Layer)

FastAPI server with three layers Gary Tan describes:
  1. API pública         → all existing endpoints
  2. Manifest discovery  → GET /.well-known/agent.json
  3. Métricas en tiempo real → GET /metrics  (liveness + reliability + confidence)
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.agent import Agent, MockAdapter, VertexAIAdapter, AnthropicAdapter
from core.memory_engine import MemoryType, Importance, LocalStorage
from core.metrics import get_metrics

LLM_BACKEND  = os.environ.get("LLM_BACKEND", "mock")
LLM_MODEL    = os.environ.get("LLM_MODEL", None)
IDENTITY_DIR = os.environ.get("IDENTITY_DIR", "./identity")
DATA_DIR     = os.environ.get("DATA_DIR", "./data/agents")

metrics = get_metrics()
_agents: dict[str, Agent] = {}

def build_llm():
    if LLM_BACKEND == "vertex":
        return VertexAIAdapter(model=LLM_MODEL or "gemini-2.0-flash-exp")
    elif LLM_BACKEND == "anthropic":
        return AnthropicAdapter(model=LLM_MODEL or "claude-sonnet-4-20250514")
    return MockAdapter()

def get_agent(agent_id: str, agent_type: str = "general") -> Agent:
    if agent_id not in _agents:
        _agents[agent_id] = Agent(
            agent_id=agent_id, agent_type=agent_type,
            llm=build_llm(), identity_dir=IDENTITY_DIR, data_dir=DATA_DIR,
        )
    return _agents[agent_id]

def total_memories() -> int:
    try:
        return sum(get_agent(aid).stats()["total_memories"] for aid in LocalStorage(DATA_DIR).list_agents())
    except:
        return 0

class InteractRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    agent_id: str = "default"
    agent_type: str = "general"
    user_input: str
    explicit_importance: Optional[str] = None

class MemoryQueryRequest(BaseModel):
    agent_id: str
    query: str
    memory_type: Optional[str] = None
    limit: int = 10

class RememberRequest(BaseModel):
    agent_id: str
    fact: str
    memory_type: Optional[str] = "long_term"

class ConsolidateRequest(BaseModel):
    agent_id: str

app = FastAPI(
    title="Agent Memory", description="Autonomous cognitive memory layer. MCP-compatible.",
    version="3.1.0", docs_url="/docs",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/.well-known/agent.json", tags=["Discovery"])
def agent_manifest():
    """Machine-readable capability manifest. Discovered by agents before they decide to use this service."""
    manifest_path = Path(__file__).parent.parent / ".well-known" / "agent.json"
    if manifest_path.exists():
        return JSONResponse(content=json.loads(manifest_path.read_text()))
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/metrics", tags=["Discovery"])
def reliability_metrics():
    """Real-time reliability metrics: liveness, reliability, confidence. Used by agent scoring functions."""
    snap = metrics.snapshot(active_agents=len(_agents), total_memories=total_memories())
    return {
        "liveness": snap.liveness,
        "reliability": {
            "success_rate_1h": snap.success_rate_1h,
            "success_rate_24h": snap.success_rate_24h,
            "uptime_pct_24h": snap.uptime_pct_24h,
            "uptime_pct_30d": snap.uptime_pct_30d,
        },
        "confidence": snap.confidence_score,
        "latency": {"p50_ms": snap.latency_p50_ms, "p95_ms": snap.latency_p95_ms, "p99_ms": snap.latency_p99_ms},
        "usage": {
            "total_operations": snap.total_operations,
            "operations_last_1h": snap.operations_last_1h,
            "operations_last_24h": snap.operations_last_24h,
            "active_agents": snap.active_agents,
            "total_memories": snap.total_memories_stored,
        },
        "timestamp": snap.timestamp,
    }

@app.get("/health", tags=["Discovery"])
def health():
    return {"status": "ok", "liveness": True, "version": "3.1.0",
            "llm_backend": LLM_BACKEND, "timestamp": datetime.now().isoformat(),
            "manifest": "GET /.well-known/agent.json", "metrics": "GET /metrics",
            "mcp": "python mcp_server.py"}

@app.get("/", tags=["Info"])
def root():
    return {"name": "Agent Memory", "version": "3.1.0",
            "discovery": {"manifest": "GET /.well-known/agent.json", "metrics": "GET /metrics", "mcp": "python mcp_server.py"},
            "api": {"interact": "POST /interact", "remember": "POST /memory/remember",
                    "query": "POST /memory/query", "context": "GET /memory/context/{agent_id}",
                    "stats": "GET /memory/stats/{agent_id}", "docs": "GET /docs"}}

@app.post("/interact", tags=["Agent"])
async def interact(req: InteractRequest):
    t0 = time.time()
    try:
        agent = get_agent(req.agent_id, req.agent_type)
        explicit = Importance(req.explicit_importance) if req.explicit_importance else None
        with metrics.track("interact", agent_id=req.agent_id):
            response = agent.interact(session_id=req.session_id, user_input=req.user_input, explicit_importance=explicit)
        return {"agent_id": response.agent_id, "session_id": response.session_id,
                "response": response.text,
                "thought": {"importance": response.thought.importance_score, "memories_stored": len(response.thought.memories_stored), "duration_ms": response.thought.duration_ms},
                "memory_stats": response.memory_stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        metrics.record("interact", req.agent_id, False, (time.time()-t0)*1000, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/remember", tags=["Memory"])
async def remember(req: RememberRequest):
    try:
        agent = get_agent(req.agent_id)
        mt = MemoryType(req.memory_type) if req.memory_type else MemoryType.LONG_TERM
        with metrics.track("store", agent_id=req.agent_id):
            mem = agent.memory.store(mt, req.fact, importance=Importance.CRITICAL, tags=["explicit"])
        return {"status": "stored", "memory_id": mem.id, "type": mem.type.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/query", tags=["Memory"])
async def query_memory(req: MemoryQueryRequest):
    try:
        agent = get_agent(req.agent_id)
        mt = MemoryType(req.memory_type) if req.memory_type else None
        with metrics.track("retrieve", agent_id=req.agent_id):
            memories = agent.memory.retrieve(memory_type=mt, limit=req.limit, query=req.query)
        return {"agent_id": req.agent_id, "count": len(memories), "memories": [m.to_dict() for m in memories]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/consolidate", tags=["Memory"])
async def consolidate(req: ConsolidateRequest):
    try:
        agent = get_agent(req.agent_id)
        with metrics.track("consolidate", agent_id=req.agent_id):
            new_mems = agent.memory.consolidate()
        return {"agent_id": req.agent_id, "consolidated": len(new_mems)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats/{agent_id}", tags=["Memory"])
async def memory_stats(agent_id: str):
    return get_agent(agent_id).stats()

@app.get("/memory/context/{agent_id}", tags=["Memory"])
async def memory_context(agent_id: str, limit_per_layer: int = 3):
    try:
        agent = get_agent(agent_id)
        ctx = agent.memory.retrieve_context(limit_per_layer=limit_per_layer)
        return {"agent_id": agent_id, "context_text": agent.memory.format_context_for_prompt(limit_per_layer),
                "layers": {layer: [m.to_dict() for m in mems] for layer, mems in ctx.items()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/session/{agent_id}", tags=["Memory"])
async def clear_session(agent_id: str):
    get_agent(agent_id).new_session()
    return {"status": "cleared", "long_term_preserved": True}

@app.delete("/memory/agent/{agent_id}", tags=["Memory"])
async def clear_agent(agent_id: str):
    agent = get_agent(agent_id)
    agent.memory.storage.delete(agent_id)
    if agent_id in _agents:
        del _agents[agent_id]
    return {"status": "deleted", "agent_id": agent_id}

@app.get("/agents", tags=["Info"])
def list_agents():
    storage = LocalStorage(DATA_DIR)
    result = []
    for aid in storage.list_agents():
        s = get_agent(aid).stats()
        result.append({"agent_id": aid, "total_memories": s["total_memories"], "memory_pressure": s["memory_pressure"]})
    return {"agents": result, "count": len(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
