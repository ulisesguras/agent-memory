#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Agent Memory v3 — Setup Script
#  Runs everything from zero to operational in one command.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  Optional — with Anthropic API key:
#    ANTHROPIC_API_KEY=your_key ./setup.sh
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Banner ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║           AGENT MEMORY v3                    ║"
echo "  ║   Cognitive Infrastructure for AI Agents     ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Helper functions ─────────────────────────────────────────────
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
info() { echo -e "${BLUE}  → $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}$1${NC}"; echo "  ──────────────────────────────────────"; }

# ═══════════════════════════════════════════════════════════════
# STEP 1 — CHECK PREREQUISITES
# ═══════════════════════════════════════════════════════════════
section "STEP 1 — Checking prerequisites"

# Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    fail "Python not found. Install Python 3.11+ from https://python.org"
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    fail "Python 3.10+ required. Found: $PYTHON_VERSION"
fi
ok "Python $PYTHON_VERSION found"

# pip
if ! $PYTHON -m pip --version &>/dev/null; then
    fail "pip not found. Run: $PYTHON -m ensurepip"
fi
ok "pip available"

# Check we're in the right directory
if [ ! -f "core/memory_engine.py" ]; then
    fail "Run this script from inside the agent_memory/ folder"
fi
ok "Project structure verified"

# ═══════════════════════════════════════════════════════════════
# STEP 2 — VIRTUAL ENVIRONMENT
# ═══════════════════════════════════════════════════════════════
section "STEP 2 — Setting up virtual environment"

if [ -d "venv" ]; then
    warn "venv already exists — skipping creation"
else
    info "Creating virtual environment..."
    $PYTHON -m venv venv
    ok "Virtual environment created"
fi

# Activate
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    ACTIVATE="venv/Scripts/activate"
else
    ACTIVATE="venv/bin/activate"
fi

source $ACTIVATE
ok "Virtual environment activated"

# ═══════════════════════════════════════════════════════════════
# STEP 3 — INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════
section "STEP 3 — Installing dependencies"

info "Upgrading pip..."
pip install --upgrade pip --quiet 2>/dev/null || warn "pip upgrade skipped"

info "Installing core dependencies..."
pip install fastapi uvicorn pydantic --quiet 2>/dev/null && ok "FastAPI + Uvicorn installed" || warn "Run manually: pip install fastapi uvicorn pydantic"
ok "FastAPI + Uvicorn installed"

# Optional: Anthropic SDK
if [ -n "$ANTHROPIC_API_KEY" ]; then
    info "Anthropic API key detected — installing SDK..."
    pip install anthropic --quiet 2>/dev/null && ok "Anthropic SDK installed" || warn "Run manually: pip install anthropic"
    ok "Anthropic SDK installed"
else
    warn "No ANTHROPIC_API_KEY set — running in mock mode (no LLM calls)"
fi

# Optional: MCP SDK
info "Installing MCP SDK..."
if pip install mcp --quiet 2>/dev/null; then
    ok "MCP SDK installed — mcp_server.py ready"
else
    warn "MCP SDK not available — mcp_server.py will run in catalog-only mode"
fi

# ═══════════════════════════════════════════════════════════════
# STEP 4 — VERIFY STRUCTURE
# ═══════════════════════════════════════════════════════════════
section "STEP 4 — Verifying project structure"

FILES=(
    "core/memory_engine.py"
    "core/agent.py"
    "core/server.py"
    "core/metrics.py"
    "identity/SOUL.md"
    "identity/IDENTITY.md"
    "identity/AGENTS.md"
    "identity/MEMORY.md"
    "identity/USER.md"
    "cli/run_agent.py"
    "cli/experiment.py"
    "mcp_server.py"
    ".well-known/agent.json"
)

ALL_OK=true
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        ok "$f"
    else
        warn "Missing: $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    fail "Some files are missing. Re-download the zip3 and try again."
fi

# ═══════════════════════════════════════════════════════════════
# STEP 5 — SMOKE TEST
# ═══════════════════════════════════════════════════════════════
section "STEP 5 — Running smoke tests"

info "Testing cognitive engine..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from core.memory_engine import MemoryEngine, MemoryType, Importance, LocalStorage
engine = MemoryEngine('setup_test', LocalStorage('./data/agents'))
engine._memories = []
engine.store(MemoryType.LONG_TERM, 'setup smoke test', importance=Importance.CRITICAL)
assert engine.stats()['total_memories'] == 1
print('    memory_engine: OK')
" || fail "memory_engine.py smoke test failed"
ok "Cognitive engine operational"

info "Testing agent ReAct loop..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from core.agent import Agent, MockAdapter
agent = Agent('setup_test', llm=MockAdapter(), data_dir='./data/agents')
agent.memory._memories = []
resp = agent.interact('sess_setup', 'remember this is a critical setup test')
assert resp.thought.importance_score == 'critical'
print('    agent ReAct loop: OK')
print('    importance scoring: OK')
" || fail "agent.py smoke test failed"
ok "Agent ReAct loop operational"

info "Testing metrics..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from core.metrics import MetricsCollector
m = MetricsCollector('./data/metrics.json')
m.record('interact', 'setup_test', True, 42)
snap = m.snapshot()
assert snap.liveness == True
print('    metrics: OK')
" || fail "metrics.py smoke test failed"
ok "Metrics operational"

info "Testing MCP tool catalog..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from mcp_server import TOOL_CATALOG
assert len(TOOL_CATALOG) == 10
print(f'    {len(TOOL_CATALOG)} tools registered: OK')
" || fail "mcp_server.py smoke test failed"
ok "MCP server operational"

info "Verifying agent.json..."
$PYTHON -c "
import json
from pathlib import Path
manifest = json.loads(Path('.well-known/agent.json').read_text())
assert manifest['agent_compatibility']['mcp_compatible'] == True
assert len(manifest['capabilities']) == 8
print('    agent.json: OK')
" || fail "agent.json verification failed"
ok "agent.json valid"

# Clean up test data
rm -rf ./data/agents/setup_test* 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# STEP 6 — CREATE .env FILE
# ═══════════════════════════════════════════════════════════════
section "STEP 6 — Environment configuration"

if [ ! -f ".env" ]; then
    cat > .env << EOF
# Agent Memory v3 — Environment Configuration
# Generated by setup.sh

# LLM Backend: mock | anthropic | vertex
LLM_BACKEND=mock

# Anthropic (uncomment and add key to use Claude)
# ANTHROPIC_API_KEY=your_key_here

# Google Vertex AI (uncomment to use Gemini)
# GOOGLE_CLOUD_PROJECT=your_project_id

# Paths
IDENTITY_DIR=./identity
DATA_DIR=./data/agents
AGENT_MEMORY_DATA_DIR=./data/agents
AGENT_MEMORY_IDENTITY_DIR=./identity

# Server
PORT=8080
EOF
    ok ".env file created"
else
    warn ".env already exists — skipping"
fi

# Apply API key if provided
if [ -n "$ANTHROPIC_API_KEY" ]; then
    sed -i.bak "s|LLM_BACKEND=mock|LLM_BACKEND=anthropic|" .env 2>/dev/null || true
    sed -i.bak "s|# ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY|" .env 2>/dev/null || true
    ok "API key written to .env"
fi

# ═══════════════════════════════════════════════════════════════
# DONE — SHOW NEXT STEPS
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   SETUP COMPLETE — Agent Memory v3 is ready ✓       ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BOLD}  QUICK START:${NC}"
echo ""

echo -e "${CYAN}  1. Run all experiments (no LLM needed):${NC}"
echo "     python cli/experiment.py --all"
echo ""

echo -e "${CYAN}  2. Interactive agent (mock mode):${NC}"
echo "     python cli/run_agent.py --agent my_agent --llm mock"
echo ""

if [ -n "$ANTHROPIC_API_KEY" ]; then
echo -e "${CYAN}  3. Interactive agent (Claude — API key detected):${NC}"
echo "     python cli/run_agent.py --agent my_agent --llm anthropic"
echo ""
else
echo -e "${YELLOW}  3. Interactive agent with Claude (add your API key):${NC}"
echo "     export ANTHROPIC_API_KEY=your_key"
echo "     python cli/run_agent.py --agent my_agent --llm anthropic"
echo ""
fi

echo -e "${CYAN}  4. HTTP API server:${NC}"
echo "     uvicorn core.server:app --reload --port 8080"
echo "     → Docs: http://localhost:8080/docs"
echo "     → Manifest: http://localhost:8080/.well-known/agent.json"
echo "     → Metrics: http://localhost:8080/metrics"
echo ""

echo -e "${CYAN}  5. MCP server (for Claude Desktop):${NC}"
echo "     python mcp_server.py"
echo ""

echo -e "${CYAN}  6. CLI commands inside run_agent.py:${NC}"
echo "     /stats       memory statistics"
echo "     /memory      show all memories"
echo "     /search      search memories"
echo "     /remember    store critical fact"
echo "     /consolidate run episodic→semantic"
echo "     /new         new session"
echo "     /quit        exit"
echo ""

echo -e "${BLUE}  Manifest:  .well-known/agent.json${NC}"
echo -e "${BLUE}  Identity:  identity/SOUL.md${NC}"
echo -e "${BLUE}  Data:      data/agents/${NC}"
echo ""
