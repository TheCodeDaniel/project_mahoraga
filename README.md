# Project Mahoraga

Project Mahoraga is an agentic AI adaptation system that learns from its own mistakes. When an AI produces a wrong response, Mahoraga opens a case, investigates it, verifies it against live sources, and updates itself — so the same mistake never happens twice. Like its namesake, it adapts to anything.

## Architecture Overview

Mahoraga is built from four internal components wired together by a FastAPI service:

- **Critic** (`mahoraga/critic/critic_agent.py`) — Searches DuckDuckGo for the claim, scores source relevance, and returns a verdict (`True`/`False`/`None`), confidence score, top sources, and an optional correction snippet.
- **Trigger** (`mahoraga/trigger/trigger_evaluator.py`) — Decides whether the adaptation pipeline should fire based on critic confidence, AI self-confidence heuristics (hedge/assertion words), and user flagging.
- **Adapter** (`mahoraga/adapter/adapter_manager.py`) — Persists confirmed corrections to a local `knowledge_base.json` for downstream fine-tuning.
- **Orchestrator** (`mahoraga/orchestrator/maestro_hook.py`) — Opens and closes investigation cases (local simulation; UiPath RPA integration pending Lab access).

### HTTP API (`mahoraga/api.py`) — served on port 8000

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/evaluate` | Run the full pipeline (Critic → Trigger → Adapter). Body: `{ "claim", "ai_response", "user_flagged" }`. Returns verdict, confidence, correction, top 3 sources, trigger status. |
| `POST` | `/case/open` | Open a Maestro case. Body: `{ "trigger_data": dict }`. Returns the full case object with a UUID case ID. |
| `POST` | `/case/close` | Close a Maestro case. Body: `{ "case_id", "resolution" }`. Returns the updated case object. |
| `GET`  | `/knowledge` | Return the full knowledge base as a JSON array. |
| `GET`  | `/health` | Liveness check. Returns `{ "status": "ok", "service": "project-mahoraga" }`. |

## Installation

TODO

## Usage

TODO
