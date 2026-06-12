from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv, set_key, unset_key
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mahoraga.adapter.adapter_manager import AdapterManager
from mahoraga.critic.critic_agent import CriticAgent
from mahoraga.orchestrator.maestro_hook import MaestroHook
from mahoraga.trigger.trigger_evaluator import TriggerEvaluator

load_dotenv()  # pick up any existing .env on startup

app = FastAPI(title="Project Mahoraga", version="0.1.0")

_hook = MaestroHook()
_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
_ENV_PATH   = Path(__file__).parent.parent / ".env"


# ---------- request models ----------

class EvaluateRequest(BaseModel):
    claim: str
    ai_response: str
    user_flagged: bool = False


class OpenCaseRequest(BaseModel):
    trigger_data: dict


class CloseCaseRequest(BaseModel):
    case_id: str
    resolution: str


class SettingsRequest(BaseModel):
    groq_api_key: str | None = None
    tavily_api_key: str | None = None


# ---------- dashboard ----------

@app.get("/")
@app.get("/dashboard")
def dashboard():
    return FileResponse(_DASHBOARD)


# ---------- settings ----------

@app.get("/settings")
def get_settings():
    return {
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
    }


@app.post("/settings")
def save_settings(req: SettingsRequest):
    env_file = str(_ENV_PATH)
    if req.groq_api_key is not None:
        key = req.groq_api_key.strip()
        if key:
            set_key(env_file, "GROQ_API_KEY", key)
            os.environ["GROQ_API_KEY"] = key
        else:
            unset_key(env_file, "GROQ_API_KEY")
            os.environ.pop("GROQ_API_KEY", None)
    if req.tavily_api_key is not None:
        key = req.tavily_api_key.strip()
        if key:
            set_key(env_file, "TAVILY_API_KEY", key)
            os.environ["TAVILY_API_KEY"] = key
        else:
            unset_key(env_file, "TAVILY_API_KEY")
            os.environ.pop("TAVILY_API_KEY", None)
    return {
        "status": "ok",
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
    }


# ---------- pipeline ----------

@app.get("/health")
def health():
    return {"status": "ok", "service": "project-mahoraga"}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    critic_result = CriticAgent().evaluate(req.claim)

    trigger = TriggerEvaluator().should_trigger(
        response=req.ai_response,
        confidence=critic_result["confidence"],
        user_flagged=req.user_flagged,
        critic_verdict=critic_result["verdict"],
    )

    adapter_result = {"status": "skipped"}
    if trigger["triggered"]:
        adapter_result = AdapterManager().update(critic_result)

    return {
        "claim": req.claim,
        "verdict": critic_result["verdict"],
        "confidence": critic_result["confidence"],
        "correction": critic_result.get("correction"),
        "sources": critic_result.get("sources", [])[:3],
        "tavily_answer": critic_result.get("tavily_answer", ""),
        "scoring_method": critic_result.get("scoring_method"),
        "triggered": trigger["triggered"],
        "reason": trigger.get("reason"),
        "adapter": adapter_result,
    }


# ---------- cases ----------

@app.post("/case/open")
def open_case(req: OpenCaseRequest):
    return _hook.open_case(req.trigger_data)


@app.post("/case/close")
def close_case(req: CloseCaseRequest):
    try:
        return _hook.close_case(req.case_id, req.resolution)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Case '{req.case_id}' not found")


@app.get("/cases")
def cases():
    return list(_hook._cases.values())


# ---------- knowledge ----------

@app.get("/knowledge")
def knowledge():
    return AdapterManager().read_knowledge_base()


# ---------- entrypoint ----------

if __name__ == "__main__":
    uvicorn.run("mahoraga.api:app", host="0.0.0.0", port=8000, reload=False)
