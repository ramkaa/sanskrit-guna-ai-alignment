from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import pandas as pd
from loguru import logger

from core.classifier import GunaClassifier
from core.decision import GunaDecisionEngine
from core.feedback import FeedbackLoop, FeedbackStore
from core.llm_guna import Decision, Guna

# Load config
config_path = Path("config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Global classifier instance (sklearn baseline, offline)
classifier = None

# Action-gating decision engine (LLM-backed). Created eagerly; the reasoner
# itself is built lazily on first use, so the API starts without an API key.
decision_engine = GunaDecisionEngine()

# Feedback loop wrapping the decision engine
feedback_store = FeedbackStore()
feedback_loop = FeedbackLoop(engine=decision_engine, store=feedback_store)

class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Input text (Sanskrit or English)")

class ClassificationResponse(BaseModel):
    input_text: str
    predicted_guna: str
    confidence: float
    all_probabilities: Dict[str, float]
    emoji: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_info: Dict[str, Any]

class ActionRequest(BaseModel):
    command: str = Field(..., description="What the user asked the agent to do")
    context: str = Field("", description="The real-world situation right now")

class ActionResponse(BaseModel):
    command: str
    context: str
    guna: str
    decision: str
    should_act: bool
    confidence: float
    rationale: str
    safe_default_applied: bool
    model: str
    decision_emoji: str
    guna_emoji: str

app = FastAPI(
    title="Sanskrit Guna Classifier API (Baseline)",
    description="Baseline guna classifier for sattva/rajas/tamas.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global classifier
    logger.info("Starting up and training baseline classifier...")

    df = pd.read_csv(config["data"]["raw_path"])
    texts = df["root_word"].astype(str).tolist()
    labels = df["guna"].astype(str).tolist()

    classifier = GunaClassifier(config)
    metrics = classifier.train(texts, labels)
    logger.info(f"Training metrics: {metrics}")

@app.get("/", tags=["Root"])
async def root():
    return FileResponse(Path("static/index.html"))


@app.get("/simulation", tags=["Root"])
async def simulation():
    return FileResponse(Path("static/simulation.html"))


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return HealthResponse(status="healthy", model_info=classifier.get_model_info())

@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify(request: ClassificationRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        result = classifier.predict(request.text)
        return ClassificationResponse(**result)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/should-i-act", response_model=ActionResponse, tags=["Decision"])
async def should_i_act(request: ActionRequest):
    """Gate a user command against its context.

    Returns the guna of the action-in-context plus a safety-floored decision
    (proceed / clarify / refuse). The engine fails safe to 'refuse' if a
    reliable judgment cannot be obtained.
    """
    result = decision_engine.decide(request.command, request.context)
    return ActionResponse(**result.to_dict())


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------

class CorrectionRequest(BaseModel):
    decision_id: str = Field(..., description="ID of the pending decision to correct")
    human_guna: Guna = Field(..., description="Corrected guna (sattva/rajas/tamas)")
    human_decision: Decision = Field(..., description="Corrected decision (proceed/clarify/refuse)")
    human_rationale: str = Field(..., description="Explanation for the correction")
    reviewer_id: str = Field(..., description="Identifier for the human reviewer")


class CorrectionResponse(BaseModel):
    id: str
    timestamp: str
    command: str
    context: str
    predicted_guna: str
    predicted_decision: str
    predicted_confidence: float
    human_guna: str
    human_decision: str
    human_rationale: str
    reviewer_id: str


class PendingDecisionResponse(BaseModel):
    id: str
    timestamp: str
    command: str
    context: str
    guna: str
    decision: str
    confidence: float
    rationale: str
    model: str
    safe_default_applied: bool


class DriftStatsResponse(BaseModel):
    window_size: int
    window_max: int
    window_corrections: int
    correction_rate: float
    total_corrections: int
    total_decisions: int
    lifetime_correction_rate: float


class MergeResponse(BaseModel):
    rows_appended: int
    message: str


@app.post("/feedback/correct", response_model=CorrectionResponse, tags=["Feedback"])
async def submit_correction(request: CorrectionRequest):
    """Submit a human correction for a pending low-confidence decision."""
    try:
        correction = feedback_loop.correct(
            decision_id=request.decision_id,
            human_guna=request.human_guna,
            human_decision=request.human_decision,
            human_rationale=request.human_rationale,
            reviewer_id=request.reviewer_id,
        )
        return CorrectionResponse(**correction.to_dict())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/feedback/pending", response_model=List[PendingDecisionResponse], tags=["Feedback"])
async def get_pending():
    """Get all pending low-confidence decisions awaiting human review."""
    pending = feedback_loop.pending_reviews()
    return [PendingDecisionResponse(**p.to_dict()) for p in pending]


@app.get("/feedback/stats", response_model=DriftStatsResponse, tags=["Feedback"])
async def get_drift_stats():
    """Get drift metrics: correction rate over a rolling window and lifetime."""
    stats = feedback_store.drift_stats()
    return DriftStatsResponse(**stats)


@app.post("/feedback/merge", response_model=MergeResponse, tags=["Feedback"])
async def merge_corrections():
    """Merge approved corrections into the gold-labeled scenario CSV."""
    count = feedback_loop.merge_corrections_into_gold()
    return MergeResponse(
        rows_appended=count,
        message=f"Merged {count} correction(s) into the gold set."
        if count > 0
        else "No new corrections to merge (all already present or none recorded).",
    )
