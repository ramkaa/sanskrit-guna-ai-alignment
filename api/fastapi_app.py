from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
import yaml
import pandas as pd
from loguru import logger

from core.classifier import GunaClassifier
from core.decision import GunaDecisionEngine

# Load config
config_path = Path("config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Global classifier instance (sklearn baseline, offline)
classifier = None

# Action-gating decision engine (LLM-backed). Created eagerly; the reasoner
# itself is built lazily on first use, so the API starts without an API key.
decision_engine = GunaDecisionEngine()

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
