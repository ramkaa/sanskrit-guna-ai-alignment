from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
import yaml
import pandas as pd
from loguru import logger

from core.classifier import GunaClassifier

# Load config
config_path = Path("config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Global classifier instance
classifier = None

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
    return {"message": "Sanskrit Guna Classifier API (baseline)", "docs": "/docs"}

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
from pathlib import Path
import yaml

CONFIG_PATH = Path("config/config.yaml")

if not CONFIG_PATH.exists():
    raise RuntimeError(f"Config file not found at {CONFIG_PATH}")

with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)

if not isinstance(config, dict):
    raise RuntimeError(f"Config could not be loaded as dict. Got: {config!r}")
