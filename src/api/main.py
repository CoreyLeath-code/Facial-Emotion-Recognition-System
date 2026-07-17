
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.pipeline.inference import EmotionPipeline
from src.src.rag.context_retriever import EmotionContextRetriever
from src.src.llm_explainer.explain import EmotionLLMExplainer
from src.api.validation import validate_image_upload
from src.src.config.settings import settings


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

pipeline: Optional[EmotionPipeline] = None
rag: Optional[EmotionContextRetriever] = None
llm: Optional[EmotionLLMExplainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy components once at startup."""
    global pipeline, rag, llm
    pipeline = EmotionPipeline()
    rag = EmotionContextRetriever()
    llm = EmotionLLMExplainer()
    yield
    # cleanup (if needed) goes here


# FastAPI App
app = FastAPI(
    title="Facial Emotion Recognition + LLM + RAG API",
    description="A production-ready API for emotion classification with LLM explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS Support
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# Request Model
class EmotionRequest(BaseModel):
    emotions: list[str]


@app.get("/")
def home():
    return {"status": "online", "message": "Emotion AI API Running Successfully!"}


@app.get("/health/live")
def health():
    """
    Health check endpoint.
    Returns model status so orchestrators can determine readiness.
    """
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    model_ready = pipeline is not None and pipeline.model is not None
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready", "model_loaded": True}


@app.post("/predict")
async def predict_emotion(image: UploadFile = File(...)):
    """
    Endpoint: Perform CNN-based emotion prediction.
    """
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    payload = await image.read(settings.MAX_UPLOAD_BYTES + 1)
    try:
        decoded = validate_image_upload(payload, image.content_type, settings.MAX_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    predictions = pipeline.predict_image(decoded)
    return {"predictions": predictions}


@app.post("/rag")
async def retrieve_context(request: EmotionRequest):
    """
    Endpoint: Retrieve psychology-backed context via RAG.
    """
    context = rag.retrieve(request.emotions)
    return {"context": context}


@app.post("/explain")
async def explain_emotions(request: EmotionRequest):
    """
    Endpoint: Generate LLM explanation using predictions + RAG context.
    """
    # Step 1: Retrieve psychology context
    rag_context = rag.retrieve(request.emotions)

    # Step 2: Generate LLM explanation
    explanation = llm.explain_emotions(emotions=request.emotions, rag_context=rag_context)

    return {"emotions": request.emotions, "context_used": rag_context, "explanation": explanation}


@app.post("/full-analysis")
async def full_analysis(image: UploadFile = File(...)):
    """
    Full pipeline:
    1. Predict emotion(s)
    2. Retrieve RAG psychology context
    3. Generate LLM explanation
    """
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    payload = await image.read(settings.MAX_UPLOAD_BYTES + 1)
    try:
        decoded = validate_image_upload(payload, image.content_type, settings.MAX_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    predictions = pipeline.predict_image(decoded)

    rag_context = rag.retrieve(predictions)

    explanation = llm.explain_emotions(emotions=predictions, rag_context=rag_context)

    return {"predictions": predictions, "context_used": rag_context, "explanation": explanation}


# Run the server (local development)
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
