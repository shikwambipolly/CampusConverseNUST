"""
src/api.py
==========
Purpose:
    FastAPI service running on Cloud Run to serve the Chatbot frontend
    and external API requests.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.orchestrator import process_query

app = FastAPI(title="CampusConverse API", version="2.0.0")

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest_student"


class ChatResponse(BaseModel):
    reply: str
    status: str = "success"
    intent: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None


@app.get("/health")
def health_check():
    """Liveness probe for Cloud Run."""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Unified endpoint for student interactions. Routes queries through:
    1. Preprocessing, Language Detection (en/af/kj), and Safety Guardrails
    2. Multi-turn Dialogue FSM & Calibrated LinearSVC Intent Router
    3. Deterministic SQLite/AlloyDB Lookups + 3-Way Hybrid RAG (Chroma + BM25 + TF-IDF)
    4. Grounded Gemini LLM Synthesis (when GEMINI_API_KEY is configured)
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = process_query(request.message, session_id=request.user_id)
        return ChatResponse(
            reply=result["response"],
            status=result["status"],
            intent=result["intent"],
            confidence=result["confidence"],
            language=result["language"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port)
