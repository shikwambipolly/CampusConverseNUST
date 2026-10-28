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
from src.dialogue import get_dialogue_manager
from src.chatbot_agent import get_chatbot_agent

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


class ResetRequest(BaseModel):
    user_id: str = "guest_student"


@app.get("/health")
def health_check():
    """Liveness and stack status probe."""
    llm_info = get_chatbot_agent().get_active_llm_info()
    use_postgres = os.getenv("USE_ALLOYDB", "false").lower() == "true"
    use_remote_chroma = os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true"
    return {
        "status": "healthy",
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "database": "postgresql" if use_postgres else "sqlite",
        "vector_store": "chromadb-remote" if use_remote_chroma else "chromadb-embedded",
    }


@app.post("/api/reset")
def reset_session_endpoint(request: ResetRequest):
    """Clears multi-turn conversation memory for the specified user session."""
    manager = get_dialogue_manager()
    session = manager.get_or_create_session(request.user_id)
    session.reset()
    session.history.clear()
    return {"status": "cleared", "user_id": request.user_id}


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
