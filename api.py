"""
api.py
======
Purpose:
    FastAPI service running on Cloud Run to serve the Chatbot frontend
    and external API requests.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_agent import CampusConverseAgent

app = FastAPI(title="CampusConverse API", version="1.0.0")

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once on startup
agent = CampusConverseAgent()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest_student"


class ChatResponse(BaseModel):
    reply: str
    status: str = "success"


@app.get("/health")
def health_check():
    """Liveness probe for Cloud Run."""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main endpoint for student interactions. Passes the query to the
    Agentic LLM which invokes MCP tools for Chroma & AlloyDB.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        reply_text = agent.answer_question(request.message)
        return ChatResponse(reply=reply_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
