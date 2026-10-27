"""
src/chatbot_agent.py
====================
Purpose:
    Chatbot agent that communicates with the MCP server / semantic layer to dynamically
    fetch context from Chroma and AlloyDB/SQLite, then orchestrates an LLM response.
    Can be called via FastAPI or a chat interface.
"""

import os
import json
import urllib.request
from typing import List, Dict, Any
from dotenv import load_dotenv

# Automatically load environment variables from .env file (e.g., GEMINI_API_KEY)
load_dotenv()

# Using Google Gemini or open-source Ollama for LLM synthesis
import google.generativeai as genai
from src.semantic_layer import get_semantic_layer

# Configure LLM settings from environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY, transport="rest")


class CampusConverseAgent:
    """
    Agentic LLM coordinator connecting to tools for RAG (Chroma + Hybrid)
    and Tabular Data (AlloyDB / SQLite).
    """

    def __init__(self):
        self.semantic = get_semantic_layer()

        # Define Tool Definitions compatible with Gemini Function Calling
        self.tools = [
            self.search_regulations,
            self.lookup_module_prereq,
            self.lookup_calendar,
            self.lookup_fees_table,
            self.lookup_programmes
        ]

        # System prompt giving strict instructions to prevent hallucination
        self.system_instruction = (
            "You are CampusConverse, an official AI academic assistant for Namibia University "
            "of Science and Technology (NUST). You provide accurate answers about institutional policies, "
            "admission requirements, course prerequisites, calendar dates, and fees.\n"
            "RULES:\n"
            "1. ALWAYS ground your answers strictly in the provided NUST database facts and policy excerpts.\n"
            "2. Do NOT guess module prerequisites, fees, or exam dates.\n"
            "3. When referencing policy excerpts, ALWAYS preserve the exact citation in the format "
            "[Source: <document> | Section: <section> | Page: <page>].\n"
            "4. Be concise, polite, and well-structured."
        )

        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    tools=self.tools,
                    system_instruction=self.system_instruction
                )
            except Exception:
                self.model = None

    # -------------------------------------------------------------------------
    # Tool Callable Methods (Mirroring MCP Tools)
    # -------------------------------------------------------------------------
    def search_regulations(self, query: str) -> str:
        """Search university policies and regulations in Chroma vector store."""
        results = self.semantic.search_policies_and_regulations(query=query)
        if not results:
            return "No matching policy text found."
        return json.dumps(results)

    def lookup_module_prereq(self, module_code: str) -> str:
        """Lookup module details and prerequisites in AlloyDB / SQLite."""
        data = self.semantic.get_module_prerequisites(module_code)
        if not data:
            return f"No course found for code {module_code}."
        return json.dumps(data)

    def lookup_calendar(self, event_keyword: str) -> str:
        """Lookup dates from the academic calendar in AlloyDB / SQLite."""
        events = self.semantic.search_academic_calendar(event_keyword)
        return json.dumps(events)

    def lookup_fees_table(self, fee_category: str) -> str:
        """Lookup tuition, hostel, banking, and administrative fees in AlloyDB / SQLite."""
        fees = self.semantic.get_fee_breakdown(fee_category)
        return json.dumps(fees)

    def lookup_programmes(self, keyword: str) -> str:
        """Lookup NUST degree/diploma programmes, NQF levels, and minimum admission points."""
        progs = self.semantic.search_programmes(keyword)
        return json.dumps(progs)

    # -------------------------------------------------------------------------
    # Grounded Synthesis & Chat Execution
    # -------------------------------------------------------------------------
    def get_active_llm_info(self) -> Dict[str, str]:
        """Return active LLM provider and model name for health/status reporting."""
        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()

        if provider == "offline":
            return {"provider": "offline-deterministic", "model": "hybrid-rag-direct"}
        if provider == "ollama" or (provider == "auto" and ollama_url and not api_key):
            return {
                "provider": "ollama",
                "model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
            }
        if provider in ("gemini", "auto") and api_key:
            return {
                "provider": "gemini",
                "model": os.getenv("GEMINI_MODEL", GEMINI_MODEL),
            }
        return {"provider": "offline-deterministic", "model": "hybrid-rag-direct"}

    def _synthesize_with_ollama(self, prompt: str) -> str:
        """Call open-source local/Docker Ollama container (/api/generate)."""
        base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).strip().rstrip("/")
        model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL).strip()
        payload = json.dumps({
            "model": model,
            "system": self.system_instruction,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
            return (data.get("response") or "").strip()

    def synthesize_with_context(self, user_message: str, retrieved_context: str) -> str:
        """
        Synthesizes a clear conversational answer from the deterministic & hybrid RAG
        context while preserving citations. Supports:
        - Open-source Ollama (LLM_PROVIDER=ollama)
        - Google Gemini API (LLM_PROVIDER=gemini or auto with GEMINI_API_KEY)
        - Offline deterministic mode (LLM_PROVIDER=offline or fallback)
        """
        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        if provider == "offline":
            return retrieved_context

        prompt = (
            f"Student Question: {user_message}\n\n"
            f"Retrieved Official NUST Context:\n{retrieved_context}\n\n"
            "Please provide a clear, helpful answer based strictly on the Retrieved Official NUST Context above. "
            "Keep all [Source: ... | Section: ... | Page: ...] citations intact at the end of your response."
        )

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()

        # 1. Open-Source Ollama Synthesis
        if provider == "ollama" or (provider == "auto" and ollama_url and not api_key):
            try:
                reply = self._synthesize_with_ollama(prompt)
                if reply:
                    return reply
            except Exception:
                pass

        # 2. Google Gemini API Synthesis
        if provider in ("gemini", "auto") and api_key:
            try:
                genai.configure(api_key=api_key, transport="rest")
                synth_model = genai.GenerativeModel(
                    model_name=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                    system_instruction=self.system_instruction
                )
                resp = synth_model.generate_content(prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        return retrieved_context

    def answer_question(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """
        Sends the user's message through the unified orchestrator pipeline
        (which combines safety, multilingual detection, SQLite/AlloyDB tools,
        3-way Hybrid RAG, and Gemini synthesis).
        """
        from src.orchestrator import process_query
        result = process_query(user_message)
        return result["response"]


_agent_instance = None


def get_chatbot_agent() -> CampusConverseAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CampusConverseAgent()
    return _agent_instance


# Quick testing block
if __name__ == "__main__":
    agent = get_chatbot_agent()
    test_query = "What is the prerequisite for PRG510S and what is the minimum lecture attendance required?"
    print(f"User: {test_query}\n")
    bot_reply = agent.answer_question(test_query)
    print(f"CampusConverse: {bot_reply}")
