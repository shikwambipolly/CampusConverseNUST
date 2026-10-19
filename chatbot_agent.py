"""
chatbot_agent.py
================
Purpose:
    Chatbot agent that communicates with the MCP server to dynamically
    fetch context from Chroma and AlloyDB, then orchestrates an LLM response.
    Can be called via FastAPI or a chat interface.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Automatically load environment variables from .env file (e.g., GEMINI_API_KEY)
load_dotenv()

# Using Google Gemini (or OpenAI / Anthropic) for the LLM
import google.generativeai as genai
from semantic_layer import SemanticLayer

# Configure Gemini API key from environment variable (populated by .env or OS)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class CampusConverseAgent:
    """
    Agentic LLM coordinator connecting to tools for RAG (Chroma)
    and Tabular Data (AlloyDB).
    """

    def __init__(self):
        # We can directly wire the semantic layer tools or connect via MCP client
        self.semantic = SemanticLayer()

        # Define Tool Definitions compatible with Gemini Function Calling
        self.tools = [
            self.search_regulations,
            self.lookup_module_prereq,
            self.lookup_calendar,
            self.lookup_fees_table
        ]

        # System prompt giving strict instructions to prevent hallucination
        self.system_instruction = (
            "You are CampusConverse, an official AI academic assistant for Namibia University "
            "of Science and Technology (NUST). You provide accurate answers about institutional policies, "
            "admission requirements, course prerequisites, calendar dates, and fees.\n"
            "RULES:\n"
            "1. ALWAYS call tools when answering factual, policy, prerequisite, or date inquiries.\n"
            "2. Do NOT guess module prerequisites or exam dates; use the exact database lookups.\n"
            "3. If an answer comes from policies, cite the source document, section, and page number.\n"
            "4. Be concise, polite, and helpful."
        )

        # Initialize the Generative Model with tools enabled
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=self.tools,
            system_instruction=self.system_instruction
        )

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
        """Lookup module details and prerequisites in AlloyDB."""
        data = self.semantic.get_module_prerequisites(module_code)
        if not data:
            return f"No course found for code {module_code}."
        return json.dumps(data)

    def lookup_calendar(self, event_keyword: str) -> str:
        """Lookup dates from the academic calendar in AlloyDB."""
        events = self.semantic.search_academic_calendar(event_keyword)
        return json.dumps(events)

    def lookup_fees_table(self, fee_category: str) -> str:
        """Lookup tuition, hostel, and administrative fees in AlloyDB."""
        fees = self.semantic.get_fee_breakdown(fee_category)
        return json.dumps(fees)

    # -------------------------------------------------------------------------
    # Chat Execution Loop
    # -------------------------------------------------------------------------
    def answer_question(self, user_message: str, history: List[Dict[str, str]] = None) -> str:
        """
        Sends the user's message to the LLM. If the LLM requests a tool call,
        Gemini's automatic function calling resolves it and returns the final response.
        """
        # Start a multi-turn chat session with automatic function calling enabled
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        
        # Send user utterance
        response = chat.send_message(user_message)
        
        # Return generated text
        return response.text


# Quick testing block
if __name__ == "__main__":
    agent = CampusConverseAgent()
    test_query = "What is the prerequisite for PRG510S and what is the minimum lecture attendance required?"
    print(f"User: {test_query}\n")
    bot_reply = agent.answer_question(test_query)
    print(f"CampusConverse: {bot_reply}")
