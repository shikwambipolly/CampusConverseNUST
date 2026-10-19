"""
src/orchestrator.py

Master Conversational Orchestrator for CampusConverse NUST.
Adheres strictly to spec.md Section 6 (System Execution Lifecycle):
1. Ingestion: Normalization, Language Identification, and Defensive Safety Guardrail.
2. State Check: Dialogue FSM evaluation (AWAITING_CLARIFICATION slot filling).
3. Intent Routing: High-confidence classification (tau >= 0.65) or Fallback Escalation.
4. Tool / RAG Execution: Deterministic SQLite lookups or Hybrid BM25/TF-IDF Policy RAG.
5. Multilingual Ingestion & Citation Grounding: Verified output with source citations.
"""

from typing import Any, Dict, Optional
from src.chatbot_agent import get_chatbot_agent
from src.dialogue import DialogueSession, get_dialogue_manager
from src.preprocessing import preprocess_input
from src.rag import search_policy
from src.tools import (
    extract_module_code,
    extract_semester,
    lookup_calendar_date,
    lookup_contact_info,
    lookup_prerequisite,
    lookup_structured_facts
)
from src.train_intent import predict_intent

# Official Fallback Escalation target conforming to spec.md Section 3.8
DEFAULT_ESCALATION = {
    "status": "fallback",
    "reason": "unresolved_or_out_of_scope",
    "message": (
        "I cannot find authoritative information regarding this request in the current NUST handbooks. "
        "Your inquiry has been escalated to campus student administration."
    ),
    "escalation_target": {
        "entity": "Office of the Registrar / Student Services",
        "email": "admissions@nust.na",
        "telephone": "+264 61 207 9111",
        "location": "Administration Building, Ground Floor, Windhoek Main Campus"
    }
}

MULTILINGUAL_FRAMING = {
    "af": "Dankie vir jou navraag. Hier is die amptelike NUST inligting:\n\n",
    "kj": "Tangi komapulo oye. Ouyelele wo-NUST owo mbuno:\n\n",
    "en": ""
}

ACADEMIC_DOMAIN_KEYWORDS = (
    "nust", "exam", "examination", "test", "lecture", "semester", "course", "module",
    "class", "student", "campus", "faculty", "department", "library", "clinic",
    "hostel", "residence", "fee", "tuition", "bursary", "loan", "nsfaf", "register",
    "registration", "apply", "application", "admission", "programme", "program",
    "degree", "diploma", "credit", "mark", "grade", "pass", "fail", "supplementary",
    "appeal", "remark", "plagiarism", "attendance", "timetable", "graduation", "rule",
    "policy", "regulation", "handbook", "yearbook", "exempt", "exemption", "wil",
    "internship", "scholarship", "society", "club", "eksamen", "klas", "fooie", "registrasie"
)


def _has_academic_domain_signal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ACADEMIC_DOMAIN_KEYWORDS) or bool(extract_module_code(text))


def process_query(user_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the end-to-end request lifecycle for an incoming student query.
    Returns:
        {
            "session_id": str,
            "status": "success" | "clarification" | "fallback" | "blocked",
            "intent": str,
            "confidence": float,
            "language": str,
            "response": str,
            "metadata": Dict[str, Any]
        }
    """
    manager = get_dialogue_manager()
    session: DialogueSession = manager.get_or_create_session(session_id)

    # 1. Ingestion: Normalization, Safety Check, and Language Detection
    pre = preprocess_input(user_text)

    # Fast Short-Circuit on safety violation
    if not pre["is_safe"]:
        session.add_turn("user", user_text)
        session.add_turn("assistant", pre["terminal_response"])
        return {
            "session_id": session.session_id,
            "status": "blocked",
            "intent": "safety_violation",
            "confidence": 1.0,
            "language": pre["language"],
            "response": pre["terminal_response"],
            "metadata": {"flagged_categories": pre["flagged_categories"]}
        }

    clean_text = pre["normalized_text"]
    lang = pre["language"]
    session.add_turn("user", clean_text)

    # 2. State Check: Evaluate if session has a pending clarification slot
    if session.current_state == "AWAITING_CLARIFICATION":
        all_resolved = session.update_slots_from_input(clean_text)

        if session.current_state == "ESCALATED":
            resp_text = (
                f"{DEFAULT_ESCALATION['message']}\n\n"
                f"Contact Point: {DEFAULT_ESCALATION['escalation_target']['entity']}\n"
                f"Email: {DEFAULT_ESCALATION['escalation_target']['email']} | "
                f"Tel: {DEFAULT_ESCALATION['escalation_target']['telephone']}"
            )
            session.add_turn("assistant", resp_text)
            session.reset()
            return {
                "session_id": session.session_id,
                "status": "fallback",
                "intent": "escalated_unresolved",
                "confidence": 1.0,
                "language": lang,
                "response": resp_text,
                "metadata": DEFAULT_ESCALATION
            }

        if all_resolved:
            # Re-execute pending intent with filled slots
            return _execute_intent(
                intent=session.pending_intent,
                clean_text=clean_text,
                session=session,
                lang=lang,
                confidence=1.0
            )
        else:
            # Re-prompt for remaining missing slots
            missing = session.missing_slots[0]
            if missing == "module_code":
                prompt = "Please specify the NUST course code (e.g., DSA612S or SWE711S) to check prerequisites."
            elif missing == "semester":
                prompt = "Which semester are you inquiring about: Semester 1 or Semester 2?"
            else:
                prompt = f"Please provide the {missing} to continue."

            session.add_turn("assistant", prompt)
            return {
                "session_id": session.session_id,
                "status": "clarification",
                "intent": session.pending_intent,
                "confidence": 1.0,
                "language": lang,
                "response": prompt,
                "metadata": {"missing_slots": session.missing_slots}
            }

    # 3. Intent Classification
    intent_res = predict_intent(clean_text)
    intent = intent_res["predicted_intent"]
    confidence = intent_res["confidence_score"]

    # 4. Smart Fallback & Retrieval Rescue:
    # Instead of blindly escalating when confidence < 0.65, check if structured DB tables
    # or 3-way Hybrid RAG have relevant NUST answers first (while still escalating non-academic OOD queries).
    if confidence < 0.65 or intent == "out_of_scope":
        structured_hit = lookup_structured_facts(clean_text)
        has_domain_signal = _has_academic_domain_signal(clean_text)
        rag_hit = search_policy(clean_text, top_k=3) if has_domain_signal else {"is_relevant": False}

        if structured_hit.get("found") or (has_domain_signal and rag_hit.get("is_relevant")):
            rescued_intent = intent if intent not in ("out_of_scope", "clarification_response") else "query_general_policy"
            return _execute_intent(
                intent=rescued_intent,
                clean_text=clean_text,
                session=session,
                lang=lang,
                confidence=max(confidence, 0.65)
            )

        framing = MULTILINGUAL_FRAMING.get(lang, "")
        resp_text = (
            framing +
            f"{DEFAULT_ESCALATION['message']}\n\n"
            f"Escalation Point: {DEFAULT_ESCALATION['escalation_target']['entity']}\n"
            f"Email: {DEFAULT_ESCALATION['escalation_target']['email']} | "
            f"Tel: {DEFAULT_ESCALATION['escalation_target']['telephone']}\n"
            f"Location: {DEFAULT_ESCALATION['escalation_target']['location']}"
        )
        session.add_turn("assistant", resp_text)
        return {
            "session_id": session.session_id,
            "status": "fallback",
            "intent": intent,
            "confidence": confidence,
            "language": lang,
            "response": resp_text,
            "metadata": DEFAULT_ESCALATION
        }

    # 5. Route Intent
    return _execute_intent(
        intent=intent,
        clean_text=clean_text,
        session=session,
        lang=lang,
        confidence=confidence
    )


def _execute_intent(
    intent: str,
    clean_text: str,
    session: DialogueSession,
    lang: str,
    confidence: float
) -> Dict[str, Any]:
    """Routes validated query to the target tool, 3-way Hybrid RAG, and optional Gemini synthesis."""
    framing = MULTILINGUAL_FRAMING.get(lang, "")
    metadata = {}
    agent = get_chatbot_agent()

    if intent == "check_module_prereq":
        code = session.collected_slots.get("module_code") or extract_module_code(clean_text)
        if not code:
            # Need slot clarification
            prompt = "Which module code would you like to check prerequisites for (e.g., DSA612S or SWE711S)?"
            session.request_clarification(intent="check_module_prereq", missing_slots=["module_code"], prompt=prompt)
            return {
                "session_id": session.session_id,
                "status": "clarification",
                "intent": intent,
                "confidence": confidence,
                "language": lang,
                "response": prompt,
                "metadata": {"missing_slots": ["module_code"]}
            }

        tool_result = lookup_prerequisite(code)
        synthesized = agent.synthesize_with_context(clean_text, tool_result["message"])
        response_text = framing + synthesized
        metadata = tool_result
        session.reset()

    elif intent == "check_calendar_date":
        sem = session.collected_slots.get("semester") or extract_semester(clean_text)
        tool_result = lookup_calendar_date(clean_text, semester=sem)
        synthesized = agent.synthesize_with_context(clean_text, tool_result["message"])
        response_text = framing + synthesized
        metadata = tool_result
        session.reset()

    elif intent == "escalate_human_contact":
        tool_result = lookup_contact_info(clean_text)
        synthesized = agent.synthesize_with_context(clean_text, tool_result["message"])
        response_text = framing + synthesized
        metadata = tool_result
        session.reset()

    elif intent == "query_general_policy":
        structured_res = lookup_structured_facts(clean_text)
        rag_result = search_policy(clean_text, top_k=3)

        context_parts = []
        if structured_res.get("found"):
            context_parts.append(structured_res["message"])
        if rag_result.get("found"):
            context_parts.append(rag_result["formatted_response"])

        combined_context = "\n\n".join(context_parts) if context_parts else rag_result["formatted_response"]
        synthesized = agent.synthesize_with_context(clean_text, combined_context)
        response_text = framing + synthesized
        metadata = {
            "chunks_retrieved": len(rag_result["chunks"]),
            "structured_category": structured_res.get("category")
        }
        session.reset()

    elif intent == "clarification_response":
        # Neutral conversational greeting / short reply
        response_text = framing + (
            "How can I assist you today? You can ask about NUST academic dates, "
            "module prerequisites, tuition fees, hostel rules, or campus contacts."
        )
        session.reset()

    else:
        # Fallback
        response_text = framing + DEFAULT_ESCALATION["message"]
        metadata = DEFAULT_ESCALATION

    session.add_turn("assistant", response_text)

    return {
        "session_id": session.session_id,
        "status": "success",
        "intent": intent,
        "confidence": confidence,
        "language": lang,
        "response": response_text,
        "metadata": metadata
    }
