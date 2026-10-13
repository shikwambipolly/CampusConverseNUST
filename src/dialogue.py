"""
src/dialogue.py

Multi-Turn Dialogue & Clarification Engine (FSM).
Adheres strictly to spec.md Section 3.6:
- Finite State Machine: IDLE -> AWAITING_CLARIFICATION -> IDLE | ESCALATED
- Dialogue state storage schema with session_id, turn_count, collected_slots, missing_slots, history
- Exceeded max turns (2) transition to ESCALATED
"""

import uuid
from typing import Any, Dict, List, Optional
from src.tools import extract_module_code, extract_semester


class DialogueSession:
    """Represents an active multi-turn conversational session"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.current_state: str = "IDLE"  # "IDLE", "AWAITING_CLARIFICATION", "ESCALATED"
        self.turn_count: int = 0
        self.pending_intent: Optional[str] = None
        self.collected_slots: Dict[str, Any] = {}
        self.missing_slots: List[str] = []
        self.history: List[Dict[str, str]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Serializes session state conforming to spec.md Section 3.6"""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "turn_count": self.turn_count,
            "pending_intent": self.pending_intent,
            "collected_slots": dict(self.collected_slots),
            "missing_slots": list(self.missing_slots),
            "history": list(self.history)
        }

    def add_turn(self, role: str, content: str):
        """Appends a turn to conversation history"""
        self.history.append({"role": role, "content": content})

    def request_clarification(self, intent: str, missing_slots: List[str], prompt: str) -> str:
        """Transitions FSM to AWAITING_CLARIFICATION and increments turn count"""
        self.current_state = "AWAITING_CLARIFICATION"
        self.pending_intent = intent
        self.missing_slots = missing_slots
        self.turn_count += 1
        self.add_turn("assistant", prompt)
        return prompt

    def update_slots_from_input(self, text: str) -> bool:
        """
        Attempts to resolve pending missing slots from the user's clarification payload.
        Returns True if all required slots are now satisfied, False otherwise.
        """
        if "semester" in self.missing_slots:
            sem = extract_semester(text)
            if sem is not None:
                self.collected_slots["semester"] = sem
                self.missing_slots.remove("semester")

        if "module_code" in self.missing_slots:
            code = extract_module_code(text)
            if code is not None:
                self.collected_slots["module_code"] = code
                self.missing_slots.remove("module_code")

        # Check if all slots resolved
        if not self.missing_slots:
            self.current_state = "IDLE"
            self.turn_count = 0
            return True

        # If turn count exceeds max turns limit (2), transition to ESCALATED
        self.turn_count += 1
        if self.turn_count > 2:
            self.current_state = "ESCALATED"
            return False

        return False

    def reset(self):
        """Resets the state back to IDLE"""
        self.current_state = "IDLE"
        self.turn_count = 0
        self.pending_intent = None
        self.collected_slots = {}
        self.missing_slots = []


class DialogueManager:
    """Manages active dialogue sessions in memory"""

    def __init__(self):
        self.sessions: Dict[str, DialogueSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> DialogueSession:
        if not session_id or session_id not in self.sessions:
            new_id = session_id or str(uuid.uuid4())
            session = DialogueSession(session_id=new_id)
            self.sessions[new_id] = session
            return session
        return self.sessions[session_id]


# Global singleton instance
_dialogue_manager = DialogueManager()


def get_dialogue_manager() -> DialogueManager:
    return _dialogue_manager
