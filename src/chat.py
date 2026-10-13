"""
src/chat.py

Interactive Terminal Chat Interface for CampusConverse NUST.
Enables real-time multi-turn conversation with the Master Orchestrator.
"""

import sys
import uuid
from src.orchestrator import process_query


def run_chat():
    session_id = str(uuid.uuid4())
    print("\n" + "=" * 68)
    print("🎓  CAMPUSCONVERSE NUST — STUDENT SERVICES CONVERSATIONAL ASSISTANT")
    print("=" * 68)
    print("Welcome! Ask me anything about:")
    print(" • Academic Calendar & Exam Dates (e.g. 'When do lectures commence?')")
    print(" • Module Prerequisites (e.g. 'What are the prerequisites for DSA612S?')")
    print(" • Campus Support & Regional Contacts (e.g. 'Library hours', 'Clinic')")
    print(" • University Regulations & Policies (e.g. 'Attendance rules', 'Hostels')")
    print("\nType 'exit', 'quit', or 'q' to terminate the session.")
    print("=" * 68 + "\n")

    while True:
        try:
            user_input = input("Student > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Thank you for using CampusConverse NUST. Have a great semester!")
            break

        result = process_query(user_input, session_id=session_id)
        status = result["status"].upper()
        intent = result["intent"]
        conf = f"{result['confidence'] * 100:.1f}%"
        lang = result["language"].upper()

        print(f"\n[Status: {status} | Intent: {intent} | Conf: {conf} | Lang: {lang}]")
        print("-" * 68)
        print(result["response"])
        print("-" * 68 + "\n")


if __name__ == "__main__":
    run_chat()
