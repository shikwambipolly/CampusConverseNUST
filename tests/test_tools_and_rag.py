"""
tests/test_tools_and_rag.py

Unit and integration tests for:
- src.tools: Deterministic database tool calling (prerequisites, calendar, contacts)
- src.rag: Hybrid BM25 + TF-IDF RAG retrieval with source citations
"""

import pytest
from src.tools import (
    extract_module_code,
    extract_semester,
    lookup_prerequisite,
    lookup_calendar_date,
    lookup_contact_info
)
from src.rag import get_rag_retriever, search_policy


class TestDatabaseTools:
    def test_extract_module_code(self):
        assert extract_module_code("Can I take DSA611S in semester 2?") == "DSA611S"
        assert extract_module_code("Prerequisites for SWE711S please") == "SWE711S"
        assert extract_module_code("When is the semester?") is None

    def test_extract_semester(self):
        assert extract_semester("When do lectures start for Semester 1?") == 1
        assert extract_semester("Exam dates in s2?") == 2
        assert extract_semester("Hello there") is None

    def test_lookup_prerequisite_found(self):
        res = lookup_prerequisite("DSA612S")
        assert res["found"] is True
        assert res["module_code"] == "DSA612S"
        assert "prerequisites" in res
        assert "Data Structures" in res["module_name"]
        assert "PRG510S" in res["prerequisites"]

    def test_lookup_prerequisite_not_found(self):
        res = lookup_prerequisite("FAKE999S")
        assert res["found"] is False
        assert "not found" in res["message"].lower()

    def test_lookup_calendar_date(self):
        res = lookup_calendar_date("When do lectures commence for Semester 1?", semester=1)
        assert res["found"] is True
        assert len(res["events"]) > 0
        assert "commence" in res["message"].lower() or "lecture" in res["message"].lower()

    def test_lookup_contact_info_regional(self):
        res = lookup_contact_info("Katima Mulilo regional centre")
        assert res["found"] is True
        assert len(res["contacts"]) > 0
        assert any("katima" in c["department"].lower() or "katima" in c["role"].lower() for c in res["contacts"])

    def test_lookup_contact_info_library(self):
        res = lookup_contact_info("NUST Main Library contact number")
        assert res["found"] is True
        assert any("library" in c["department"].lower() for c in res["contacts"])


class TestHybridRAG:
    @classmethod
    def setup_class(cls):
        cls.retriever = get_rag_retriever()

    def test_corpus_loaded(self):
        assert len(self.retriever.chunks) >= 2300

    def test_search_attendance_policy(self):
        res = search_policy("What is the minimum attendance requirement to write exams?")
        assert res["found"] is True
        assert len(res["chunks"]) == 3
        # Check source citation format
        assert "Source:" in res["formatted_response"]
        assert "2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf" in res["formatted_response"]

    def test_search_hostel_rules(self):
        res = search_policy("hostel residence curfew and visitor rules")
        assert res["found"] is True
        assert len(res["chunks"]) == 3
        assert any("hostel" in c["text"].lower() or "residence" in c["text"].lower() for c in res["chunks"])
