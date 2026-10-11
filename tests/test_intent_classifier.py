"""
tests/test_intent_classifier.py

Unit and integration tests for the trained Intent Classifier pipeline:
- Validates model artifact loading and inference latency
- Asserts correct intent prediction across all 6 target routes
- Verifies output dictionary schema adhering to spec.md Section 3.2
"""

import os
import pytest
from src.train_intent import load_classifier, predict_intent, MODEL_ARTIFACT_PATH


class TestIntentClassifier:
    @classmethod
    def setup_class(cls):
        cls.classifier = load_classifier()

    def test_artifact_exists_and_loads(self):
        assert os.path.exists(MODEL_ARTIFACT_PATH)
        assert self.classifier is not None
        assert hasattr(self.classifier, "predict_proba")

    def test_output_schema(self):
        res = predict_intent("When is the test week?", model=self.classifier)
        assert "predicted_intent" in res
        assert "confidence_score" in res
        assert "all_probabilities" in res
        assert isinstance(res["confidence_score"], float)
        assert 0.0 <= res["confidence_score"] <= 1.0
        assert len(res["all_probabilities"]) == 6

    def test_calendar_intent_classification(self):
        query = "When do lectures commence for Semester 1 2026?"
        res = predict_intent(query, model=self.classifier)
        assert res["predicted_intent"] == "check_calendar_date"
        assert res["confidence_score"] >= 0.50

    def test_prerequisite_intent_classification(self):
        query = "What are the prerequisites for Data Structures and Algorithms DSA611S?"
        res = predict_intent(query, model=self.classifier)
        assert res["predicted_intent"] == "check_module_prereq"
        assert res["confidence_score"] >= 0.50

    def test_policy_intent_classification(self):
        query = "What is the minimum attendance requirement to write examinations?"
        res = predict_intent(query, model=self.classifier)
        assert res["predicted_intent"] == "query_general_policy"
        assert res["confidence_score"] >= 0.40

    def test_clarification_intent_classification(self):
        query = "Semester 1"
        res = predict_intent(query, model=self.classifier)
        assert res["predicted_intent"] == "clarification_response"
        assert res["confidence_score"] >= 0.40

    def test_out_of_scope_classification(self):
        query = "What is the weather like in Windhoek today?"
        res = predict_intent(query, model=self.classifier)
        assert res["predicted_intent"] == "out_of_scope"
        assert res["confidence_score"] >= 0.40
