"""
src/train_intent.py

Trains, validates, evaluates, and exports the classical intent classification models
for the CampusConverse NUST routing subsystem.

Adheres strictly to spec.md Section 3.2 & Section 5:
- Model A1: Multinomial Logistic Regression with Word+Char TF-IDF representation.
- Model A2: Calibrated Linear Support Vector Classifier (LinearSVC).
- Evaluation: Precision, Recall, Macro F1, Weighted F1, Confusion Matrix, Latency.
- Artifact: models/intent_classifier.joblib
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features import build_feature_extractor
from src.preprocessing import normalize_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "intents")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "intent_classifier.joblib")


def load_dataset_split(split_name: str) -> Tuple[List[str], List[str]]:
    """Loads text and intent labels from the specified split file"""
    path = os.path.join(DATA_DIR, f"{split_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    labels = [r["intent"] for r in records]
    return texts, labels


def train_and_benchmark_models() -> Pipeline:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[*] Loading partitioned intent dataset splits...")
    X_train, y_train = load_dataset_split("train")
    X_val, y_val = load_dataset_split("val")
    X_test, y_test = load_dataset_split("test")

    print(f"    - Train set : {len(X_train)} samples")
    print(f"    - Val set   : {len(X_val)} samples")
    print(f"    - Test set  : {len(X_test)} samples")

    # Define candidate pipelines using Word+Char TF-IDF FeatureUnion
    candidates = {
        "LogisticRegression": Pipeline([
            ("feats", build_feature_extractor()),
            ("clf", LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "CalibratedLinearSVC": Pipeline([
            ("feats", build_feature_extractor()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.5, class_weight="balanced", random_state=42),
                method="sigmoid",
                cv=3
            ))
        ])
    }

    print("\n[*] Training and tuning candidate models on Validation Set...")
    val_results = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro")
        weighted_f1 = f1_score(y_val, y_val_pred, average="weighted")
        val_results[name] = {
            "pipeline": pipe,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
        print(f"    -> {name:20}: Val Macro F1 = {macro_f1:.4f} | Weighted F1 = {weighted_f1:.4f}")

    # Select champion model based on Macro F1
    champion_name = max(val_results, key=lambda k: val_results[k]["macro_f1"])
    champion_pipe = val_results[champion_name]["pipeline"]
    print(f"\n[+] Champion Model Selected: {champion_name} (Val Macro F1 = {val_results[champion_name]['macro_f1']:.4f})")

    # Final evaluation on held-out Test set
    print("\n" + "=" * 65)
    print(f"FINAL EVALUATION ON HELD-OUT TEST SET ({len(X_test)} samples)")
    print("=" * 65)

    start_time = time.perf_counter()
    y_test_pred = champion_pipe.predict(X_test)
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / len(X_test)) * 1000

    report = classification_report(y_test, y_test_pred, digits=4)
    print(report)

    # Confusion matrix
    classes = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    print("Confusion Matrix:")
    print(f"{'':25} " + " ".join([f"{c[:8]:>8}" for c in classes]))
    for idx, row in enumerate(cm):
        print(f"{classes[idx]:25} " + " ".join([f"{cnt:>8d}" for cnt in row]))

    print("-" * 65)
    print(f"Inference Latency: {avg_latency_ms:.3f} ms per query ({len(X_test)/total_time:.1f} queries/sec)")
    print("=" * 65)

    # Save winning model artifact
    joblib.dump(champion_pipe, MODEL_ARTIFACT_PATH)
    print(f"\n[+] Successfully saved trained model pipeline to: {MODEL_ARTIFACT_PATH}")

    return champion_pipe


def load_classifier() -> Pipeline:
    """Loads the trained model artifact from disk, or trains it if not present"""
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        return train_and_benchmark_models()
    return joblib.load(MODEL_ARTIFACT_PATH)


def predict_intent(text: str, model: Optional[Pipeline] = None) -> Dict:
    """
    Inference interface conforming to spec.md Section 3.2:
    Returns:
        {
            "predicted_intent": str,
            "confidence_score": float,
            "all_probabilities": Dict[str, float]
        }
    """
    clf = model or load_classifier()
    clean_text = normalize_text(text)

    # Predict probabilities via calibrated model
    probs = clf.predict_proba([clean_text])[0]
    classes = clf.classes_
    best_idx = np.argmax(probs)
    predicted_intent = str(classes[best_idx])
    confidence_score = float(probs[best_idx])

    prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}

    return {
        "predicted_intent": predicted_intent,
        "confidence_score": round(confidence_score, 4),
        "all_probabilities": prob_dict
    }


if __name__ == "__main__":
    train_and_benchmark_models()
