"""
src/features.py

Feature Engineering Subsystem for Intent Classification.
Adheres strictly to spec.md Section 3.2:
- Representation: TF-IDF vectorization with sublinear term-frequency scaling.
- Combined Feature Representation:
  1. Word N-grams (1, 2): Captures semantic phrase collocations (e.g., 'exam dates', 'pass mark', 'prerequisite for').
  2. Sub-word Character N-grams (3, 5): Preserves robustness across typos, prefixes, suffixes, and morphological variations.
"""

from typing import Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from src.preprocessing import normalize_text


def build_word_vectorizer(
    max_features: Optional[int] = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Word-level TF-IDF vectorizer preserving course codes and campus acronyms"""
    return TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b",
        strip_accents=None
    )


def build_char_vectorizer(
    max_features: Optional[int] = 3000,
    ngram_range: Tuple[int, int] = (3, 5),
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Character n-gram vectorizer within word boundaries for morphological resilience"""
    return TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer="char_wb",
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        min_df=2
    )


def build_feature_extractor() -> FeatureUnion:
    """Constructs a unified Word + Character-level FeatureUnion pipeline"""
    return FeatureUnion([
        ("word_tfidf", build_word_vectorizer()),
        ("char_tfidf", build_char_vectorizer())
    ])
