"""
src/preprocessing.py

Input Moderation, Normalization, Language Detection, and Safety Filter Pipeline.
Adheres strictly to spec.md Section 3.1:
1. Input Normalization: Strip non-printable ASCII, preserve native diacritics and Unicode.
2. Language Detection: Character n-gram based language identification (en, af, kj).
3. Safety / Moderation: Regex and lexicon filter catching abusive content and prompt injection.
"""

import math
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 1. INPUT NORMALIZATION
# ---------------------------------------------------------------------------

# Non-printable ASCII regex (0x00-0x1F, 0x7F, excluding standard \t, \n, \r)
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")
MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")

# Smart quotation and typography normalization
SMART_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "—": "-", "–": "-", "…": "..."
}


def normalize_text(text: str) -> str:
    """
    Normalizes raw user input:
    - Strips non-printable ASCII control characters.
    - Preserves native Unicode characters and diacritics (e.g., German/Afrikaans/Oshiwambo umlauts & accents).
    - Standardizes smart typography (quotes, dashes, ellipses).
    - Collapses repeated whitespace while preserving structural formatting.
    - Preserves casing of module codes (e.g. 'DSA611S') and acronyms ('NUST', 'FEBE', 'FCI').
    """
    if not text:
        return ""

    # Strip control characters
    cleaned = NON_PRINTABLE_RE.sub("", text)

    # Normalize unicode to NFC (canonical composition) to preserve diacritics
    cleaned = unicodedata.normalize("NFC", cleaned)

    # Standardize typographical marks
    for orig, repl in SMART_QUOTES.items():
        cleaned = cleaned.replace(orig, repl)

    # Collapse horizontal spaces
    lines = cleaned.split("\n")
    cleaned_lines = [MULTIPLE_SPACES_RE.sub(" ", line).strip() for line in lines]
    cleaned = "\n".join(cleaned_lines)
    cleaned = MULTIPLE_NEWLINES_RE.sub("\n\n", cleaned)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# 2. LANGUAGE DETECTION (Character N-gram Profile Algorithm)
# ---------------------------------------------------------------------------

# Curated reference n-gram distributions for campus languages:
# en: English (NUST primary medium of instruction)
# af: Afrikaans (Widely spoken regional lingua franca)
# kj: Oshikwanyama / Oshiwambo (Major indigenous Namibian language)
LANGUAGE_PROFILES = {
    "en": {
        "the", "and", "ing", "tion", "for", "that", "with", "you", "this",
        "are", "not", "student", "course", "exam", "what", "when", "how",
        "where", "who", "which", "semester", "fees", "registration", "from",
        "have", "will", "can", "module", "academic", "university", "faculty"
    },
    "af": {
        "die", "het", "nie", "van", "wat", "vir", "met", "een", "is",
        "ons", "hulle", "haar", "maar", "asseblief", "wanneer", "waar",
        "hoeveel", "kursus", "eksamen", "koshuis", "dankie", "goeie", "dag",
        "inligting", "akademie", "student", "skool", "geld", "fasiliteit"
    },
    "kj": {
        "na", "ya", "wa", "mo", "nga", "kwa", "shi", "nga", "nawa",
        "oshike", "peni", "naini", "ngapi", "omu", "oka", "eendjo", "ofikola",
        "oimaliwa", "otai", "hovele", "ondapandula", "unene", "omulongi",
        "omulongwa", "oipuphulu", "elongo", "ongalo", "efimbo", "oshilongwa"
    }
}

# Strong distinctive function words / tokens for quick unambiguous triage
DISTINCTIVE_MARKERS = {
    "af": {
        "asseblief", "baie", "dankie", "wanneer", "hoeveel", "koshuis",
        "eksamen", "goeiedag", "goeiemôre", "watter", "waarom", "alleenlik"
    },
    "kj": {
        "naini", "peni", "oshike", "ngapi", "ondapandula", "ofikola",
        "oimaliwa", "otai", "hovele", "unene", "ongalo", "waaluka", "ongeipi"
    }
}


def _extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates padded character n-grams from text"""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    words = clean.split()
    ngrams = set()
    for w in words:
        padded = f"_{w}_"
        for i in range(len(padded) - n + 1):
            ngrams.add(padded[i:i+n])
    return ngrams


def detect_language(text: str) -> Dict[str, any]:
    """
    Evaluates input text via an n-gram and distinctive token profile matcher.
    Returns:
        {
            "language": "en" | "af" | "kj",
            "confidence": float (0.0 to 1.0),
            "is_supported": bool
        }
    """
    normalized = normalize_text(text).lower()
    tokens = set(re.findall(r"\b\w+\b", normalized))

    if not tokens:
        return {"language": "en", "confidence": 1.0, "is_supported": True}

    # 1. Check for unambiguous distinctive tokens
    for lang, markers in DISTINCTIVE_MARKERS.items():
        overlap = tokens.intersection(markers)
        if overlap:
            confidence = min(0.95, 0.70 + 0.15 * len(overlap))
            return {"language": lang, "confidence": round(confidence, 2), "is_supported": True}

    # 2. Compute token overlap score against language profiles
    scores = {}
    for lang, profile in LANGUAGE_PROFILES.items():
        overlap_count = len(tokens.intersection(profile))
        scores[lang] = overlap_count

    # 3. Add character trigram score
    trigrams = _extract_char_ngrams(normalized, n=3)
    for lang, profile in LANGUAGE_PROFILES.items():
        # Count sub-word matches
        match_count = sum(1 for p in profile if p in normalized)
        scores[lang] += match_count * 0.5

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]
    total_score = sum(scores.values())

    # Fallback to English (institutional default) if scores are flat or zero
    if total_score == 0 or best_score == 0:
        return {"language": "en", "confidence": 0.80, "is_supported": True}

    confidence = round(best_score / total_score, 2)
    # Calibrate confidence bounded between 0.50 and 0.99
    confidence = max(0.50, min(0.99, confidence))

    return {
        "language": best_lang,
        "confidence": confidence,
        "is_supported": True
    }


# ---------------------------------------------------------------------------
# 3. SAFETY / INPUT MODERATION FILTER
# ---------------------------------------------------------------------------

# Abusive, profane, and harassment patterns
HARASSMENT_KEYWORDS = {
    "bitch", "bastard", "fuck", "fucking", "shit", "asshole", "idiot",
    "stupid", "moron", "cunt", "slut", "whore"
}

# Prompt injection and jailbreak attack patterns
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b",
    r"(?i)\bbypass\s+(?:system\s+)?guardrails?\b",
    r"(?i)\byou\s+are\s+now\s+(?:DAN|unrestricted|jailbroken)\b",
    r"(?i)\breveal\s+(?:the\s+)?system\s+prompt\b",
    r"(?i)\bshow\s+(?:me\s+)?your\s+(?:hidden\s+)?instructions\b",
    r"(?i)\bdrop\s+table\s+[a-z_]+\b",
    r"(?i)\b(?:exec|eval)\s*\(",
    r"(?i)<\s*script\s*>",
]

TERMINAL_SAFETY_MESSAGE = (
    "I cannot process requests containing abusive language, harassment, or "
    "security exploit attempts. Please rephrase your query with respect to "
    "NUST academic and student services."
)

PROMPT_INJECTION_MESSAGE = (
    "Security notice: System instruction overrides, jailbreak attempts, or code "
    "injection patterns are strictly prohibited. Please state your NUST campus inquiry."
)


def check_safety(text: str) -> Dict[str, any]:
    """
    Executes a lexical and pattern-based safety filter on user input.
    Returns:
        {
            "is_safe": bool,
            "flagged_categories": List[str],
            "terminal_response": Optional[str]
        }
    """
    if not text:
        return {"is_safe": True, "flagged_categories": [], "terminal_response": None}

    lower_text = text.lower()
    flagged = []

    # 1. Prompt Injection & Jailbreak Detection
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text):
            flagged.append("prompt_injection")
            return {
                "is_safe": False,
                "flagged_categories": ["prompt_injection"],
                "terminal_response": PROMPT_INJECTION_MESSAGE
            }

    # 2. Profanity & Harassment Detection
    tokens = set(re.findall(r"\b\w+\b", lower_text))
    if tokens.intersection(HARASSMENT_KEYWORDS):
        flagged.append("harassment_profanity")

    # Check multi-word harassment phrases
    if any(phrase in lower_text for phrase in ["kill yourself", "go to hell", "go die"]):
        if "harassment_profanity" not in flagged:
            flagged.append("harassment_profanity")

    if flagged:
        return {
            "is_safe": False,
            "flagged_categories": flagged,
            "terminal_response": TERMINAL_SAFETY_MESSAGE
        }

    return {
        "is_safe": True,
        "flagged_categories": [],
        "terminal_response": None
    }


# ---------------------------------------------------------------------------
# 4. UNIFIED PRE-PROCESSING PIPELINE
# ---------------------------------------------------------------------------

def preprocess_input(text: str) -> Dict[str, any]:
    """
    Unified entry point executing all Stage 3 pre-processing steps:
    1. Input Normalization
    2. Safety & Moderation Filter
    3. Language Detection
    """
    normalized = normalize_text(text)
    safety_result = check_safety(normalized)

    # If flagged as unsafe, return immediate terminal response without downstream evaluation
    if not safety_result["is_safe"]:
        return {
            "raw_text": text,
            "normalized_text": normalized,
            "is_safe": False,
            "flagged_categories": safety_result["flagged_categories"],
            "terminal_response": safety_result["terminal_response"],
            "language": "en",
            "language_confidence": 1.0,
            "is_supported_language": True
        }

    lang_result = detect_language(normalized)

    return {
        "raw_text": text,
        "normalized_text": normalized,
        "is_safe": True,
        "flagged_categories": [],
        "terminal_response": None,
        "language": lang_result["language"],
        "language_confidence": lang_result["confidence"],
        "is_supported_language": lang_result["is_supported"]
    }
