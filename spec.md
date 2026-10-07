# Technical Specification Document: CampusConverse

**Project Name:** CampusConverse — A Multilingual Conversational Agent for Student Services  
**Target Domain:** Namibia University of Science and Technology (NUST)  
**Reference Scope:** Faculty of Computing and Informatics (FCI) & Office of the Registrar  
**File Name:** `spec.md`  

---

## 1. System Overview & Scope Definition

CampusConverse is a hybrid conversational system designed to resolve student administrative queries. It departs from simple, monolithic Retrieval-Augmented Generation (RAG) by enforcing a **Router-Based Agent Architecture**. 

The system treats queries along two fundamental paths:
1. **Deterministic Lookups (High Factuality/Low Tolerance for Error):** Handled via explicit Tool Calling against structured tables (dates, prerequisites, fee brackets).
2. **Semantic Policies & Regulations (Unstructured Explanations):** Handled via Hybrid Semantic Retrieval grounded strictly in official institutional documentation.

```
                                [ Student Input ]
                                        │
                                [ Pre-Processing ]
                         (Language Detection & Moderation)
                                        │
                         [ Intent & Entity Classifier ]
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
  [ Missing Slots? ]           [ Structured Intent ]       [ Unstructured Intent ]
          │                             │                             │
 [ Clarification FSM ]          [ Tool Execution ]               [ RAG Engine ]
          │                    (SQLite / JSON Lookups)      (Vector + BM25 Retrieval)
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                           [ Grounding & Confidence ]
                             (Verification Filter)
                                   /         \
                      Confidence ≥ τ         Confidence < τ
                                 /             \
                  [ Grounded Response ]   [ Fallback / Human Escalation ]
```

### 1.1 In-Scope Capabilities
* **Intent Classification:** Comparative evaluation between Classical ML and Transformer-based models.
* **Deterministic Lookups:** Dynamic tool-calling over structured datasets for dates, module codes, and prerequisites.
* **Grounded Policy Retrieval:** Chunked, vector-indexed retrieval over unstructured handbook policies.
* **Stateful Dialogue Handling:** Multi-turn slot filling for ambiguous or incomplete inquiries.
* **Multilingual Input/Output:** Zero-shot query routing and response translation across English, Afrikaans, and Oshiwambo.
* **Confidence Gating:** Strict fallback routing when confidence metrics fall below predefined thresholds.

### 1.2 Out-of-Scope Constraints
* Write/Transactional operations (e.g., student self-registration, fee payment processing).
* Arbitrary, open-domain generative conversation.

---

## 2. Data Engineering Specification

Data must be segregated into structured and unstructured corpora before processing.

### 2.1 Structured Corpus (Deterministic Lookups)
Structured tables isolate critical academic facts to eliminate generative hallucinations.

* **Storage Target:** Embedded SQLite or validated JSON schemas.
* **Schema Definition:**

```json
{
  "academic_calendar": [
    {
      "academic_year": 2026,
      "semester": 1,
      "event_name": "Test Week 1",
      "start_date": "2026-04-13",
      "end_date": "2026-04-18"
    }
  ],
  "module_catalog": [
    {
      "module_code": "DSA612S",
      "module_name": "Data Structures and Algorithms",
      "faculty": "Computing and Informatics",
      "department": "Computer Science",
      "prerequisites": ["PRG510S"],
      "corequisites": []
    }
  ],
  "contacts": [
    {
      "department": "Faculty of Computing and Informatics - General Enquiries",
      "role": "Faculty Officer",
      "email": "fci@nust.na",
      "telephone": "+264 61 207 2000"
    }
  ]
}
```

### 2.2 Unstructured Corpus (RAG Engine)
* **Sources:** NUST General Information and Regulations Prospectus; FCI Yearbook; Examination Regulations.
* **Extraction Protocol:** Extract raw text from source PDFs, stripping header/footer metadata, page numbering, and visual artifacts.
* **Chunking Strategy:**
  * **Algorithm:** Recursive Character Splitting based on document hierarchy (Sections $\rightarrow$ Paragraphs).
  * **Target Chunk Size:** 400 tokens (~1,600 characters).
  * **Chunk Overlap:** 50 tokens (~200 characters) to preserve contextual boundaries.
* **Chunk Metadata Schema:**
  ```json
  {
    "chunk_id": "FCI_YB_2026_SEC_4_1",
    "source_document": "FCI_Yearbook_2026.pdf",
    "section_title": "Rules for Progression into Next Academic Year",
    "page_number": 42,
    "faculty": "FCI"
  }
  ```

### 2.3 Intent Classification Dataset
* **Volume:** 400 labeled student utterances.
* **Splits:** 70% Train, 15% Validation, 15% Held-out Test.
* **Target Intent Schema:**
  1. `check_calendar_date` (Structured)
  2. `check_module_prereq` (Structured)
  3. `query_general_policy` (Unstructured RAG)
  4. `escalate_human_contact` (Direct Handoff)
  5. `clarification_response` (Multi-turn Slot Payload)
  6. `out_of_scope` (OOD Detection)

---

## 3. Subsystem Architecture

### 3.1 Input Moderation & Language Ingestion
1. **Input Normalization:** Strip non-printable ASCII characters; preserve native diacritics and Unicode characters.
2. **Language Detection:**
   * Evaluate query via an n-gram language detection algorithm.
   * Tag input with ISO 639-1 language code (`en`, `af`, `kj`).
3. **Safety / Moderation:** Run a regex- and list-based lexical filter to catch abusive/profane content. If flagged, return an immediate terminal error response without triggering model execution.

---

### 3.2 Intent Classification Subsystem

The architecture requires two parallel classifiers to satisfy comparative benchmarks:

```
                          [ Input Utterance ]
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
        [ Classical Model ]                [ Transformer Model ]
         TF-IDF Representation              Contextual Embeddings
                 │                                   │
      Linear Classifier (SVM/LR)               SetFit / RoBERTa
                 │                                   │
          Intent + Score                      Intent + Score
                 └─────────────────┬─────────────────┘
                                   │
                        [ Benchmark Comparator ]
                        (Selected for Production)
```

#### Implementation Models
* **Model A (Classical ML):** 
  * Representation: TF-IDF vectorization (unigrams and bigrams, sublinear term-frequency scaling).
  * Classifier: Linear Support Vector Classifier (`LinearSVC`) with calibrated probabilities or Multinomial Logistic Regression.
* **Model B (Transformer-Based):** 
  * Architecture: Sentence-Transformer fine-tuned via Contrastive Learning (`SetFit`) or fine-tuned `DistilBERT-base-uncased`.
* **Output Interface:**
  ```json
  {
    "predicted_intent": "check_calendar_date",
    "confidence_score": 0.92,
    "extracted_entities": [
      {
        "entity_type": "event_type",
        "raw_text": "Test 1"
      }
    ]
  }
  ```

---

### 3.3 Routing Subsystem

The Router processes the classifier payload using a deterministic state matrix:

```
[ Predicted Intent ]
        │
        ├─► `check_calendar_date`   ──► Trigger: Calendar Lookup Tool
        ├─► `check_module_prereq`   ──► Trigger: Prerequisite Lookup Tool
        ├─► `escalate_human_contact`──► Trigger: Human Escalation Formatter
        ├─► `query_general_policy`  ──► Trigger: Unstructured RAG Pipeline
        └─► `out_of_scope`          ──► Trigger: Out-of-Domain Fallback
```

---

### 3.4 Deterministic Tool-Calling Engine

For queries requiring verifiable factual precision, language model inference is bypassed in favor of direct database queries.

* **Registered Tools:**
  1. `lookup_prerequisite(module_code: str) -> dict`
  2. `lookup_calendar_date(event_name: str, semester: int) -> dict`
  3. `lookup_contact_info(department: str) -> dict`
* **Execution Flow:**
  1. Extract required arguments from entity resolution.
  2. If arguments are missing or malformed, dispatch a state change to the **Clarification FSM**.
  3. Execute exact match over local SQLite / JSON tables.
  4. Pass raw record directly to the Context Formatter.

---

### 3.5 Retrieval-Augmented Generation (RAG) Subsystem

When the intent maps to institutional policy queries:

```
               [ Input Query ]
                      │
           [ Query Transformation ]
      (Resolve references via History)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Dense Vector Index ]      [ Sparse BM25 Index ]
(Multilingual MiniLM)          (Lexical Search)
        │                           │
        └─────────────┬─────────────┘
                      │
         [ Reciprocal Rank Fusion ]
                      │
             Top-k Scored Chunks
                      │
           [ Similarity Filtering ]
             (Distance Threshold)
```

1. **Embedding Model:** Dense multilingual encoder (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) generating normalized fixed-dimension dense vectors.
2. **Indexing Strategy:** 
   * **Dense Search:** Cosine similarity via flat/HNSW vector index.
   * **Sparse Search:** Exact-match BM25 index over the same chunks.
   * **Hybrid Merging:** Reciprocal Rank Fusion (RRF) to combine dense and sparse rankings:
     $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$
3. **Retrieval Depth:** Top $k=3$ unique passages.
4. **Context Window Assembly:** Retain chunk metadata directly within the prompt context injection to support strict source citation.

---

### 3.6 Multi-Turn Dialogue & Clarification Engine

To handle incomplete student queries (e.g., *"When is Test 1?"* without specifying the semester or module):

#### Finite State Machine (FSM) Specification
```
 [State: IDLE] 
       │ 
       ▼ (User Query Missing Mandatory Slots)
 [State: AWAITING_CLARIFICATION]
       │
       ├─► (Valid Clarification Received) ──► Transition: [State: IDLE] & Execute Intent
       └─► (Exceeded Max Turns [2])       ──► Transition: [State: ESCALATED]
```

#### Dialogue State Storage Schema
```json
{
  "session_id": "uuid4_string",
  "current_state": "AWAITING_CLARIFICATION",
  "turn_count": 1,
  "pending_intent": "check_calendar_date",
  "collected_slots": {
    "event_type": "Test 1"
  },
  "missing_slots": ["semester"],
  "history": [
    {"role": "user", "content": "When is Test 1?"},
    {"role": "assistant", "content": "Which semester are you inquiring about: Semester 1 or Semester 2?"}
  ]
}
```

---

### 3.7 Grounded Generation & Anti-Hallucination Guardrails

The LLM is restricted to a strictly defined context synthesis task.

#### 1. System Prompt Contract
```text
SYSTEM: You are the official NUST Student Services Conversational Agent.
You must answer the student's question using EXCLUSIVELY the provided context below.
Constraints:
1. Do not rely on external or general knowledge.
2. If the context does not contain sufficient facts to answer the question with absolute certainty, declare that the information is unavailable.
3. Every factual claim must include a citation referencing the source document and page number.
4. If responding in Afrikaans or Oshiwambo, preserve all proper nouns, module codes, and dates exactly as written.

CONTEXT:
{context_data}

CONVERSATION HISTORY:
{history}

STUDENT QUERY:
{query}
```

#### 2. Verification Filter (Post-Generation Critic)
* If the LLM generates a response without grounded citations from the context block, the response is discarded.
* If discarded, the system activates the Fallback Subsystem.

---

### 3.8 Fallback & Escalation Engine

The system defaults to a structured handoff under any of the following conditions:
* Intent classification confidence falls below the acceptance threshold: $\tau_{intent} < 0.65$.
* Semantic search similarity falls below the vector threshold: $\tau_{retrieval} < 0.70$.
* The multi-turn clarification turn limit is reached ($turns > 2$) without resolving required slots.
* The LLM generation critic detects ungrounded output.

#### Escalation Output Contract
The response must explicitly decline the answer and return a validated contact point:
```json
{
  "status": "fallback",
  "reason": "low_confidence_retrieval",
  "message": "I cannot find authoritative information regarding this request in the current NUST handbooks.",
  "escalation_target": {
    "entity": "Office of the Registrar",
    "email": "admissions@nust.na",
    "location": "Administration Building, Ground Floor, Windhoek Campus"
  }
}
```

---

## 4. Multilingual Processing Strategy

To satisfy the multilingual design requirement:

1. **Query Processing:**
   * Incoming non-English utterances (e.g., Afrikaans, Oshiwambo) are indexed directly against the multilingual vector space via the multilingual embedding model without intermediate machine translation. This preserves domain-specific terms and context.
2. **Deterministic Lookups:**
   * Slot and entity extraction maps multilingual terms (e.g., *"Toets 1"*) to standardized schema values (`"Test 1"`).
3. **Response Generation:**
   * The generation layer receives the retrieved English handbook passages and is instructed to generate the final response directly in the user's detected language.
   * If local terms lack unambiguous translations, the system falls back to English terminology alongside the native-language framing.

---

## 5. Formal Evaluation Framework

System evaluation requires quantitative metrics for both classification and generation subsystems.

### 5.1 Intent Classification Benchmark
Evaluate the classical model against the transformer model on the held-out test split ($N=60$):
* **Quantitative Metrics:**
  * Categorical Accuracy
  * Macro-averaged Precision, Recall, and F1-score
  * Per-class Confusion Matrix
  * Mean Inference Latency per sample (ms)
  * Disk and Runtime Memory Footprint (MB)

### 5.2 Retrieval & Grounded Generation Evaluation
Evaluate using an offline evaluation set of 30 distinct student query-context-answer triples:
* **Context Relevance:** Ratio of retrieved chunks containing the ground-truth answer.
* **Faithfulness (Groundedness):** Percentage of factual claims in the generated response directly traceable to the retrieved context chunks.
* **Answer Relevance:** Semantic alignment between the student question and the generated answer.
* **Escalation Accuracy:** 100% of out-of-scope or unanswerable queries must successfully route to fallback without generating speculative responses.

---

## 6. System Execution Lifecycle

For every user message, the execution pipeline runs through the following sequence:

```text
Sequence of Execution per Request:
1. Ingestion: Sanitize incoming string; identify user session; check lexical profanity.
2. Language Identification: Detect input language; attach metadata to session.
3. State Check: Check if session ID has a pending FSM state (e.g., AWAITING_CLARIFICATION).
   - If pending: Treat input as slot payload; validate against expected slot schema.
   - If idle: Run Intent Classification (Transformer).
4. Confidence Evaluation: Check if intent confidence score meets the threshold τ >= 0.65.
   - If below threshold: Trigger Fallback Engine (Immediate Return).
5. Route Intent:
   - If Tool Intent: Check required slots.
     * If slots missing: Transition FSM to AWAITING_CLARIFICATION; yield question to user.
     * If slots valid: Execute database query; format result into context.
   - If Policy Intent: 
     * Compute query vector; execute hybrid vector/BM25 retrieval over handbooks.
     * Evaluate retrieval similarity score against threshold τ >= 0.70.
     * If below threshold: Trigger Fallback Engine (Immediate Return).
6. Generation: Pass formatted context + conversation history to the grounded LLM prompt.
7. Verification: Parse output for required citations.
   - If valid: Return localized response to student; persist turn to session state.
   - If invalid: Trigger Fallback Engine; route to official human contact point.
```