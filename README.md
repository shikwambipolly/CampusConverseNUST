# CampusConverse

CampusConverse is a multilingual, router-based conversational agent designed for student administrative services at the Namibia University of Science and Technology (NUST). It handles both deterministic lookups (academic calendars, module prerequisites, contact directories) and unstructured policy inquiries (handbooks, admissions, examination regulations) across English, Afrikaans, and Oshiwambo.

---

## Project Structure

```
CampusConverseNUST/
├── README.md                     # Project overview and quickstart instructions
├── pipeline_documentation.md     # Comprehensive stage-by-stage pipeline documentation
├── spec.md                       # Formal technical specification and system architecture
├── requirements.txt              # Pinned Python dependencies
├── raw_data/
│   └── manifest.json             # Document catalog and categorization metadata
├── data/
│   ├── nust_facts.db             # Relational SQLite database (3,494 records across 4 tables)
│   ├── chunks/
│   │   └── corpus_chunks.json    # 1,745 passage chunks across 35 institutional documents
│   └── intents/                  # Labeled intent datasets (train, val, test, combined)
├── models/
│   └── intent_classifier.joblib  # Calibrated LinearSVC champion intent model (86.67% accuracy)
├── src/
│   ├── preprocessing.py          # Normalization, language detection (en, af, kj), safety guardrails
│   ├── features.py               # Word + sub-word character TF-IDF FeatureUnion extractor
│   ├── train_intent.py           # Model training, validation tuning, and inference interface
│   ├── tools.py                  # Deterministic SQLite lookups (prereqs, calendar, contacts)
│   ├── rag.py                    # Hybrid BM25 + TF-IDF RAG search with citation formatting
│   ├── dialogue.py               # Dialogue State Manager and Clarification FSM
│   ├── orchestrator.py           # Master Conversational Orchestrator running request lifecycle
│   └── chat.py                   # Real-time interactive terminal conversation interface
├── scripts/
│   ├── download_data.py          # Automated crawler for NUST PDFs and resources
│   ├── extract_structured.py     # Structured database ingestion script
│   ├── extract_unstructured.py   # PDF text extraction and semantic chunking script
│   └── build_intent_dataset.py   # Gold-standard 400-sample intent dataset builder
└── tests/
    ├── test_preprocessing.py     # 16 tests: normalization, multilingual, guardrails
    ├── test_intent_classifier.py # 7 tests: model artifact, inference schema, routes
    ├── test_tools_and_rag.py     # 10 tests: SQLite tools, BM25+TF-IDF RAG search
    └── test_orchestrator.py      # 8 tests: end-to-end conversation flows and FSM
```

---

## Quickstart & Usage

### 1. Prerequisites & Virtual Environment
```bash
git clone https://github.com/stanleyshikwambi/CampusConverseNUST.git
cd CampusConverseNUST

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Talk to CampusConverse (Interactive Terminal Chat)
Launch the interactive assistant directly in your terminal:
```bash
python -m src.chat
```
Try asking:
* *"When do lectures commence for Semester 1?"*
* *"What are the prerequisites for DSA612S?"*
* *"What are the hostel residence rules?"*
* *"Phone number for the Katima Mulilo campus office"*
* *"Goeiedag, wanneer begin die eksamen vir semester 1?"*

### 3. Run the Full Test Suite
Execute all 41 unit and integration tests:
```bash
pytest tests/ -v
```

### 4. Retrain the Intent Classification Model
To train and benchmark candidate models (`LogisticRegression` vs `CalibratedLinearSVC`):
```bash
python -m src.train_intent
```

---

## System Architecture Highlights
* **Zero External API Dependencies:** Runs 100% locally on CPU without external API keys or cloud tokens.
* **Deterministic Precision:** Zero hallucinations on factual queries (timetables, course prerequisites, campus directories) through direct SQLite tool dispatch.
* **Hybrid RAG Engine:** BM25 lexical search combined with TF-IDF cosine similarity via Reciprocal Rank Fusion (RRF) over 1,745 policy chunks.
* **Defensive Safety Guardrail:** Fast short-circuit interception for prompt injections, code exploits, and profanity.
* **Multi-Turn Clarification FSM:** Automatically detects missing query slots (e.g. course code or semester) and clarifies before execution.
* **Multilingual Readiness:** Automatic language detection and courtesy framing for English (`en`), Afrikaans (`af`), and Oshikwanyama (`kj`).

---

## Documentation
* See [spec.md](spec.md) for the complete system architecture, intent schemas, and evaluation framework.
* See [pipeline_documentation.md](pipeline_documentation.md) for detailed documentation of every stage from data acquisition to model deployment.
