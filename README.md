# CampusConverse

CampusConverse is a multilingual, router-based conversational agent designed for student administrative services at the Namibia University of Science and Technology (NUST). It handles both deterministic lookups (academic calendars, module prerequisites, fee tables, programme requirements, contact directories) and unstructured policy inquiries (handbooks, admissions, examination regulations) across English, Afrikaans, Oshiwambo, and German.

---

## Project Structure

```
CampusConverseNUST/
├── .env.example                  # Template for environment variables (Gemini, Ollama, Chroma, Postgres/AlloyDB)
├── .dockerignore                 # Excludes local caches/venvs from Docker build contexts
├── Dockerfile.backend            # Image 1: FastAPI backend + 3-Way Hybrid RAG + SQLite/Postgres + ML models
├── docker-compose.yml            # Single-command multi-container orchestration (backend + frontend + optional OSS profile)
├── README.md                     # Project overview, Docker setup, open-source guide, and quickstart instructions
├── pipeline_documentation.md     # Comprehensive stage-by-stage pipeline documentation
├── spec.md                       # Formal technical specification and system architecture
├── requirements.txt              # Pinned Python dependencies
├── frontend/                     # Image 2: Standalone React 18 + TypeScript + Vite Web UI
│   ├── Dockerfile                # Multi-stage Node 20 build -> Nginx 1.27 Alpine production image
│   ├── nginx.conf                # Serves SPA on port 80 and reverse-proxies /api & /health to backend:8080
│   ├── package.json              # React, TypeScript, Vite, and Lucide icon dependencies
│   └── src/
│       ├── App.tsx               # NUST-branded chat UI with live stack status, citations & NLP telemetry chips
│       ├── api.ts                # Typed API client (/api/chat, /api/reset, /health) & citation parser
│       ├── types.ts              # TypeScript interfaces for messages, citations, and health telemetry
│       └── index.css             # Responsive NUST Navy & Gold styling
├── raw_data/
│   └── manifest.json             # Document catalog and categorization metadata
├── data/
│   ├── nust_facts.db             # Relational SQLite database (11 structured tables)
│   ├── chunks/
│   │   └── corpus_chunks.json    # 3,121 passage chunks across 35 institutional documents
│   └── intents/                  # Labeled intent datasets (train, val, test, combined)
├── chroma_db/                    # Local persistent Chroma vector store (3,121 embedded chunks)
├── models/
│   └── intent_classifier.joblib  # Calibrated LinearSVC champion intent model (86.67% accuracy)
├── src/
│   ├── api.py                    # FastAPI production web server (/api/chat, /api/reset, /health)
│   ├── chat.py                   # Real-time interactive terminal conversation interface
│   ├── chatbot_agent.py          # LLM coordinator supporting Gemini API, open-source Ollama, & offline fallback
│   ├── dialogue.py               # Dialogue State Manager and Clarification FSM
│   ├── features.py               # Word + sub-word character TF-IDF FeatureUnion extractor
│   ├── mcp_server.py             # FastMCP Model Context Protocol server
│   ├── orchestrator.py           # Unified Master Conversational Orchestrator (multi-intent + hybrid rescue)
│   ├── preprocessing.py          # Normalization, language detection (en, af, kj), safety guardrails
│   ├── rag.py                    # 3-Way Hybrid RAG (Chroma Dense Vectors + BM25 + TF-IDF via RRF)
│   ├── semantic_layer.py         # Chroma vector search & PostgreSQL/AlloyDB/SQLite lookup layer
│   ├── tools.py                  # Deterministic SQLite lookups (prereqs, calendar, fees, programmes, contacts)
│   └── train_intent.py           # Model training, validation tuning, and inference interface
├── scripts/
│   ├── download_data.py          # Automated crawler for NUST PDFs and resources
│   ├── extract_structured.py     # Structured database ingestion script
│   ├── extract_unstructured.py   # PDF text extraction and semantic chunking script
│   ├── build_intent_dataset.py   # Gold-standard 400-sample intent dataset builder
│   └── index_chroma.py           # Chroma vector database indexing and embedding script
└── tests/
    ├── test_preprocessing.py     # 16 tests: normalization, multilingual, guardrails
    ├── test_intent_classifier.py # 7 tests: model artifact, inference schema, routes
    ├── test_tools_and_rag.py     # 10 tests: SQLite tools, 3-way Hybrid RAG search
    └── test_orchestrator.py      # 8 tests: end-to-end conversation flows and FSM
```

---

## Architecture & Open-Source Stack Alternatives

CampusConverse is engineered with a modular, container-native architecture where **every proprietary cloud service can be swapped for a 100% open-source Docker equivalent**:

| Architectural Layer | Default Local / Cloud Option | 100% Open-Source Docker Option | Configuration (`docker-compose.yml` / `.env`) |
| :--- | :--- | :--- | :--- |
| **Web Frontend UI** | React 18 + TypeScript + Nginx (`campusconverse-frontend`) | **100% Open-Source** (`node:20-alpine` $\rightarrow$ `nginx:1.27-alpine`) | Runs on `http://localhost:3000` |
| **Backend API & Orchestrator** | FastAPI + Uvicorn (`campusconverse-backend`) | **100% Open-Source** (`python:3.11-slim`) | Runs on `http://localhost:8080` |
| **Intent & Safety Guardrails** | `scikit-learn` Calibrated `LinearSVC` + Regex/N-gram | **100% Open-Source** | Embedded in backend (`<1 ms` CPU inference) |
| **3-Way Hybrid RAG** | **ChromaDB** + HuggingFace **`all-MiniLM-L6-v2`** + **`rank-bm25`** + **`scikit-learn` TF-IDF** | **100% Open-Source** | Pre-indexed inside `campusconverse-backend` (or external `chromadb/chroma` via `USE_REMOTE_CHROMA=true`) |
| **Structured Relational DB** | **SQLite** (`data/nust_facts.db`) or **Google Cloud AlloyDB** | **SQLite** (embedded) or **PostgreSQL 16** (`postgres:16-alpine`) | Set `USE_ALLOYDB=true` + `--profile opensource` for PostgreSQL 16 |
| **LLM Synthesis Layer** | **Google Gemini API** (`gemini-3.5-flash-lite`) or **Offline Deterministic** | **Ollama** (`ollama/ollama` running `llama3.2:3b`, `qwen2.5:3b`, or `mistral`) | Set `LLM_PROVIDER=ollama` + `--profile opensource` |

---

## One-Command Docker Quickstart (Recommended)

The entire system is packaged into **two separate Docker images** (`ghcr.io/shikwambipolly/campusconverse-backend` and `ghcr.io/shikwambipolly/campusconverse-frontend`) orchestrated via `docker-compose.yml`.

### Option A — Run Pre-Built Images Without Cloning the Repo (Zero-Clone 1-Liner)
Once the images are published via the **Build & Publish Docker Images (GHCR)** GitHub Action, anyone with Docker installed can run the entire stack without cloning the repository:
```bash
curl -sSL https://raw.githubusercontent.com/shikwambipolly/CampusConverseNUST/main/docker-compose.yml | docker compose -f - pull && \
curl -sSL https://raw.githubusercontent.com/shikwambipolly/CampusConverseNUST/main/docker-compose.yml | docker compose -f - up
```

### Option B — Build & Run Locally From a Cloned Repo
```bash
docker compose up --build
```
* **React + TypeScript Web UI:** Open **`http://localhost:3000`** in your browser.
* **FastAPI Backend & Swagger Docs:** Available at **`http://localhost:8000/docs`** (configurable via `BACKEND_PORT`).
* **Health & Active Stack Probe:** Available at **`http://localhost:8000/health`**.

*(Optional: If you want Google Gemini synthesis enabled in Docker, export `GEMINI_API_KEY="your-key"` or add it to `.env` before running `docker compose up --build`. Without an API key, it runs 100% offline using deterministic SQL + cited 3-Way Hybrid RAG).*

### Option C — Publish Updated Images to GitHub Container Registry (`ghcr.io`)
Image publishing is configured as a **manual GitHub Action** (`.github/workflows/docker-publish.yml`):
1. Go to the repository on GitHub $\rightarrow$ **Actions** tab.
2. Select **Build & Publish Docker Images (GHCR)** on the left.
3. Click **Run workflow** (choose branch `main` and tag `latest`).

### Option D — Run with the 100% Open-Source LLM & PostgreSQL Stack (`--profile opensource`)
To spin up the open-source **Ollama** local LLM server (`ollama/ollama`) and **PostgreSQL 16** (`postgres:16-alpine`) alongside the UI and backend:
```bash
LLM_PROVIDER=ollama docker compose --profile opensource up --build -d

# Pull a compact open-weight model into the running Ollama container (one-time):
docker exec -it campusconverse-ollama ollama pull llama3.2:3b
```

---

## Unified Request Lifecycle

Both the React Web UI (`frontend/`), the FastAPI server (`src/api.py`), and the interactive terminal (`src/chat.py`) run through the **same unified pipeline** in `src/orchestrator.py`:

1. **Preprocessing & Safety Guardrail (`src/preprocessing.py`):** Normalizes text, detects language (`en`, `af`, `kj`), and immediately blocks prompt injections, SQL exploits, or unsafe inputs.
2. **Multi-Turn Dialogue FSM (`src/dialogue.py`):** Tracks conversation state per `session_id` / `user_id` and resolves missing slots (e.g., asking for a course code when a student asks *"What are the prerequisites for this module?"*).
3. **Multi-Intent & Calibrated Intent Classification (`src/train_intent.py` + `src/orchestrator.py`):** Splits compound questions (up to 3 sub-queries) and routes high-confidence queries directly to specialized tools, while low-confidence or general queries automatically check both structured NUST tables and 3-Way Hybrid RAG before escalating.
4. **Deterministic Structured Lookups (`src/tools.py` & `src/semantic_layer.py`):** Queries 11 tables in `data/nust_facts.db` (or PostgreSQL / AlloyDB):
   - `module_catalog` & `exam_timetable` (prerequisites, corequisites, exam dates)
   - `academic_calendar` (semester milestones, registration, recess)
   - `fee_structure`, `residence_fees`, `fee_payment_schedule`, `banking_details` (tuition, hostel & meal rates, bank details)
   - `programmes` & `admission_point_scale` (degree codes, NQF levels, minimum points)
   - `contacts` & `student_societies` (regional centres, campus offices, student clubs)
5. **3-Way Hybrid RAG (`src/rag.py` & `src/semantic_layer.py`):** Fuses **ChromaDB dense embeddings (`all-MiniLM-L6-v2`) + BM25 lexical search + TF-IDF cosine similarity** across 3,121 policy chunks via Reciprocal Rank Fusion (RRF) with strict document, section, and page citations.
6. **Flexible LLM Synthesis (`src/chatbot_agent.py`):** Supports **Google Gemini API** (`LLM_PROVIDER=gemini`), open-source **Ollama** (`LLM_PROVIDER=ollama`), or **100% offline deterministic mode** (`LLM_PROVIDER=offline`).

---

## Local Development Without Docker

### 1. Setup Python Virtual Environment & Backend
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the ChromaDB Vector Index (One-Time Setup for all 3,121 chunks)
python scripts/index_chroma.py

# Start the FastAPI Backend on port 8080
uvicorn src.api:app --reload --port 8080
```

### 2. Start the React + TypeScript Frontend (Dev Mode)
In a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser (Vite automatically proxies `/api` and `/health` requests to `http://localhost:8080`).

### 3. Alternative Interfaces & Testing
* **Interactive Terminal Chat:**
  ```bash
  python -m src.chat
  ```
* **Standalone FastMCP Server:**
  ```bash
  python -m src.mcp_server
  ```
* **Automated Test Suite (41 Unit & Integration Tests):**
  ```bash
  pytest tests/ -v
  ```

---

## Documentation
* See [spec.md](spec.md) for the complete system architecture, intent schemas, and evaluation framework.
* See [pipeline_documentation.md](pipeline_documentation.md) for detailed documentation of every stage from data acquisition to model deployment.
