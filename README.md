# CampusConverse

CampusConverse is a multilingual, router-based conversational agent designed for student administrative services at the Namibia University of Science and Technology (NUST). It handles both deterministic lookups (academic calendars, module prerequisites, contact directories) and unstructured policy inquiries (handbooks, admissions, examination regulations) across English, Afrikaans, and Oshiwambo.

---

## Project Structure

```
CampusConverseNUST/
├── .gitignore                    # Git ignore rules (ignores raw PDFs and virtualenv)
├── README.md                     # Project overview and setup instructions
├── requirements.txt              # Pinned Python dependencies
├── spec.md                       # Technical specification and architecture
├── pipeline_documentation.md     # NLP pipeline stage-by-stage documentation
├── raw_data/
│   └── manifest.json             # Document catalog and categorization metadata
├── scripts/
│   └── download_data.py          # Automated crawler/downloader for NUST documents
└── data/                         # Extracted SQLite database and RAG text chunks
```

---

## Quickstart & Reproducibility

### 1. Prerequisites
* Python 3.10+
* Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/stanleyshikwambi/CampusConverseNUST.git
cd CampusConverseNUST

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Acquire Data
To download the institutional prospectuses, calendars, fee guides, and policies from the NUST portal:
```bash
python scripts/download_data.py
```
This populates `raw_data/` and verifies downloads against `raw_data/manifest.json`.

---

## Documentation
* See [spec.md](spec.md) for the complete system architecture, intent schemas, and evaluation framework.
* See [pipeline_documentation.md](pipeline_documentation.md) for the active log of data acquisition, text extraction, pre-processing, feature engineering, and modeling stages.
