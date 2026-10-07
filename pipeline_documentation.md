# CampusConverse: NLP Pipeline Documentation

**Project Name:** CampusConverse — Multilingual Conversational Agent for Student Services  
**Target Institution:** Namibia University of Science and Technology (NUST)  
**Scope:** University-Wide (All Faculties, Registrar, Fees, Support Services)  
**File:** `pipeline_documentation.md`  

---

## Architecture Overview

CampusConverse uses a hybrid agentic architecture that decouples deterministic structured lookups (calendar dates, module prerequisites, fees, contacts) from unstructured semantic retrieval (regulations, admission rules, conduct, policies).

```
[ Student Utterance ] ──► [ Ingestion & Moderation ] ──► [ Intent & Entity Classifier ]
                                                                   │
           ┌───────────────────────────────────────────────────────┴──────────────────────────────┐
           ▼                                                       ▼                              ▼
  [ Missing Slots? ]                                     [ Structured Intent ]          [ Unstructured Intent ]
           │                                                       │                              │
  [ Clarification FSM ]                                   [ Tool Execution ]             [ Hybrid RAG Engine ]
                                                       (SQLite / JSON Lookups)         (Dense + BM25 Fusion)
```

---

## 1. Data Acquisition

### 1.1 Source Definition
* **Primary Source:** Namibia University of Science and Technology (NUST) Official Documents & Forms Portal: `https://www.nust.na/documents-and-forms`.
* **Institutional Scope:** University-wide coverage across all academic faculties, administrative units, and student support structures:
  * Faculty of Computing and Informatics (FCI)
  * Faculty of Engineering and the Built Environment (FEBE)
  * Faculty of Commerce, Human Sciences and Education (FCHSE)
  * Faculty of Health, Natural Resources and Applied Sciences (FHNRAS)
  * Harold Pupkewitz Graduate School of Business (HP-GSB)
  * Office of the Registrar & Student Services

### 1.2 Acquisition Method
* Automated extraction via custom ingestion script [`scripts/download_data.py`].
* The script crawls the target portal, filters out irrelevant operational procurement notices and legacy historical reports (pre-2021), and downloads active handbooks, regulations, calendars, and directories directly into `raw_data/`.
* A machine-readable manifest (`raw_data/manifest.json`) is maintained to track:
  * Document Title
  * Source URL
  * Local File Path
  * Functional Target (`deterministic_calendar`, `deterministic_catalog`, `deterministic_contacts`, `deterministic_fees`, `unstructured_policy`)

### 1.3 Acquired Corpora Breakdown

Execution of [`scripts/download_data.py`] acquired **42 academic documents** totaling **53.36 MB** stored in `raw_data/`, indexed by [`raw_data/manifest.json`].

| Target Category | Document Count | Key Representative Files | Primary Downstream Purpose |
| :--- | :--- | :--- | :--- |
| **Deterministic Calendars** | 2 | `2026-INSTITUTIONAL-CALENDAR.pdf`<br>`June_2026_Exam_Timetable.pdf` | Deterministic tool lookups for semester milestones, test weeks, and exam dates. |
| **Curriculum Catalogs & Handbooks** | 5 | Prospectuses 2025/2026 for **FCI**, **FEBE**, **FCHSE**, **FHNRAS**, and **HP-GSB** | Dual use: Tabular extraction of module codes, prerequisites, and corequisites into SQLite; narrative chapters into RAG. |
| **Institutional Regulations & Policies** | 26 | `2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf`<br>`REGISTRATION_GUIDE_2025.pdf`<br>`SRC_Constitution.pdf`<br>`Guide_to_Prospective_Students_2027.pdf` | Chunked semantic retrieval (BM25 + Dense Multilingual Embeddings) for student policies, admissions, and progression rules. |
| **Academic Forms & Procedures** | 6 | `Exit-Examinations-Application-Form.pdf`<br>`Special-Examinations-Application-Form.pdf`<br>`APPLICATION-FOR-COURSE-RECOGNITION-EXEMPTION-FORM.pdf`<br>`APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf`<br>`Recognition_of_prior_learning.pdf`<br>`RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf` | Procedural guidelines and eligibility requirements for administrative appeals and applications. |
| **Financials & Fees** | 2 | `NUST-Fees-Booklet-2026.pdf`<br>`payment_plan_2022.pdf` | Structured fee tables and payment deadline policies. |
| **Directories & Contacts** | 1 | `Faculty_Officers_Info.pdf` | Direct routing and escalation contacts for faculty officers and student support. |

---

## 2. Text Extraction & Cleanup
*(Pending)*

---

## 3. Pre-Processing
*(Pending)*

---

## 4. Feature Engineering
*(Pending)*

---

## 5. Modelling
*(Pending)*

---

## 6. Evaluation
*(Pending)*
