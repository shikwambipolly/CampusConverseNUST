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

The text extraction stage splits the acquired corpus into two distinct pathways:
1. **Deterministic Structured Data (Stage 2A):** Extraction of high-precision tabular facts (module catalogs, prerequisites, academic calendar, contact directories) into an embedded relational SQLite database.
2. **Unstructured Narrative Data (Stage 2B):** Extraction and chunking of continuous policy text, general regulations, codes of conduct, and administrative procedures for dense and lexical vector retrieval (Pending).

---

### 2.1 Deterministic Fact Extraction & Structured Database (Stage 2A)

#### 2.1.1 Architectural Rationale (SQLite vs. MongoDB)
CampusConverse implements an embedded relational database (`data/nust_facts.db`) via Python's standard library `sqlite3`:
* **Zero External Dependencies:** Eliminates the operational overhead and failure modes of running an external database daemon (e.g. `mongod`), enabling reproducible out-of-the-box local execution and CI testing.
* **Strict Relational Integrity:** Enforces unique constraints on course codes, structured date columns (`YYYY-MM-DD`), and indexed lookups.
* **Native JSON Array Storage:** Prerequisite and corequisite relationships are stored as native JSON text arrays (`["PRG510S", "DBF510S"]`), maintaining schema flexibility while preserving relational guarantees.

#### 2.1.2 Extraction Methodology & Challenges
Structured extraction is executed via `scripts/extract_structured.py` and processes raw university prospectuses, the 2026 Institutional Calendar, and the Faculty Officers Directory.

Several document anomalies in university prospectuses required specialized parsing mechanisms:
1. **Dual-Table Structural Archetypes:**
   * *Standard Curriculum Tables:* Formatted as `[Course Title] [Course Code] [Prerequisite] [NQF Level] [Credit]`. The course code appears after the title (`col >= 15`).
   * *Transition & Equivalence Tables:* Formatted as `[Course Code 1] [Course Title 1] [Course Code 2] [Course Title 2]`. The course code appears at the line start (`col < 15`).
   * *Solution:* 2D spatial column-offset detection dynamically identifies table archetype and assigns Title/Prerequisite text blocks accordingly.
2. **Multi-Line Continuation & Layout Drift:**
   * Multi-column table wrapping frequently concatenated adjacent column text or paragraph prose onto previous modules.
   * *Solution:* Implemented multi-column chunk splitting (`\s{4,}`), indent-based column boundaries, and explicit line-continuation resets on blank lines and section headers.
3. **Strict Course Code Validation:**
   * Narrative text mentions (e.g., *"detailed in the syllabi for WIL420 as a precondition"*) generated false-positive module entries.
   * *Solution:* Enforced strict 7-character regex validation `\b([A-Z]{3}[0-9]{3}[A-Z])\b`, rejecting narrative references.
4. **Comprehensive Title & Prerequisite Sanitization:**
   * Elective footnote markers, asterisks, and bullet points (`*`, `●`, `•`, `:`, `-`) were stripped using regex boundaries.
   * Trailing unclosed brackets from `Title (CODE)` parenthetical references were systematically balanced and stripped.
   * Specialization track headers (`STRANDS:`, `SPECIALISATIONS:`) were excluded from module continuation lines.
   * Generic thesis and dissertation modules were disambiguated by contextually appending qualification names (e.g., `Thesis (Master of Computer Science)`).

#### 2.1.3 Structured Database Statistics (`data/nust_facts.db`)

| Table | Record Count | Schema Highlights | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`module_catalog`** | **2,769** | `module_code` (UNIQUE), `module_name`, `faculty`, `department`, `prerequisites` (JSON), `corequisites` (JSON) | Deterministic answering of course prerequisite chains, module naming, and faculty mappings. |
| **`academic_calendar`** | **24** | `academic_year`, `semester`, `event_name`, `start_date`, `end_date`, `category` | High-precision resolution of test weeks, registration windows, and exam dates. |
| **`contacts`** | **24** | `department`, `role`, `officer_name`, `email`, `telephone` | Escalation routing for faculty officers, admissions, exams, and student debt queries. |

*Quality Verification:* Audit confirms **0 asterisks**, **0 unbalanced parentheses**, **0 trailing colons/bullets**, and **0 malformed prerequisite strings** across all 2,769 records.

---

### 2.2 Unstructured Text Extraction for RAG (Stage 2B)
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
