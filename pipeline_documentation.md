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

The deterministic database comprises **11 relational tables** and **3,494 total records** (file size: ~696 KB):

| # | Table Name | Record Count | Schema Highlights | Primary Use Case |
|---|:---|:---:|:---|:---|
| 1 | **`module_catalog`** | **2,769** | `module_code` (PK), `module_name`, `faculty`, `department`, `prerequisites` (JSON), `corequisites` (JSON) | Deterministic resolution of module prerequisite chains, course names, and faculty mappings. |
| 2 | **`exam_timetable`** | **399** | `course_code`, `course_title`, `exam_date` (YYYY-MM-DD), `session_time`, `duration_minutes`, `session_opportunity` | Exact lookup for June 2026 1st Opportunity exam dates, sessions (`08H00`, `13H00`, `18H00`), and durations. |
| 3 | **`programmes`** | **194** | `programme_code` (PK), `programme_name`, `faculty`, `nqf_level`, `duration_years`, `minimum_points`, `admission_requirements` | Comprehensive catalog of undergraduate, honours, master, and doctoral degree offerings and admission points. |
| 4 | **`contacts`** | **45** | `department`, `role`, `officer_name`, `email`, `telephone` | Escalation routing across faculty officers, central administrative desks, student support facilities (Library, Campus Health Clinic, Counselling, DSS), and **13 Regional & Satellite Campuses** (Gobabis, Keetmanshoop, Rundu, Walvis Bay, Lüderitz, Eenhana, etc.). |
| 5 | **`admission_point_scale`** | **32** | `qualification_type`, `symbol_or_grade`, `points` (1–12), `description` | Official APS conversion scale across NSSCO, NSSCAS, NSSCH, Cambridge A/AS Levels, and South African NSC. |
| 6 | **`academic_calendar`** | **24** | `academic_year`, `semester`, `event_name`, `start_date`, `end_date`, `category` | High-precision resolution of test weeks, lecture periods, recess periods, and registration deadlines. |
| 7 | **`student_societies`** | **21** | `society_name`, `slug`, `category`, `description`, `mission`, `contact_email`, `url` | Directory of registered campus societies (academic, religious, safety, media, tech, sports) scraped directly from the official portal. |
| 8 | **`fee_structure`** | **16** | `fee_category`, `fee_name`, `amount_nad`, `frequency`, `notes` | Administrative, registration, late penalty, exit exam, remarking, and recognition of prior learning (RPL) rates. |
| 9 | **`fee_payment_schedule`**| **12** | `deadline_date`, `course_type`, `percentage_due`, `cancellation_credit_percentage`, `discount_percentage`, `description` | Payment installment deadlines, course cancellation refund tiers (100%, 50%, 0%), and early payment settlement discounts. |
| 10 | **`residence_fees`** | **11** | `item_name`, `category`, `amount_nad`, `period`, `refundable`, `notes` | Hostel room tariffs (Small, Medium, Balcony, Double), breakage deposits, compulsory 50% meal levies, and dining vouchers. |
| 11 | **`banking_details`** | **1** | `bank_name`, `account_name`, `branch_name`, `branch_code`, `account_number`, `account_type`, `reference_format`, `contact_email` | Official FNB Ausspannplatz bank account credentials, branch code, and mandatory student number payment reference format. |

*Quality Verification:* Audit confirms **0 asterisks**, **0 unbalanced parentheses**, **0 trailing colons/bullets**, and **0 malformed prerequisite strings** across all 2,769 module records. Indexes exist on all frequently queried columns (`module_code`, `exam_date`, `course_code`, `programme_code`, `society_slug`, `department`).

---

### 2.2 Unstructured Text Extraction for RAG (Stage 2B)

Stage 2B extracts, cleans, and segments continuous institutional policy narratives, academic regulations, disciplinary codes, and procedures into tokenized semantic chunks for the Hybrid RAG engine.

#### 2.2.1 Extraction & Pre-Chunking Cleaning Protocol
Executed via `scripts/extract_unstructured.py`:
1. **Running Artifact Elimination:** Systematically strips running headers (*"General Information and Regulations 2026"*, *"Namibia University of Science and Technology"*), footers, standalone page pagination (`[ 42 ]`, `Page 12 of 100`), and dense Table of Contents leader lines (`.....`).
2. **Tabular Exclusion:** Detects high-density course curriculum pages in faculty prospectuses (`> 8` module codes on a single page) and skips them to avoid polluting narrative policy retrieval with raw tabular listings already indexed deterministically in SQLite.
3. **Dynamic Hierarchical Section Tracking:** Recursively tracks regulatory clause prefixes (e.g., `AC4 ATTENDANCE OF LECTURES`, `AC5 EVALUATION AND EXAMINATION`, `G14 ISSUING OF DOCUMENTS`, `SR1 RULES AND CONDUCT`, `F.1.3 PAYMENT REQUIREMENTS`) to enrich metadata and ensure passages remain context-grounded.

#### 2.2.2 Chunking Strategy & Schema
Adhering to Section 2.2 of the project specification:
* **Algorithm:** Recursive Character Splitting preserving structural paragraph breaks and sentence boundaries using delimiter priority `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`.
* **Target Chunk Size:** 400 tokens (~1,600 characters).
* **Chunk Overlap:** 50 tokens (~200 characters) to preserve contextual coherence across clause transitions.
* **Minimum Chunk Length:** 100 characters (discards trivial trailing fragment noise).
* **Destination:** `data/chunks/corpus_chunks.json` (2.6 MB).

Each chunk strictly adheres to the schema:
```json
{
  "chunk_id": "GEN_REG_P14_C2",
  "source_document": "2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf",
  "section_title": "AC4 Attendance Of Lectures",
  "page_number": 14,
  "faculty": "Institutional / General",
  "approx_tokens": 182,
  "text": "AC4.1 Attendance Requirements: A student registered for a course is required to attend all lectures, tutorials, and practicals. A student must achieve a minimum of 80% attendance to be eligible for continuous assessment and examinations..."
}
```

#### 2.2.3 Unstructured Corpus Analytics

The extracted corpus contains **1,745 high-quality semantic chunks** totaling **325,679 tokens** (~1.30 MB text) across **35 source documents and official campus portals**:

| Document Category | Source Document / Source Name | Chunks | Key Policy & Domain Coverage |
| :--- | :--- | :---: | :--- |
| **Core Academic Regulations & Strategic Plan** | `2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf`<br>`Final_NUST_Re-aligned_and_Reviewed_Strategic_Plan_2021-2025_0723.pdf` | **264** | 80% lecture attendance requirement, 40% examination subminimum, 50% course pass mark, 75% distinction threshold, academic exclusion, student disciplinary code, institutional mission and strategic priorities. |
| **Faculty Prospectuses** | `2026-FEBE-PROSPECTUS.pdf`<br>`FCHSE_Prospectus_2026.pdf`<br>`FHNRAS-Prospectus-2025.pdf`<br>`2026_Prospectus_FCI.pdf`<br>`Prospectus_HP-GSB_2026.pdf` | **1,101** | Faculty-specific progression rules, work-integrated learning (WIL) prerequisites, degree completion roadmaps, and departmental evaluation criteria across all 5 faculties. |
| **Admissions & Application Schemes** | `Guide_to_Prospective_Students_2027.pdf`<br>`2027_eMature_Age_Application_Form.pdf`<br>`2027_eUndergraduate_Application_Form.pdf`<br>`2027-ePostGraduate_Application_Form.pdf`<br>`2027_eAdvanced_Standing_Application_Form.pdf`<br>`2027-Pocket-Guide-Postgraduate.pdf`<br>`Brochure_STEMProgrammes_20223.pdf`<br>`Notice-current_and_prospective-students.pdf` | **149** | Mature Age Entry (25+ age requirement, experience, English & Math test rules), Advanced Standing credit transfers (UNAM/IUM articulation), undergraduate/postgraduate application checklists, and STEM offerings. |
| **Student Administration & Resumption** | `REGISTRATION_GUIDE_2025.pdf`<br>`RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf`<br>`NUST-Luderitz-Satelite-Campus-Flyer.pdf` | **37** | Step-by-step portal self-registration, student portal PIN reset, application for resumption of studies following leave of absence, and Lüderitz campus programmes. |
| **Postgraduate Supervision & Research** | `Guidelines-for-the-Supervision-and-Examination-of-Postgraduate-Degrees.pdf`<br>`REGISTRATION-OF-RESEARCH-TOPICS.pdf` | **61** | Master and Doctoral thesis proposal submission, supervisor quotas, external examiner appointment, anti-plagiarism guidelines, and research topic registration. |
| **Fees Policy & Student Debtors** | `NUST-Fees-Booklet-2026.pdf` (Policy Preamble pp. 1–14)<br>`CLAIMS-PROCEDURES.pdf`<br>`Revised_NUST_Refund_Form.pdf` | **20** | NSFAS student funding obligations, monthly debit order facility, SADC vs Non-SADC international student levies, fee installment rules, and tuition refund claim cutoffs. |
| **Governance & Student Affairs** | `SRC_Constitution.pdf`<br>`Residence-Application-2027.pdf` | **48** | Student rights, campus governance, SRC executive portfolios, hostel residency conduct, dining hall rules, and breakage deposit refund conditions. |
| **Examinations, Appeals & Clearances**| `EMC-Selections-Information-Booklet-2026.pdf`<br>`Updated-Notice-of-Expectation-to-Graduate_0.pdf`<br>`APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf`<br>`Graduation-Procedures_042026.pdf`<br>`Recognition_of_prior_learning.pdf`<br>`Exit-Examinations-Application-Form.pdf`<br>`Special-Examinations-Application-Form.pdf` | **36** | Expectation to graduate checklist (library and student accounts debt clearance), N$370 remark application window (3-day cutoff), exit exam single-course criteria, medical special exam petitions, gown hiring, and emergency medical care selections. |
| **Student Societies & Extracurriculars**| `student_societies.json` (Live Portal Scrape) | **21** | Mission statements, executive contacts, eligibility criteria, and activities for all 21 officially registered NUST societies. |
| **Verified Campus Support Services** | `library.nust.na` (Official Portal)<br>`2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf` (Section DSS) | **2** | Official NUST Library operating hours (Mon–Fri 08:00–22:00, Sat 09:00–17:00, Sun 14:00–20:00), lending limits, overdue fines, Campus Health Clinic hours (adjacent to Shangri-La), full-time nurse, and student counselling services. |
| **Total** | **35 Source Documents & Campus Portals** | **1,745** | **100% Institutional Asset & Campus Life Coverage** |

---

### 2.3 Intent Classification Dataset (Stage 2C)

Stage 2C synthesizes and partitions the gold-standard dataset of **400 annotated student utterances** across 6 system routing intents, adhering strictly to Section 2.3 of the project specification.

#### 2.3.1 Stratified Splitting & Design Methodology
Executed via `scripts/build_intent_dataset.py`:
* **Total Volume:** Exactly 400 labeled student utterances.
* **Stratified Splits (seed = 42):**
  * **Train Set:** **280 samples (70.0%)** $\rightarrow$ `data/intents/train.json` (42 KB)
  * **Validation Set:** **60 samples (15.0%)** $\rightarrow$ `data/intents/val.json` (8.9 KB)
  * **Held-out Test Set:** **60 samples (15.0%)** $\rightarrow$ `data/intents/test.json` (9.0 KB)
  * **Unified Dataset:** `data/intents/intent_dataset.json` (65 KB)
* **Linguistic Diversity:** Captures authentic Namibian English campus jargon (*"supps"*, *"CASS"*, *"submin"*, *"hostel"*), abbreviations (*"FCI"*, *"FEBE"*, *"FCHSE"*), formal regulatory clauses, and out-of-domain conversational queries.

#### 2.3.2 Intent Distribution & Downstream Routing Matrix

| Intent Label | Total | Train (70%) | Val (15%) | Test (15%) | Architectural Target Subsystem | Example Utterance |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`check_calendar_date`** | **70** | 49 | 10 | 11 | SQLite `academic_calendar` & `exam_timetable` | *"When do lectures commence for Semester 1 2026?"* |
| **`check_module_prereq`** | **75** | 53 | 11 | 11 | SQLite `module_catalog` & `programmes` | *"Can I register for DSA611S if I failed PRG510S?"* |
| **`query_general_policy`** | **90** | 63 | 14 | 13 | Hybrid RAG Engine (`corpus_chunks.json`) | *"What is the minimum attendance required to write exams?"* |
| **`escalate_human_contact`** | **55** | 39 | 8 | 8 | SQLite `contacts` & Escalation Handoff | *"What is the telephone number for the Rundu campus office?"* |
| **`clarification_response`** | **50** | 34 | 8 | 8 | Multi-turn Conversational Slot Filler | *"Semester 1"* / *"Full-time mode"* |
| **`out_of_scope`** | **60** | 42 | 9 | 9 | Guardrail / Fallback Policy Rejection | *"What is the weather like in Windhoek today?"* |
| **Total** | **400** | **280** | **60** | **60** | **Exact Stratified Proportions** | **400 Authentic Utterances** |

#### 2.3.3 Dataset Schema Example (`data/intents/intent_dataset.json`)
```json
{
  "id": "UTT_0042",
  "text": "What do I need to take Object Oriented Programming OOP611S?",
  "intent": "check_module_prereq",
  "split": "train"
}
```

## 3. Pre-Processing

Stage 3 implements input normalization, language detection, and defensive safety moderation, adhering strictly to Section 3.1 of the project specification. All logic is encapsulated in `src/preprocessing.py` and validated by unit tests in `tests/test_preprocessing.py`.

```
                         [ Raw User Query ]
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │     Input Normalization     │
                   │ (Unicode NFC, Strip ASCII)  │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │ Safety & Moderation Filter  │
                   └──────────────┬──────────────┘
                                  │
                     Unsafe? ─────┴───── Safe?
                        │                  │
                        ▼                  ▼
              [ Terminal Error ]  ┌──────────────────┐
              (Fast Short-Circuit)│Language Detection│
                                  │   (en, af, kj)   │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                 [ Clean Processed ]
                                 (To Intent Router)
```

### 3.1 Input Normalization (`normalize_text`)
1. **Control Character Stripping:** Strips non-printable ASCII control characters (`0x00–0x1F`, `0x7F`, preserving standard `\t` and `\n`).
2. **Canonical Unicode Composition (NFC):** Normalizes composite diacritics into canonical precomposed characters, preserving campus facility names and multilingual loanwords (e.g. *Lüderitz*, *Höpker*, *Goeiemôre*).
3. **Typography Standardization:** Converts curly/smart quotation marks (`‘’`, `“”`), em/en-dashes (`—`, `–`), and ellipses (`…`) into standard ASCII counterparts.
4. **Whitespace & Structure Preservation:** Collapses redundant horizontal spaces while retaining necessary paragraph structure.
5. **Casing & Code Integrity:** Preserves the case integrity of course codes (e.g. `DSA611S`, `PRG510S`), qualification tiers (`NSSCO`, `NSSCAS`), and campus acronyms (`FCI`, `FEBE`, `GSB`, `SRC`).

### 3.2 Multilingual Language Ingestion (`detect_language`)
1. **N-gram Profile Matcher:** Evaluates token overlaps and sub-word character trigrams against reference distributions for the three primary campus communication languages:
   * `en`: English (NUST official medium of instruction).
   * `af`: Afrikaans (regional lingua franca).
   * `kj`: Oshikwanyama / Oshiwambo (widely spoken indigenous Namibian language).
2. **Distinctive Marker Triage:** Recognizes immediate high-confidence lexical markers (*"asseblief"*, *"hoeveel"*, *"koshuis"*, *"naini"*, *"oimaliwa"*, *"ondapandula"*).
3. **Calibrated Confidence & Fallback:** Binds confidence scores between 0.50 and 0.99, gracefully defaulting to institutional English (`en`) for ambiguous or acronym-only shorthand queries.

### 3.3 Safety & Input Moderation Filter (`check_safety`)
1. **Defensive Guardrails:** Intercepts malicious, toxic, or adversarial inputs before any ML inference or database access occurs:
   * **Profanity & Harassment Filter:** Lexical matching on abusive terms and self-harm phrases.
   * **Prompt Injection & Jailbreak Interceptor:** Regular expressions catching adversarial bypasses (`ignore previous instructions`, `bypass guardrails`, `DAN mode`, `reveal system prompt`, `DROP TABLE`, `eval(`).
2. **Terminal Fast Short-Circuit:** If flagged, the pipeline immediately returns a calibrated, polite error message with `is_safe: false` and the specific violation tag, preventing unnecessary computational overhead.

### 3.4 Verification & Test Coverage (`tests/test_preprocessing.py`)
The pre-processing pipeline is verified by a dedicated test suite with **16 test cases** across all components:
* `TestNormalization` (6/6 passing): Empty inputs, non-printable character stripping, whitespace collapsing, typography standardization, diacritics preservation, and module code casing.
* `TestLanguageDetection` (4/4 passing): High-confidence identification of English, Afrikaans, and Oshiwambo, with robust neutral fallback.
* `TestSafetyAndModeration` (4/4 passing): Clean prompt pass-through, harassment detection, and prompt injection interception.
* `TestUnifiedPreprocessPipeline` (2/2 passing): End-to-end safe execution and unsafe short-circuiting.

---

## 4. Feature Engineering
*(Pending)*

---

## 5. Modelling
*(Pending)*

---

## 6. Evaluation
*(Pending)*
