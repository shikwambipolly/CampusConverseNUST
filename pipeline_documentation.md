# CampusConverse: NLP Pipeline Documentation

**Project Name:** CampusConverse — Multilingual Conversational Agent for Student Services  
**Target Institution:** Namibia University of Science and Technology (NUST)  
**Scope:** University-Wide (All Faculties, Registrar, Fees, Support Services)  
**File:** `pipeline_documentation.md`  

---

## Architecture Overview

CampusConverse uses a unified hybrid agentic architecture that combines deterministic structured lookups (calendar dates, module prerequisites, fees, programmes, contacts) with 3-way hybrid semantic retrieval (BM25 + Sublinear TF-IDF + Chroma Dense Embeddings), multi-intent clause decomposition, multi-turn coreference resolution, and grounded Gemini synthesis.

```
[ Student Utterance ] ──► [ Normalization, Safety & Language Detection ]
                                               │
                                               ▼
                      [ Dialogue FSM & Multi-Turn Coreference Enrichment ]
                                               │
                                               ▼
                      [ Calibrated LinearSVC Intent Classifier + Rescue ]
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     ▼                                                   ▼
            [ Missing Slots? ]                         [ Multi-Intent Clause Decomposition ]
                     │                                                   │
           [ Clarification FSM ]                     ┌───────────────────┴───────────────────┐
                                                     ▼                                       ▼
                                       [ Deterministic SQLite Tools ]            [ 3-Way Hybrid RAG Engine ]
                                     (Prereqs, Dates, Fees, Contacts)         (BM25 + TF-IDF + Chroma Dense RRF)
                                                     │                                       │
                                                     └───────────────────┬───────────────────┘
                                                                         ▼
                                                    [ Grounded Gemini Synthesis + Citations ]
                                                         (Deterministic Offline Fallback)
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
| **Curriculum Catalogs & Handbooks** | 5 | Prospectuses 2025/2026 for **FCI**, **FEBE**, **FCHSE**, **FHNRAS**, and **HP-GSB** | Dual use: Tabular extraction of module codes, prerequisites, and corequisites into SQLite; degree curriculum breakdowns and narrative chapters into 3-Way Hybrid RAG. |
| **Institutional Regulations & Policies** | 26 | `2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf`<br>`REGISTRATION_GUIDE_2025.pdf`<br>`SRC_Constitution.pdf`<br>`Guide_to_Prospective_Students_2027.pdf` | Chunked semantic retrieval (BM25 + TF-IDF + Chroma Dense Embeddings) for student policies, admissions, and progression rules. |
| **Academic Forms & Procedures** | 6 | `Exit-Examinations-Application-Form.pdf`<br>`Special-Examinations-Application-Form.pdf`<br>`APPLICATION-FOR-COURSE-RECOGNITION-EXEMPTION-FORM.pdf`<br>`APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf`<br>`Recognition_of_prior_learning.pdf`<br>`RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf` | Procedural guidelines and eligibility requirements for administrative appeals and applications. |
| **Financials & Fees** | 2 | `NUST-Fees-Booklet-2026.pdf`<br>`payment_plan_2022.pdf` | Structured fee tables and payment deadline policies. |
| **Directories & Contacts** | 1 | `Faculty_Officers_Info.pdf` | Direct routing and escalation contacts for faculty officers and student support. |

---

## 2. Text Extraction & Cleanup

The text extraction stage splits the acquired corpus into two complementary pathways:
1. **Deterministic Structured Data (Stage 2A):** Extraction of high-precision tabular facts (module catalogs, prerequisites, academic calendar, programmes, fees, contact directories) into an embedded relational SQLite database (`data/nust_facts.db` / `data/nust_campus.db`).
2. **Unstructured Narrative & Degree Curriculum Data (Stage 2B):** Extraction, qualification-header enrichment, and chunking of institutional policies, general regulations, and full degree curriculum tables for 3-way hybrid lexical and dense vector retrieval (`data/chunks/corpus_chunks.json` and `data/chroma_db/`).

---

### 2.1 Deterministic Fact Extraction & Structured Database (Stage 2A)

#### 2.1.1 Architectural Rationale (SQLite & Semantic Layer)
CampusConverse implements an embedded relational database (`data/nust_facts.db` and `data/nust_campus.db`, supporting AlloyDB in cloud deployments via `src/semantic_layer.py`):
* **Zero External Dependencies:** Eliminates the operational overhead and failure modes of running an external database daemon locally, enabling reproducible out-of-the-box local execution and CI testing.
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

#### 2.1.3 Structured Database Statistics (`data/nust_facts.db` & `data/nust_campus.db`)

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

### 2.2 Unstructured Text & Curriculum Table Extraction for RAG (Stage 2B)

Stage 2B extracts, cleans, and segments institutional policy narratives, academic regulations, disciplinary codes, and full degree curriculum breakdowns into tokenized semantic chunks for the 3-Way Hybrid RAG engine.

#### 2.2.1 Extraction & Pre-Chunking Cleaning Protocol
Executed via `scripts/extract_unstructured.py` and indexed into ChromaDB via `scripts/ingest_data.py`:
1. **Running Artifact Elimination:** Systematically strips running headers (*"General Information and Regulations 2026"*, *"Namibia University of Science and Technology"*), footers, standalone page pagination (`[ 42 ]`, `Page 12 of 100`), and dense Table of Contents leader lines (`.....`).
2. **Qualification Header Tracking & Curriculum Table Indexing:** Dynamically tracks active degree qualification headings across pages (`BACHELOR OF ...`, `MASTER OF ...`, `DIPLOMA IN ...`, `DOCTOR OF ...`) and prepends a contextual header (`Programme: <QUALIFICATION_NAME> (Curriculum, Modules & Courses)`) to every prospectus chunk. Curriculum breakdown tables (`Year 1`, `Semester 1`, module codes, NQF levels, credits, and prerequisites) are preserved and linked to their parent programme so students can query module breakdowns for any degree and year across all faculties.
3. **Dynamic Hierarchical Section Tracking:** Recursively tracks regulatory clause prefixes (e.g., `AC4 ATTENDANCE OF LECTURES`, `AC5 EVALUATION AND EXAMINATION`, `G14 ISSUING OF DOCUMENTS`, `SR1 RULES AND CONDUCT`, `F.1.3 PAYMENT REQUIREMENTS`) to enrich metadata and ensure passages remain context-grounded.

#### 2.2.2 Chunking Strategy & Schema
Adhering to Section 2.2 of the project specification:
* **Algorithm:** Recursive Character Splitting preserving structural paragraph breaks and sentence boundaries using delimiter priority `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]`.
* **Target Chunk Size:** 400 tokens (~1,600 characters).
* **Chunk Overlap:** 50 tokens (~200 characters) to preserve contextual coherence across clause transitions.
* **Minimum Chunk Length:** 100 characters (discards trivial trailing fragment noise).
* **Destinations:** `data/chunks/corpus_chunks.json` (~3.6 MB) and persistent ChromaDB vector index (`data/chroma_db/`, collection `nust_policies`).

Each chunk strictly adheres to the schema:
```json
{
  "chunk_id": "FCI_PROSP_P61_C1",
  "source_document": "2026_Prospectus_FCI.pdf",
  "section_title": "BACHELOR OF ARTIFICIAL INTELLIGENCE HONOURS 08BAIH - Curriculum & Modules",
  "page_number": 61,
  "faculty": "Faculty of Computing and Informatics (FCI)",
  "approx_tokens": 245,
  "text": "Programme: BACHELOR OF ARTIFICIAL INTELLIGENCE HONOURS 08BAIH (Curriculum, Modules & Courses)\nCURRICULUM\nSemester 1\nCourse Title Course Code Pre-Requisite NQF Level NQF Credit\nResearch Methodology RME811S None 8 15..."
}
```

#### 2.2.3 Unstructured Corpus Analytics

The extracted corpus contains **2,340 high-quality semantic chunks** across **35 source documents and official campus portals**:

| Document Category | Source Document / Source Name | Chunks | Key Policy & Domain Coverage |
| :--- | :--- | :---: | :--- |
| **Core Academic Regulations & Strategic Plan** | `2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf`<br>`Final_NUST_Re-aligned_and_Reviewed_Strategic_Plan_2021-2025_0723.pdf` | **264** | 80% lecture attendance requirement, 40% examination subminimum, 50% course pass mark, 75% distinction threshold, academic exclusion, student disciplinary code, institutional mission and strategic priorities. |
| **Faculty Prospectuses & Degree Curricula** | `2026-FEBE-PROSPECTUS.pdf`<br>`FCHSE_Prospectus_2026.pdf`<br>`FHNRAS-Prospectus-2025.pdf`<br>`2026_Prospectus_FCI.pdf`<br>`Prospectus_HP-GSB_2026.pdf` | **1,696** | Complete year-by-year and semester-by-semester module tables (`252` distinct programme headers), NQF credit values, faculty-specific progression rules, work-integrated learning (WIL) rules, and departmental evaluation criteria across all 5 faculties. |
| **Admissions & Application Schemes** | `Guide_to_Prospective_Students_2027.pdf`<br>`2027_eMature_Age_Application_Form.pdf`<br>`2027_eUndergraduate_Application_Form.pdf`<br>`2027-ePostGraduate_Application_Form.pdf`<br>`2027_eAdvanced_Standing_Application_Form.pdf`<br>`2027-Pocket-Guide-Postgraduate.pdf`<br>`Brochure_STEMProgrammes_20223.pdf`<br>`Notice-current_and_prospective-students.pdf` | **149** | Mature Age Entry (25+ age requirement, experience, English & Math test rules), Advanced Standing credit transfers (UNAM/IUM articulation), undergraduate/postgraduate application checklists, and STEM offerings. |
| **Student Administration & Resumption** | `REGISTRATION_GUIDE_2025.pdf`<br>`RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf`<br>`NUST-Luderitz-Satelite-Campus-Flyer.pdf` | **37** | Step-by-step portal self-registration, student portal PIN reset, application for resumption of studies following leave of absence, and Lüderitz campus programmes. |
| **Postgraduate Supervision & Research** | `Guidelines-for-the-Supervision-and-Examination-of-Postgraduate-Degrees.pdf`<br>`REGISTRATION-OF-RESEARCH-TOPICS.pdf` | **61** | Master and Doctoral thesis proposal submission, supervisor quotas, external examiner appointment, anti-plagiarism guidelines, and research topic registration. |
| **Fees Policy & Student Debtors** | `NUST-Fees-Booklet-2026.pdf` (Policy Preamble pp. 1–14)<br>`CLAIMS-PROCEDURES.pdf`<br>`Revised_NUST_Refund_Form.pdf` | **20** | NSFAS student funding obligations, monthly debit order facility, SADC vs Non-SADC international student levies, fee installment rules, and tuition refund claim cutoffs. |
| **Governance & Student Affairs** | `SRC_Constitution.pdf`<br>`Residence-Application-2027.pdf` | **48** | Student rights, campus governance, SRC executive portfolios, hostel residency conduct, dining hall rules, and breakage deposit refund conditions. |
| **Examinations, Appeals & Clearances**| `EMC-Selections-Information-Booklet-2026.pdf`<br>`Updated-Notice-of-Expectation-to-Graduate_0.pdf`<br>`APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf`<br>`Graduation-Procedures_042026.pdf`<br>`Recognition_of_prior_learning.pdf`<br>`Exit-Examinations-Application-Form.pdf`<br>`Special-Examinations-Application-Form.pdf` | **36** | Expectation to graduate checklist (library and student accounts debt clearance), N$370 remark application window (3-day cutoff), exit exam single-course criteria, medical special exam petitions, gown hiring, and emergency medical care selections. |
| **Student Societies & Extracurriculars**| `student_societies.json` (Live Portal Scrape) | **21** | Mission statements, executive contacts, eligibility criteria, and activities for all 21 officially registered NUST societies. |
| **Verified Campus Support Services** | `library.nust.na` (Official Portal)<br>`2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf` (Section DSS) | **2** | Official NUST Library operating hours (Mon–Fri 08:00–22:00, Sat 09:00–17:00, Sun 14:00–20:00), lending limits, overdue fines, Campus Health Clinic hours (adjacent to Shangri-La), full-time nurse, and student counselling services. |
| **Total** | **35 Source Documents & Campus Portals** | **2,340** | **100% Institutional Policy, Degree Curriculum & Campus Life Coverage** |

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

Stage 4 extracts discriminative lexical and sub-word numerical representations from normalized student queries, adhering strictly to Section 3.2 of the project specification. Implemented in `src/features.py`.

```
                        [ Normalized Query ]
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       [ Word-Level TF-IDF ]           [ Char-Level TF-IDF ]
      Unigrams & Bigrams (1, 2)       Sub-word N-grams (3, 5)
       Sublinear Term-Frequency        Boundary Aware (char_wb)
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                        [ FeatureUnion Vector ]
                      (Dense + Sparse Compound)
```

### 4.1 Word-Level TF-IDF (`build_word_vectorizer`)
* **N-gram Range:** Unigrams and bigrams (`ngram_range=(1, 2)`) to capture multi-word campus terminology (e.g., *"exam dates"*, *"pass mark"*, *"prerequisite for"*, *"hostel fee"*).
* **Scaling:** Sublinear term frequency scaling (`sublinear_tf=True`, mapping $tf \mapsto 1 + \log(tf)$) to prevent high-frequency campus keywords from dominating sparse feature spaces.
* **Token Pattern:** `(?u)\b[a-zA-Z0-9_-]+\b`, preserving alphanumeric course codes (`DSA611S`, `PRG510S`), qualification tiers (`NSSCO`, `NSSCAS`), and campus acronyms (`FCI`, `FEBE`, `GSB`, `SRC`).

### 4.2 Sub-Word Character-Level TF-IDF (`build_char_vectorizer`)
* **Morphological Resilience:** Employs character n-grams within word boundaries (`analyzer="char_wb"`, `ngram_range=(3, 5)`).
* **Out-of-Vocabulary Robustness:** Mitigates student typing variations, minor misspellings, and inflectional morphology across regional loanwords and technical course names without relying on external stemming libraries.

### 4.3 FeatureUnion Representation (`build_feature_extractor`)
Combines word-level and character-level vectorizers into a single unified `FeatureUnion` representation pipeline that feeds directly into downstream classifiers.

---

## 5. Modelling

Stage 5 trains, tunes, and exports the statistical intent classification model, adhering to Section 3.2 and Section 5 of the project specification. Implemented in `src/train_intent.py`.

### 5.1 Architecture & Candidate Benchmarks
Two classical pipelines were trained on the **Train set (280 samples)** and tuned on the **Validation set (60 samples)**:
1. **Model A1 (Logistic Regression):** Multinomial Logistic Regression (`C=2.5`, `class_weight="balanced"`, `max_iter=1000`).
2. **Model A2 (Calibrated LinearSVC - Champion):** `LinearSVC(C=1.5, class_weight="balanced")` calibrated via `CalibratedClassifierCV(method="sigmoid", cv=3)` to output well-calibrated class probability distributions for confidence thresholding.

### 5.2 Validation Tuning Comparison

| Candidate Model | Representation | Val Accuracy | Val Macro F1 | Val Weighted F1 | Selection |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression** | Word + Char TF-IDF | 86.67% | 0.8750 | 0.8805 | Candidate |
| **Calibrated LinearSVC** | Word + Char TF-IDF | **88.33%** | **0.8922** | **0.8983** | **Champion** |

### 5.3 Exported Artifact
The winning model pipeline is serialized to `models/intent_classifier.joblib` and exposes a production inference interface `predict_intent(text: str)` returning:
```json
{
  "predicted_intent": "check_calendar_date",
  "confidence_score": 0.9412,
  "all_probabilities": {
    "check_calendar_date": 0.9412,
    "check_module_prereq": 0.0121,
    "clarification_response": 0.0084,
    "escalate_human_contact": 0.0105,
    "out_of_scope": 0.0118,
    "query_general_policy": 0.0160
  }
}
```

---

## 6. Evaluation

Stage 6 assesses the champion model on the untouched **held-out Test set (60 samples)**, measuring classification accuracy, per-class metrics, confusion matrix, and production inference latency.

### 6.1 Test Set Classification Report

```
                        precision    recall  f1-score   support

   check_calendar_date     1.0000    1.0000    1.0000        11
   check_module_prereq     1.0000    1.0000    1.0000        11
clarification_response     0.8889    1.0000    0.9412         8
escalate_human_contact     0.8000    0.5000    0.6154         8
          out_of_scope     0.7500    0.6667    0.7059         9
  query_general_policy     0.7500    0.9231    0.8276        13

              accuracy                         0.8667        60
             macro avg     0.8648    0.8483    0.8483        60
          weighted avg     0.8669    0.8667    0.8594        60
```

### 6.2 Key Metric Highlights
* **Accuracy:** **86.67%** (52 of 60 unseen queries classified correctly).
* **Deterministic Intents:**
  * `check_calendar_date`: **1.0000 Precision / 1.0000 Recall / 1.0000 F1** (11/11).
  * `check_module_prereq`: **1.0000 Precision / 1.0000 Recall / 1.0000 F1** (11/11).
  * `clarification_response`: **0.8889 Precision / 1.0000 Recall / 0.9412 F1** (8/8).
* **Policy RAG Routing:**
  * `query_general_policy`: **0.7500 Precision / 0.9231 Recall / 0.8276 F1** (12/13).
* **Inference Speed:** **0.187 ms per query** (~5,350 queries per second) on local CPU, satisfying real-time conversational SLAs.

### 6.3 Confusion Matrix

| Actual \ Predicted | `check_cal` | `check_mod` | `clarif` | `escalate` | `out_of_sc` | `query_pol` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`check_calendar_date`** | **11** | 0 | 0 | 0 | 0 | 0 |
| **`check_module_prereq`** | 0 | **11** | 0 | 0 | 0 | 0 |
| **`clarification_response`** | 0 | 0 | **8** | 0 | 0 | 0 |
| **`escalate_human_contact`** | 0 | 0 | 1 | **4** | 2 | 1 |
| **`out_of_scope`** | 0 | 0 | 0 | 0 | **6** | 3 |
| **`query_general_policy`** | 0 | 0 | 0 | 1 | 0 | **12** |

### 6.4 Verification Suite (`tests/`)
All **23 unit and integration tests** pass in **1.70 seconds**:
* `tests/test_preprocessing.py`: 16/16 tests passing.
* `tests/test_intent_classifier.py`: 7/7 tests passing.

---

## 7. Retrieval, Semantic Layer & Tool-Calling Backends

Stage 7 implements the answer-retrieval engines invoked by the intent routing subsystem, adhering strictly to Section 3.4 and Section 3.5 of the project specification.

```
                         [ Routed Single / Multi-Intent Query ]
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
     [ Deterministic DB Tools & Semantic Layer ]        [ 3-Way Hybrid Policy & Curriculum RAG ]
     (Prereqs, Dates, Contacts, Fees, Programmes)            (2,340 Chunks / 35 Documents)
                      │                                           │
      `data/nust_facts.db` / `nust_campus.db`      BM25 + TF-IDF (20k) + Chroma Dense Embeddings
                      │                                           │
                      │                           Reciprocal Rank Fusion (RRF) + Domain Boosts
                      │                                           │
                      └─────────────────────┬─────────────────────┘
                                            ▼
                         [ Grounded Context & Verifiable Citations ]
```

### 7.1 Deterministic Tool-Calling Engine & Semantic Layer (`src/tools.py`, `src/semantic_layer.py`)
For factual queries requiring exact precision, language model hallucination is prevented by querying relational SQLite / AlloyDB tables directly:
1. **`lookup_prerequisite(module_code_or_query: str)`:**
   * Uses regex pattern matching (`\b[A-Z]{3}[0-9]{3}[A-Z0-9]\b`) to extract course codes (e.g., `DSA612S`).
   * Queries `module_catalog` / `modules` returning prerequisites, corequisites, faculty, and department.
   * Supports full-sentence natural-language course-name matching (`INSTR(LOWER(?), LOWER(module_name)) > 0` for `LENGTH(module_name) >= 10`), prioritizing rows with populated prerequisites so queries like *"What is the prerequisite for Data Structures and Algorithms?"* resolve directly (`DSA521S` $\rightarrow$ `PAP521S`).
2. **`lookup_calendar_date(query: str, semester: Optional[int])`:**
   * Extracts semester indicators (`s1`, `semester 1`, `s2`, `semester 2`) and maps event keywords to structured SQL categories (`examination`, `assessment`, `registration`, `recess`, `orientation`, `lecture_period`).
   * Prevents generic start-verbs (`"start"`, `"begin"`, `"commence"`) from overriding specific event categories (e.g., ensuring *"when do first semester exams start?"* retrieves `First Opportunity Examinations (May/June Session)` rather than lecture commencement).
   * Supports Afrikaans calendar keywords (`registrasie`, `eksamen`, `toets`, `vakansie`, `reses`, `lesing`) and cross-references `exam_timetable` when a module code is present.
3. **`lookup_contact_info(query: str)`:**
   * Resolves inquiries across all **13 Regional & Satellite Campuses** (*Katima Mulilo*, *Rundu*, *Walvis Bay*, *Lüderitz*, *Keetmanshoop*, *Gobabis*, *Eenhana*, *Otjiwarongo*, *Outapi*, *Ongwediva*, *Tsumeb*, *Rietfontein*), campus support units (*Main Library*, *Student Health Clinic*, *Student Counselling*, *Examinations*, *DSS*, *SRC*), and all **Faculty / School / Department Officers** (*Computing & Informatics*, *Engineering & Built Environment*, *Health / Natural Resources / Applied Sciences*, *Commerce / Human Sciences / Education*, *HP-GSB*).
4. **`lookup_structured_facts(query: str)`:**
   * Deterministic lookup engine across `fee_structure`, `residence_fees`, `fee_payment_schedule`, `banking_details`, `admission_point_scale` (APS calculator), `programmes` (degree admission requirements & NQF levels), and `student_societies`.

### 7.2 3-Way Hybrid Retrieval-Augmented Generation (`src/rag.py`)
For institutional policies, degree curriculum module breakdowns, admissions criteria, hostel curfews, exam remarking rules, and student governance:
1. **3-Way Hybrid Search:**
   * **Sparse BM25 Index:** Exact lexical matching over word-tokenized passages (`rank_bm25.BM25Okapi`).
   * **Sublinear TF-IDF Index:** Unigram/bigram cosine similarity vector space (`max_features=20000`, `sublinear_tf=True`).
   * **Dense Chroma Vector Search:** Semantic dense embeddings (`all-MiniLM-L6-v2` via `NUSTSemanticLayer.search_policies_and_regulations` over `data/chroma_db/`).
2. **Reciprocal Rank Fusion (RRF):** Merges rankings across all three rankers over a candidate pool (`pool_size = max(top_k * 20, 120)`, $k=60$):
   $$RRF\_Score(d) = \frac{1}{60 + rank_{BM25}(d)} + \frac{1}{60 + rank_{TFIDF}(d)} + \frac{1.25}{60 + rank_{Chroma}(d)}$$
3. **Domain-Specific Reranking & Boosting:**
   * **Exact Course Code Boosting:** Queries mentioning a specific NUST module code (`\b[A-Za-z]{3}[0-9]{3}[A-Za-z0-9]\b`, e.g., `DSA612S`) apply a `+0.045` RRF boost to chunks containing that exact code.
   * **Programme-Aware Curriculum Table Boosting:** Maps `Programme:` headers across sibling chunks (`prog_to_indices`), propagates base relevance (`0.90x`) from top-ranked programme introduction pages to sibling curriculum table chunks, boosts chunks with $\ge 3$ course codes and `"curriculum"`/`"semester "` in their body (`1.85x` when the queried `"year X"` matches, `1.45x` otherwise), and automatically includes companion `_C2` chunks on the same page.
4. **Institutional Citation Injection:** Assembles retrieved passages with verifiable citations:
   `[Source: <source_document> | Section: <section_title> | Page: <page_number>]`.

### 7.3 Verification & Test Coverage (`tests/test_tools_and_rag.py`)
The suite of **10 unit tests** validates:
* Corpus indexing across all **2,340 chunks** (`assert len(self.retriever.chunks) >= 2300`).
* Regex and natural-language extraction of course codes, course names, and semester numbers.
* Exact prerequisite retrieval on real syllabus entries (`DSA612S` $\rightarrow$ `PRG510S`).
* Calendar milestone, exam window, and multilingual Afrikaans lookups.
* Regional centre, campus support, and faculty officer contacts.
* High-precision 3-Way Hybrid RAG retrieval citing `2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf`, faculty prospectuses, and hostel governance documents.

---

## 8. Unified Master Orchestration, Multi-Intent Routing & Grounded Synthesis

Stage 8 integrates all subsystems into a unified conversational agent adhering to Sections 3.6–3.8 and Section 6 of the project specification.

```
                          [ Incoming Student Query ]
                                      │
                   ┌──────────────────▼──────────────────┐
                   │ Stage 3: Normalization & Moderation │
                   └──────────────────┬──────────────────┘
                                      │
                         [ Is Query Safe & Clean? ]
                                      │
                      No ─────────────┴───────────── Yes
                      ▼                               ▼
             [ Terminal Block ]             [ Check Dialogue FSM ]
           (Security/Harassment)                      │
                                      ┌───────────────┴───────────────┐
                                      │                               │
                                  [ Idle ]              [ Awaiting Clarification ]
                                      │                               │
                     [ Classify Intent + Smart Rescue ]     [ Fill Missing Slots ]
                                      │                               │
                   [ Actionable Intent or Rescued Hit? ]    [ All Slots Satisfied? ]
                                      │                               │
                       No ────────────┴──── Yes             No ───────┴────── Yes
                       ▼                     ▼              ▼                  ▼
           [ Unanswered Contact ]            └──────────────┼──────────────────┘
                 Fallback                                   │
                                                            ▼
                                     [ Multi-Turn Coreference Enrichment ]
                                                            │
                                                            ▼
                                     [ Multi-Intent Clause Decomposition ]
                                                            │
                                                            ▼
                                  [ Execute SQLite Tools + 3-Way Hybrid RAG ]
                                                            │
                                                            ▼
                                  [ Grounded Gemini Synthesis + Citations ]
                                                            │
                                            [ Multilingual Framing ]
```

### 8.1 Clarification FSM & Multi-Turn Context Resolution (`src/dialogue.py`, `src/orchestrator.py`)
* **FSM States:** `IDLE` $\rightarrow$ `AWAITING_CLARIFICATION` $\rightarrow$ `IDLE` or `ESCALATED` (max 2 clarification attempts).
* **Multi-Turn Coreference Enrichment (`_enrich_with_turn_context`):** Resolves follow-up questions containing pronouns or definite references (`it`, `its`, `that`, `this`, `they`, `the programme`, `the module`) against the prior user turn in `session.history`, automatically carrying over module codes (e.g., Turn 1: *"What is the prerequisite for DSA612S?"* $\rightarrow$ Turn 2: *"How many credits is it worth?"*) and degree programme contexts.
* **Prerequisite vs. Curriculum Disambiguation:** When `check_module_prereq` is predicted without a module code or matched course name, slot clarification is only requested if the query contains prerequisite keywords (`"prereq"`, `"require"`, `"before taking"`); general curriculum questions (*"What courses do I take in Year 1 of Bachelor of Engineering in Civil Engineering?"*) automatically route to `query_general_policy`.

### 8.2 Smart Retrieval Rescue & Multi-Intent Decomposition (`src/orchestrator.py`)
* **Smart Fallback & Retrieval Rescue:** When classifier confidence $< 0.65$ or an `IDLE`-state query is misclassified as `out_of_scope` / `clarification_response`, the orchestrator checks `lookup_structured_facts` and 3-Way Hybrid RAG (`_has_academic_domain_signal`) before rejecting the query.
* **Generalized Multi-Intent Clause Decomposition (`_detect_active_intents`):** Splits compound student queries on conjunctions and clause boundaries (`and`, `also`, `plus`, `;`, `?`), classifies each sub-clause, and executes all distinct intents in a single turn (e.g., *Prerequisite + Policy*, *Policy + Calendar Exam Date*, *Calendar + Regional Contact*).
* **Transparent Unanswered Fallback:** When a query is out-of-scope or cannot be answered from official documents, the assistant clearly states that the question cannot be answered and provides Student Services contact details (`ask@nust.na`, `+264 61 207 9111`) without falsely claiming an automated ticket escalation.

### 8.3 Grounded Gemini Synthesis & Interactive Interfaces (`src/chatbot_agent.py`, `src/chat.py`, `app.py`, `src/mcp_server.py`)
* **Grounded Synthesis (`CampusConverseAgent.synthesize_with_context`):** Uses Gemini (`gemini-3.5-flash-lite`) to synthesize clear, student-friendly answers strictly grounded in the retrieved SQLite and 3-Way Hybrid RAG context while preserving all `[Source: ... | Section: ... | Page: ...]` citations. Automatically responds in the user's language (English, Afrikaans, Oshiwambo) and falls back deterministically to the cited context when offline or without an API key.
* **Unified Interfaces:**
  * **Terminal CLI (`src/chat.py`):** `python -m src.chat`
  * **Streamlit Web UI (`app.py`):** `streamlit run app.py`
  * **FastMCP Tool Server (`src/mcp_server.py`):** Exposes NUST semantic layer tools over the Model Context Protocol (MCP).

### 8.4 Complete Verification Suite (`tests/`)
All **41 unit and integration tests** pass:
* `tests/test_preprocessing.py`: **16/16 tests passing** (normalization, language detection, defensive guardrails).
* `tests/test_intent_classifier.py`: **7/7 tests passing** (pipeline loading, schema validation, route predictions).
* `tests/test_tools_and_rag.py`: **10/10 tests passing** (database tools, $\ge 2,300$ chunk verification, 3-Way Hybrid RAG retrieval).
* `tests/test_orchestrator.py`: **8/8 tests passing** (end-to-end multi-turn flows, slot filling, fallback, citations).
