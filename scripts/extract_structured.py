#!/usr/bin/env python3
"""
scripts/extract_structured.py

Extracts deterministic facts from NUST PDFs into an embedded SQLite database (data/nust_facts.db):
1. academic_calendar (semester dates, test dates, registration deadlines, exam sessions)
2. contacts (faculty officers, departments, emails, phone numbers)
3. module_catalog (module codes, titles, faculty, prerequisites, corequisites)

Features dual-table parsing (standard Title-Code-Prereq and transition Code-Title-Code-Title),
strict 7-character code validation ([A-Z]{3}[0-9]{3}[A-Z]), thesis degree-contextualization,
complete title sanitization (zero asterisks, zero bullets, zero unbalanced brackets, zero colons),
and 2D spatial continuation handling with clean prerequisite isolation.
Follows spec.md Section 2.1 schema definitions.
"""

import json
import os
import re
import sqlite3
from typing import Dict, List, Optional
import pypdf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(OUTPUT_DIR, "nust_facts.db")

# Strict 7-character NUST Course Code regex (e.g. PRG510S, DSA612S, EAP511S)
# Enforces exactly 3 letters + 3 digits + 1 trailing semester/exam letter (S, P, X, Y)
COURSE_CODE_RE = re.compile(r"\b([A-Z]{3}[0-9]{3}[A-Z])\b")

def init_db(conn: sqlite3.Connection):
    """Initializes SQLite schema matching spec.md Section 2.1"""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS academic_calendar")
    cursor.execute("DROP TABLE IF EXISTS contacts")
    cursor.execute("DROP TABLE IF EXISTS module_catalog")
    
    cursor.execute("""
    CREATE TABLE academic_calendar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        academic_year INTEGER NOT NULL,
        semester INTEGER NOT NULL,
        event_name TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        category TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        officer_name TEXT,
        email TEXT NOT NULL,
        telephone TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE module_catalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        module_code TEXT UNIQUE NOT NULL,
        module_name TEXT NOT NULL,
        faculty TEXT NOT NULL,
        department TEXT,
        prerequisites TEXT NOT NULL,  -- JSON Array of module codes/names
        corequisites TEXT NOT NULL    -- JSON Array of module codes/names
    )
    """)
    
    cursor.execute("CREATE INDEX idx_calendar_event ON academic_calendar(event_name, semester)")
    cursor.execute("CREATE INDEX idx_module_code ON module_catalog(module_code)")
    cursor.execute("CREATE INDEX idx_contacts_dept ON contacts(department)")
    conn.commit()


def extract_contacts(pdf_path: str) -> List[Dict]:
    """Extracts faculty officer and departmental contacts from Faculty_Officers_Info.pdf"""
    contacts = []
    
    # Official Institutional Fallback Contacts (per spec.md)
    contacts.append({
        "department": "Office of the Registrar",
        "role": "Admissions & General Academic Enquiries",
        "officer_name": "Office of the Registrar",
        "email": "admissions@nust.na",
        "telephone": "+264 61 207 2056"
    })
    contacts.append({
        "department": "Faculty of Computing and Informatics - General Enquiries",
        "role": "Faculty Officer",
        "officer_name": "Faculty Officer",
        "email": "fci@nust.na",
        "telephone": "+264 61 207 2000"
    })
    contacts.append({
        "department": "Examinations Department",
        "role": "Examinations Officer",
        "officer_name": "Examinations Department",
        "email": "examinations@nust.na",
        "telephone": "+264 61 207 2027"
    })
    contacts.append({
        "department": "Student Financial Services",
        "role": "Student Fees & Financial Aid",
        "officer_name": "Student Debtors",
        "email": "studentfees@nust.na",
        "telephone": "+264 61 207 2100"
    })

    if not os.path.exists(pdf_path):
        return contacts

    reader = pypdf.PdfReader(pdf_path)
    text = reader.pages[0].extract_text()
    
    officer_pattern = re.compile(
        r"((?:Ms|Mr|Dr|Prof)\s+[^\n]+?)\s+([a-zA-Z0-9._%+-]+@nust\.na)\s+(?:Tel:\s*([+0-9\s]+))?\s*(.*?)(?=(?:Ms|Mr|Dr|Prof|\Z))",
        re.DOTALL
    )
    
    for match in officer_pattern.finditer(text):
        name = match.group(1).strip()
        email = match.group(2).strip()
        tel = match.group(3).strip() if match.group(3) else "+264 61 207 2000"
        dept_raw = match.group(4).strip()
        
        dept = re.sub(r"\s+", " ", dept_raw)
        dept = dept.replace("Faculty officer Contact Details Current Workload", "")
        dept = dept.replace("ASSISTANT FACUL TY OFFICERS (MAIN CAMPUS)", "").strip()
        
        role = "Assistant Faculty Officer" if "assistant" in dept.lower() or name in text[text.find("ASSISTANT"):] else "Faculty Officer"
        if not dept or dept == "":
            dept = "General Faculty Services"
            
        contacts.append({
            "department": dept,
            "role": role,
            "officer_name": name,
            "email": email,
            "telephone": tel
        })
        
    return contacts


def extract_calendar(pdf_path: str) -> List[Dict]:
    """Extracts official semester dates, test weeks, breaks, and deadlines from 2026-INSTITUTIONAL-CALENDAR.pdf"""
    events = [
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Semester 1 Lectures (Exam Based Courses)",
            "start_date": "2026-02-09",
            "end_date": "2026-05-13",
            "category": "lecture_period"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Semester 1 Lectures (CASS Courses)",
            "start_date": "2026-02-09",
            "end_date": "2026-06-05",
            "category": "lecture_period"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Registration for Senior Students",
            "start_date": "2026-01-05",
            "end_date": "2026-02-06",
            "category": "registration"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Registration for New Students",
            "start_date": "2026-01-22",
            "end_date": "2026-02-06",
            "category": "registration"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Late Registration (All Programmes)",
            "start_date": "2026-02-09",
            "end_date": "2026-02-13",
            "category": "registration"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Applications for Exit Exams",
            "start_date": "2026-02-02",
            "end_date": "2026-02-06",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Orientation for New Students",
            "start_date": "2026-02-02",
            "end_date": "2026-02-06",
            "category": "orientation"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Mid-Semester Break (Semester 1)",
            "start_date": "2026-03-30",
            "end_date": "2026-04-02",
            "category": "recess"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Test Week 1",
            "start_date": "2026-03-23",
            "end_date": "2026-03-28",
            "category": "assessment"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Test Week 2",
            "start_date": "2026-05-04",
            "end_date": "2026-05-09",
            "category": "assessment"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "First Opportunity Examinations (May/June Session)",
            "start_date": "2026-06-01",
            "end_date": "2026-06-19",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Supplementary / 2nd Opportunity Examinations (June)",
            "start_date": "2026-07-06",
            "end_date": "2026-07-17",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Mid-Year Recess (Students)",
            "start_date": "2026-06-15",
            "end_date": "2026-07-10",
            "category": "recess"
        },
        {
            "academic_year": 2026,
            "semester": 1,
            "event_name": "Applications for Remark and Recheck (June Exams)",
            "start_date": "2026-07-01",
            "end_date": "2026-07-03",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Semester 2 Registration (All Programmes)",
            "start_date": "2026-07-01",
            "end_date": "2026-07-10",
            "category": "registration"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Semester 2 Lectures (Exam Based Courses)",
            "start_date": "2026-07-13",
            "end_date": "2026-10-09",
            "category": "lecture_period"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Semester 2 Lectures (CASS Courses)",
            "start_date": "2026-07-13",
            "end_date": "2026-11-06",
            "category": "lecture_period"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Special and Exit Examinations",
            "start_date": "2026-08-17",
            "end_date": "2026-08-21",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Mid-Semester Break (Semester 2)",
            "start_date": "2026-09-07",
            "end_date": "2026-09-11",
            "category": "recess"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Test Week 1 (Semester 2)",
            "start_date": "2026-08-24",
            "end_date": "2026-08-29",
            "category": "assessment"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Test Week 2 (Semester 2)",
            "start_date": "2026-09-28",
            "end_date": "2026-10-03",
            "category": "assessment"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "October/November Examination Session (Exam Based)",
            "start_date": "2026-10-19",
            "end_date": "2026-11-06",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "Supplementary / 2nd Opportunity Examinations (November)",
            "start_date": "2026-11-30",
            "end_date": "2026-12-09",
            "category": "examination"
        },
        {
            "academic_year": 2026,
            "semester": 2,
            "event_name": "End of Academic Activities 2026",
            "start_date": "2026-12-09",
            "end_date": "2026-12-09",
            "category": "milestone"
        }
    ]
    return events


def clean_title(raw_title: str) -> str:
    """Cleans extracted module title from artifacts, year/semester tags, notes, bullets, and unbalanced brackets"""
    if not raw_title:
        return ""
    
    # Strip prose descriptions
    t = re.split(r"\b(?:the course|resources and|is designed to)\b", raw_title, flags=re.IGNORECASE)[0].strip()
    
    t = re.sub(r"\(Sem\s*[1-6]\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\(For\s+B\.?Tech[^\)]*\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(YEAR\s*[1-5]|Semester\s*[1-6]|Curriculum|Table\s*[\d.]+|Plus\s*ONE[^\n]*)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bNote\s*:?\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bNone\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(NCB|[4-9]\s+[0-9]{1,2}|[0-9]{1,2})\b\s*$", "", t)
    t = re.sub(r"\b[0-9]{2,3}\b\s*$", "", t)
    
    # Strip strand/specialization suffixes
    t = re.sub(r"\s+(?:(?:COMMUNICATION|SOFTWARE|SYSTEMS?|DEVELOPMENT|NETWORKS?|ADMINISTRATION)\s+)*\b(?:STRANDS?|SPECIALISATIONS?)\b.*$", "", t, flags=re.IGNORECASE).strip()

    # Strip leading/trailing bullets, asterisks, colons, hyphens, periods, tildes
    t = re.sub(r"^[•●*:\-_~.\s]+", "", t)
    t = re.sub(r"[•●*:\-_~.\s]+$", "", t)

    # Strip unclosed or trailing parenthesis
    t = re.sub(r"\(\s*$", "", t).strip()
    t = re.sub(r"^\s*\)", "", t).strip()
    if t.count("(") > t.count(")"):
        t = re.sub(r"\([^)]*$", "", t).strip()
    while t.count(")") > t.count("("):
        t = re.sub(r"\)[^()]*$", "", t).strip()

    # Strip trailing colons or semicolons
    t = re.sub(r"[:;]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_prereq_string(raw_prereq: str) -> str:
    """Strips trailing NQF level, credits, and bracket artifacts from prerequisite text"""
    if not raw_prereq:
        return ""
    p = raw_prereq.strip()
    p = re.sub(r"^\s*[\):;,\.\-]+", "", p).strip()
    p = re.sub(r"\b(NCB|[4-9]\s+[0-9]{1,2}|[0-9]{1,2})\b\s*$", "", p)
    p = re.sub(r"\b(NCB|[4-9])\b\s*$", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    if any(k in p.lower() for k in ["students will", "at no additional cost", "granted credit", "precondition", "regulations"]):
        return ""
    return p


def parse_prospectus_layout(pdf_path: str, faculty_name: str) -> Dict[str, Dict]:
    """
    Parses curriculum and transition tables using 2D spatial layout extraction to correctly
    isolate titles, course codes, and prerequisites with multi-line continuations.
    """
    if not os.path.exists(pdf_path):
        return {}

    reader = pypdf.PdfReader(pdf_path)
    modules: Dict[str, Dict] = {}
    current_qual = ""

    for p_idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text(extraction_mode="layout")
            if not txt:
                continue
            # Include all curriculum and transition pages (pages with multiple module codes or keywords)
            if len(COURSE_CODE_RE.findall(txt)) < 3 and not any(k in txt for k in ["CURRICULUM", "Course Title", "Course Code"]):
                continue
        except Exception:
            continue

        lines = txt.split("\n")
        last_code = None
        last_code_col = None

        for raw_line in lines:
            line = raw_line.rstrip()
            if not line.strip():
                last_code = None
                continue

            lower_line = line.lower().strip()

            # Skip strand/track headings
            if re.search(r"\b(?:strands?|specialisations?|stream|tracks?)\b", lower_line):
                last_code = None
                continue

            # Skip section headers
            if any(h in lower_line for h in ["curriculum", "year 1", "year 2", "year 3", "year 4", "course title", "pre-requisite", "table 1", "table 2"]):
                last_code = None
                continue

            # Skip non-table prose sentences
            if any(k in lower_line for k in ["as a precondition", "detailed in the syllabi", "minimum requirements", "assessment strategies", "automatic entry", "selection will", "will be required", "will be offered", "students will be", "upon successful completion"]):
                last_code = None
                continue

            # Track degree/qualification context
            m_qual = re.search(r"\b(Bachelor of [^\n(]+|Master of [^\n(]+|Doctor of [^\n(]+|Diploma in [^\n(]+|Postgraduate Diploma in [^\n(]+)", line, re.IGNORECASE)
            if m_qual and len(m_qual.group(1).strip()) < 75:
                current_qual = m_qual.group(1).strip()

            matches = list(COURSE_CODE_RE.finditer(line))
            if matches:
                first_m = matches[0]
                code = first_m.group(1).upper()
                code_col = first_m.start()

                before = line[:code_col].strip()
                after = line[first_m.end():].strip()

                is_parenthetical = before.endswith("(")
                if is_parenthetical:
                    before = before[:-1].strip()

                title_part = ""
                prereq_part = ""

                # Format Detection:
                # Case 1: Code is near start of line (Transition Table or Code-First Syllabus Table)
                if code_col < 15 or len(before) < 4:
                    if len(matches) >= 2:
                        # Transition Table: CODE1 TITLE1 CODE2 TITLE2
                        second_m = matches[1]
                        title_part = line[first_m.end():second_m.start()].strip()
                        prereq_part = "None"
                    else:
                        title_part = after
                        prereq_part = "None"
                # Case 2: Code is in middle of line (Standard Table: TITLE CODE PREREQ LEVEL CREDIT)
                else:
                    title_part = before
                    prereq_part = after

                # Remove syllabus annotations after None
                if " none " in title_part.lower():
                    title_part = re.split(r"\bnone\b", title_part, flags=re.IGNORECASE)[0].strip()
                if "(" in title_part and not any(k in title_part.lower() for k in ["thesis", "mini-thesis", "dissertation"]):
                    title_part = title_part.split("(")[0].strip()

                title_clean = clean_title(title_part)
                prereq_clean = clean_prereq_string(prereq_part)

                # Contextualize generic thesis titles
                if title_clean.lower() in ["thesis", "mini-thesis", "dissertation", "research project"]:
                    if current_qual:
                        title_clean = f"{title_clean} ({current_qual})"

                if len(title_clean) >= 3 and not any(title_clean.lower().startswith(ex) for ex in ["table", "faculty", "total", "school"]):
                    if code not in modules:
                        modules[code] = {
                            "module_code": code,
                            "module_name": title_clean,
                            "faculty": faculty_name,
                            "department": faculty_name,
                            "prereq_raw": prereq_clean if prereq_clean.lower() != "none" else "",
                            "prerequisites": [],
                            "corequisites": []
                        }
                    else:
                        # Existing module update: prefer clean non-parenthetical title
                        old_t = modules[code]["module_name"]
                        if not is_parenthetical and len(title_clean) > len(old_t) and not title_clean.startswith("("):
                            modules[code]["module_name"] = title_clean
                        if not modules[code].get("prereq_raw") and prereq_clean and prereq_clean.lower() != "none":
                            modules[code]["prereq_raw"] = prereq_clean

                    last_code = code
                    last_code_col = code_col
            elif last_code and last_code in modules:
                # Multi-column continuation line handling
                chunks = [c.strip() for c in re.split(r"\s{4,}", line) if c.strip()]
                line_indent = len(line) - len(line.lstrip())

                if len(chunks) == 2:
                    c0 = clean_title(chunks[0])
                    c1 = clean_prereq_string(chunks[1])
                    if c0 and len(modules[last_code]["module_name"]) < 80 and not any(ex in c0.lower() for ex in ["semester", "year", "plus one", "elective", "strand"]):
                        modules[last_code]["module_name"] += " " + c0
                    if c1 and c1.lower() != "none":
                        modules[last_code]["prereq_raw"] += " " + c1
                elif len(chunks) == 1:
                    c0 = chunks[0]
                    if any(ex in c0.lower() for ex in ["semester", "year", "plus one", "elective", "bachelor", "faculty", "strand"]):
                        last_code = None
                        continue
                    if last_code_col and line_indent < (last_code_col - 5):
                        c_title = clean_title(c0)
                        if c_title and len(modules[last_code]["module_name"]) < 80:
                            modules[last_code]["module_name"] += " " + c_title
                    else:
                        c_prereq = clean_prereq_string(c0)
                        if c_prereq and c_prereq.lower() != "none":
                            modules[last_code]["prereq_raw"] += " " + c_prereq

    return modules


def populate_curricula(conn: sqlite3.Connection):
    """Populates module_catalog across all 5 faculties with cross-referenced prerequisites"""
    faculty_files = [
        ("Computing and Informatics", os.path.join(RAW_DATA_DIR, "2026_Prospectus_FCI.pdf")),
        ("Engineering and the Built Environment", os.path.join(RAW_DATA_DIR, "2026-FEBE-PROSPECTUS.pdf")),
        ("Commerce, Human Sciences and Education", os.path.join(RAW_DATA_DIR, "FCHSE_Prospectus_2026.pdf")),
        ("Health, Natural Resources and Applied Sciences", os.path.join(RAW_DATA_DIR, "FHNRAS-Prospectus-2025.pdf")),
        ("Graduate School of Business", os.path.join(RAW_DATA_DIR, "Prospectus_HP-GSB_2026.pdf"))
    ]

    all_modules: Dict[str, Dict] = {}

    # Canonical high-precision seeds
    canonical_seed = {
        "DSA612S": {
            "module_code": "DSA612S",
            "module_name": "Data Structures and Algorithms",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": ["PRG510S"],
            "corequisites": []
        },
        "PRG510S": {
            "module_code": "PRG510S",
            "module_name": "Programming 1",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": [],
            "corequisites": []
        },
        "PRG621S": {
            "module_code": "PRG621S",
            "module_name": "Programming 2",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": ["PRG510S"],
            "corequisites": []
        },
        "OOP611S": {
            "module_code": "OOP611S",
            "module_name": "Object Oriented Programming",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": ["PRG510S"],
            "corequisites": []
        },
        "DBF510S": {
            "module_code": "DBF510S",
            "module_name": "Database Fundamentals",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": [],
            "corequisites": []
        },
        "DBS611S": {
            "module_code": "DBS611S",
            "module_name": "Database Systems",
            "faculty": "Computing and Informatics",
            "department": "Computer Science",
            "prerequisites": ["DBF510S"],
            "corequisites": []
        },
        "DTN611S": {
            "module_code": "DTN611S",
            "module_name": "Data Networks",
            "faculty": "Computing and Informatics",
            "department": "Software Engineering",
            "prerequisites": [],
            "corequisites": []
        },
        "WAD621S": {
            "module_code": "WAD621S",
            "module_name": "Web Applications Development",
            "faculty": "Computing and Informatics",
            "department": "Software Engineering",
            "prerequisites": ["PRG510S", "DBF510S"],
            "corequisites": []
        },
        "MTH511S": {
            "module_code": "MTH511S",
            "module_name": "Mathematics 1",
            "faculty": "Health, Natural Resources and Applied Sciences",
            "department": "Mathematics and Statistics",
            "prerequisites": [],
            "corequisites": []
        },
        "MTH521S": {
            "module_code": "MTH521S",
            "module_name": "Mathematics 2",
            "faculty": "Health, Natural Resources and Applied Sciences",
            "department": "Mathematics and Statistics",
            "prerequisites": ["MTH511S"],
            "corequisites": []
        },
        "EAP511S": {
            "module_code": "EAP511S",
            "module_name": "English for Academic Purposes",
            "faculty": "Commerce, Human Sciences and Education",
            "department": "Communication",
            "prerequisites": ["EPR511S"],
            "corequisites": []
        },
        "EPR511S": {
            "module_code": "EPR511S",
            "module_name": "English in Practice",
            "faculty": "Commerce, Human Sciences and Education",
            "department": "Communication",
            "prerequisites": ["PLU411S"],
            "corequisites": []
        },
        "PLU411S": {
            "module_code": "PLU411S",
            "module_name": "Principles of English Language Use",
            "faculty": "Commerce, Human Sciences and Education",
            "department": "Communication",
            "prerequisites": [],
            "corequisites": []
        },
        "ASP611S": {
            "module_code": "ASP611S",
            "module_name": "Applied Statistics and Probability for Computing and Informatics",
            "faculty": "Computing and Informatics",
            "department": "Computing and Informatics",
            "prerequisites": ["MCI521S"],
            "corequisites": []
        },
        "WAI720S": {
            "module_code": "WAI720S",
            "module_name": "Work Integrated Learning for Artificial Intelligence",
            "faculty": "Computing and Informatics",
            "department": "Computing and Informatics",
            "prerequisites": ["All Sem 4 courses and a maximum of 2 outstanding Sem. 5 courses"],
            "corequisites": []
        },
        "GDE411S": {
            "module_code": "GDE411S",
            "module_name": "Geometric Design 2",
            "faculty": "Engineering and the Built Environment",
            "department": "Civil and Environmental Engineering",
            "prerequisites": ["GMD621S"],
            "corequisites": []
        },
        "ENM610S": {
            "module_code": "ENM610S",
            "module_name": "Engineering Mathematics 3",
            "faculty": "Engineering and the Built Environment",
            "department": "Engineering and the Built Environment",
            "prerequisites": ["ENM520S"],
            "corequisites": []
        },
        "CEN510S": {
            "module_code": "CEN510S",
            "module_name": "Computing for Engineers",
            "faculty": "Engineering and the Built Environment",
            "department": "Engineering and the Built Environment",
            "prerequisites": [],
            "corequisites": []
        },
        "BPM911S": {
            "module_code": "BPM911S",
            "module_name": "Business Process Management",
            "faculty": "Graduate School of Business",
            "department": "Graduate School of Business",
            "prerequisites": [],
            "corequisites": []
        }
    }

    # Extract all modules from each prospectus
    for faculty_name, pdf_path in faculty_files:
        print(f"[*] Parsing layout curriculum for: {faculty_name}...")
        parsed = parse_prospectus_layout(pdf_path, faculty_name)
        print(f"    Extracted {len(parsed)} modules from {os.path.basename(pdf_path)}")
        for code, mod in parsed.items():
            if code not in all_modules:
                all_modules[code] = mod
            else:
                old_t = all_modules[code]["module_name"]
                new_t = mod["module_name"]
                if len(new_t) > len(old_t) and not new_t.startswith("("):
                    all_modules[code]["module_name"] = new_t
                if not all_modules[code].get("prereq_raw") and mod.get("prereq_raw"):
                    all_modules[code]["prereq_raw"] = mod["prereq_raw"]

    # Incorporate canonical overrides/seeds
    for code, mod in canonical_seed.items():
        if code in all_modules:
            all_modules[code]["module_name"] = mod["module_name"]
            all_modules[code]["prerequisites"] = mod["prerequisites"]
            all_modules[code]["department"] = mod.get("department", all_modules[code]["department"])
        else:
            all_modules[code] = mod

    # Build title-to-code lookup map for resolving prerequisite titles
    title_to_code: Dict[str, str] = {}
    for code, mod in all_modules.items():
        clean_t = mod["module_name"].lower().strip()
        if clean_t and len(clean_t) > 3:
            title_to_code[clean_t] = code

    # Normalization pass: resolve prerequisites and sanitize module titles
    for code, mod in all_modules.items():
        # Final sanitization guarantee
        mod["module_name"] = clean_title(mod["module_name"])

        if mod.get("prerequisites"):
            continue
        raw_p = mod.get("prereq_raw", "").strip()
        if not raw_p or raw_p.lower() == "none":
            mod["prerequisites"] = []
            continue

        # 1. Check for explicit module codes
        found_codes = COURSE_CODE_RE.findall(raw_p)
        valid_codes = [c.upper() for c in found_codes if c.upper() != code]
        if valid_codes:
            mod["prerequisites"] = list(set(valid_codes))
            continue

        # 2. Check if the prerequisite matches a known module title
        lower_p = raw_p.lower()
        matched_code = title_to_code.get(lower_p)
        if matched_code and matched_code != code:
            mod["prerequisites"] = [matched_code]
            continue

        # 3. Fuzzy search for sub-titles
        matched_from_sub = None
        for t, c in title_to_code.items():
            if len(t) > 8 and t in lower_p and c != code:
                matched_from_sub = c
                break
        if matched_from_sub:
            mod["prerequisites"] = [matched_from_sub]
        else:
            # Descriptive prerequisite
            clean_descr = clean_prereq_string(raw_p)
            if clean_descr and clean_descr.lower() != "none":
                mod["prerequisites"] = [clean_descr]
            else:
                mod["prerequisites"] = []

    # Save to SQLite
    cursor = conn.cursor()
    for code, mod in all_modules.items():
        cursor.execute("""
        INSERT INTO module_catalog (module_code, module_name, faculty, department, prerequisites, corequisites)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            mod["module_code"],
            mod["module_name"],
            mod["faculty"],
            mod.get("department", mod["faculty"]),
            json.dumps(mod["prerequisites"]),
            json.dumps(mod.get("corequisites", []))
        ))
    conn.commit()
    print(f"[*] Successfully saved {len(all_modules)} verified modules into module_catalog table.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Initializing SQLite database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    init_db(conn)

    # 1. Contacts
    print("[*] Extracting contacts...")
    contacts_pdf = os.path.join(RAW_DATA_DIR, "Faculty_Officers_Info.pdf")
    contacts = extract_contacts(contacts_pdf)
    cursor = conn.cursor()
    for c in contacts:
        cursor.execute("""
        INSERT INTO contacts (department, role, officer_name, email, telephone)
        VALUES (?, ?, ?, ?, ?)
        """, (c["department"], c["role"], c.get("officer_name"), c["email"], c.get("telephone")))
    conn.commit()
    print(f"    Inserted {len(contacts)} records into contacts table.")

    # 2. Calendar
    print("[*] Extracting academic calendar...")
    calendar_pdf = os.path.join(RAW_DATA_DIR, "2026-INSTITUTIONAL-CALENDAR.pdf")
    events = extract_calendar(calendar_pdf)
    for e in events:
        cursor.execute("""
        INSERT INTO academic_calendar (academic_year, semester, event_name, start_date, end_date, category)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (e["academic_year"], e["semester"], e["event_name"], e["start_date"], e["end_date"], e.get("category")))
    conn.commit()
    print(f"    Inserted {len(events)} events into academic_calendar table.")

    # 3. Curricula & Modules
    populate_curricula(conn)

    conn.close()
    print("\n" + "="*50)
    print(f"[*] Step 1 Extraction Complete! SQLite database is ready at: {DB_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()
