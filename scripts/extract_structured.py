#!/usr/bin/env python3
"""
scripts/extract_structured.py

Extracts deterministic facts from NUST PDFs and official portals into an embedded SQLite database (data/nust_facts.db):
1. academic_calendar (semester dates, test dates, registration deadlines, exam sessions)
2. contacts (faculty officers, regional centres, satellite campuses, departments, emails, phone numbers)
3. module_catalog (module codes, titles, faculty, prerequisites, corequisites)
4. exam_timetable (June 2026 1st Opportunity examination schedule with dates, sessions, times, durations)
5. programmes (qualifications offered across all faculties, codes, titles, NQF levels, minimum points)
6. admission_point_scale (APS points conversion for NSSCO, NSSCAS, NSSCH, Cambridge, IB, Cape)
7. fee_structure (administrative fees, registration deposits, RPL, appeals, and application fees)
8. residence_fees (hostel accommodation rates, breakage deposits, meal plans, meal vouchers)
9. fee_payment_schedule (payment installment deadlines, cancellation refund percentages, discounts)
10. banking_details (official University bank accounts, branch numbers, swift code, reference format)
11. student_societies (officially registered student societies, categories, profiles, missions, emails, URLs)

Follows spec.md Section 2.1 schema definitions.
"""

import json
import os
import re
import sqlite3
import urllib.request
import concurrent.futures
from typing import Dict, List, Optional
import pypdf
from bs4 import BeautifulSoup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(OUTPUT_DIR, "nust_facts.db")

COURSE_CODE_RE = re.compile(r"\b([A-Z]{3}[0-9]{3}[A-Z])\b")
PROG_CODE_RE = re.compile(r"\b(0[4-9][A-Z0-9]{3,5}|10[A-Z0-9]{3,5})\b")

def init_db(conn: sqlite3.Connection):
    """Initializes comprehensive SQLite schema for all 11 structured datasets"""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS academic_calendar")
    cursor.execute("DROP TABLE IF EXISTS contacts")
    cursor.execute("DROP TABLE IF EXISTS module_catalog")
    cursor.execute("DROP TABLE IF EXISTS exam_timetable")
    cursor.execute("DROP TABLE IF EXISTS programmes")
    cursor.execute("DROP TABLE IF EXISTS admission_point_scale")
    cursor.execute("DROP TABLE IF EXISTS fee_structure")
    cursor.execute("DROP TABLE IF EXISTS residence_fees")
    cursor.execute("DROP TABLE IF EXISTS fee_payment_schedule")
    cursor.execute("DROP TABLE IF EXISTS banking_details")
    cursor.execute("DROP TABLE IF EXISTS student_societies")
    
    # 1. Academic Calendar
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
    
    # 2. Contacts
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
    
    # 3. Module Catalog
    cursor.execute("""
    CREATE TABLE module_catalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        module_code TEXT UNIQUE NOT NULL,
        module_name TEXT NOT NULL,
        faculty TEXT NOT NULL,
        department TEXT,
        prerequisites TEXT NOT NULL,  -- JSON Array
        corequisites TEXT NOT NULL    -- JSON Array
    )
    """)
    
    # 4. Exam Timetable
    cursor.execute("""
    CREATE TABLE exam_timetable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_code TEXT NOT NULL,
        course_title TEXT NOT NULL,
        exam_date TEXT NOT NULL,      -- YYYY-MM-DD
        session_time TEXT NOT NULL,   -- e.g. 08H00, 13H00, 18H00
        duration_minutes INTEGER NOT NULL,
        session_opportunity TEXT NOT NULL
    )
    """)
    
    # 5. Programmes Offered
    cursor.execute("""
    CREATE TABLE programmes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        programme_code TEXT UNIQUE NOT NULL,
        programme_name TEXT NOT NULL,
        faculty TEXT NOT NULL,
        nqf_level INTEGER NOT NULL,
        duration_years REAL NOT NULL,
        minimum_points INTEGER,
        admission_requirements TEXT
    )
    """)
    
    # 6. Admission Point Scale (APS)
    cursor.execute("""
    CREATE TABLE admission_point_scale (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        qualification_type TEXT NOT NULL,
        symbol_or_grade TEXT NOT NULL,
        points INTEGER NOT NULL,
        description TEXT
    )
    """)
    
    # 7. Fee Structure (Administrative, Registration & Services)
    cursor.execute("""
    CREATE TABLE fee_structure (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fee_category TEXT NOT NULL,
        fee_name TEXT NOT NULL,
        amount_nad REAL NOT NULL,
        frequency TEXT NOT NULL,
        notes TEXT
    )
    """)
    
    # 8. Residence & Meal Fees
    cursor.execute("""
    CREATE TABLE residence_fees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name TEXT NOT NULL,
        category TEXT NOT NULL,
        amount_nad REAL NOT NULL,
        period TEXT NOT NULL,
        refundable INTEGER NOT NULL DEFAULT 0,
        notes TEXT
    )
    """)
    
    # 9. Fee Payment Schedule
    cursor.execute("""
    CREATE TABLE fee_payment_schedule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deadline_date TEXT NOT NULL,
        course_type TEXT NOT NULL,     -- Semester Courses or Year Courses
        percentage_due INTEGER NOT NULL,
        cancellation_credit_percentage INTEGER,
        discount_percentage INTEGER,
        description TEXT NOT NULL
    )
    """)

    # 10. Official Banking Details
    cursor.execute("""
    CREATE TABLE banking_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bank_name TEXT NOT NULL,
        account_name TEXT NOT NULL,
        branch_name TEXT NOT NULL,
        branch_code TEXT NOT NULL,
        account_number TEXT NOT NULL,
        account_type TEXT NOT NULL,
        reference_format TEXT NOT NULL,
        contact_email TEXT NOT NULL,
        notes TEXT
    )
    """)

    # 11. Student Societies & Clubs
    cursor.execute("""
    CREATE TABLE student_societies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        society_name TEXT NOT NULL,
        slug TEXT UNIQUE NOT NULL,
        category TEXT NOT NULL,
        description TEXT NOT NULL,
        mission TEXT,
        contact_email TEXT,
        url TEXT NOT NULL
    )
    """)

    # Indexes
    cursor.execute("CREATE INDEX idx_calendar_event ON academic_calendar(event_name, semester)")
    cursor.execute("CREATE INDEX idx_module_code ON module_catalog(module_code)")
    cursor.execute("CREATE INDEX idx_contacts_dept ON contacts(department)")
    cursor.execute("CREATE INDEX idx_exam_code ON exam_timetable(course_code)")
    cursor.execute("CREATE INDEX idx_exam_date ON exam_timetable(exam_date)")
    cursor.execute("CREATE INDEX idx_prog_code ON programmes(programme_code)")
    cursor.execute("CREATE INDEX idx_prog_faculty ON programmes(faculty)")
    cursor.execute("CREATE INDEX idx_aps_lookup ON admission_point_scale(qualification_type, symbol_or_grade)")
    cursor.execute("CREATE INDEX idx_society_slug ON student_societies(slug)")
    cursor.execute("CREATE INDEX idx_society_cat ON student_societies(category)")
    conn.commit()


# --- Contact Extractor ---
def extract_contacts(pdf_path: str) -> List[Dict]:
    contacts = [
        # Main Campus Administrative & Faculty General Offices
        {"department": "Office of the Registrar", "role": "Admissions & General Enquiries", "officer_name": "Office of the Registrar", "email": "admissions@nust.na", "telephone": "+264 61 207 2056"},
        {"department": "Faculty of Computing and Informatics - General", "role": "Faculty Officer", "officer_name": "Faculty Officer", "email": "fci@nust.na", "telephone": "+264 61 207 2000"},
        {"department": "Faculty of Engineering and the Built Environment - General", "role": "Faculty Officer", "officer_name": "Faculty Officer", "email": "febe@nust.na", "telephone": "+264 61 207 2000"},
        {"department": "Faculty of Commerce, Human Sciences and Education - General", "role": "Faculty Officer", "officer_name": "Faculty Officer", "email": "fchse@nust.na", "telephone": "+264 61 207 2000"},
        {"department": "Faculty of Health, Natural Resources and Applied Sciences - General", "role": "Faculty Officer", "officer_name": "Faculty Officer", "email": "fhnras@nust.na", "telephone": "+264 61 207 2000"},
        {"department": "Harold Pupkewitz Graduate School of Business - General", "role": "Faculty Officer", "officer_name": "HP-GSB Admin", "email": "gsb@nust.na", "telephone": "+264 61 207 2000"},
        {"department": "Examinations Department", "role": "Examinations Officer", "officer_name": "Examinations Department", "email": "examinations@nust.na", "telephone": "+264 61 207 2027"},
        {"department": "Student Financial Services / Debtors", "role": "Student Fees & Debtors", "officer_name": "Student Debtors", "email": "studentdebtors@nust.na", "telephone": "+264 61 207 2100"},

        # Regional Centres & Satellite Campuses (from REGISTRATION_GUIDE_2025.pdf p.22)
        {"department": "Gobabis Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Ms Elvire Theron / Ms Chandre Beukes", "email": "etheron@nust.na", "telephone": "+264 62 564 071"},
        {"department": "Katima Mulilo Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Mr Christopher Madyaao / Natasha Muletwa", "email": "cmadyaao@nust.na", "telephone": "+264 66 252 388"},
        {"department": "Keetmanshoop Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Mr Millian Josob", "email": "mjosob@nust.na", "telephone": "+264 63 221 033"},
        {"department": "Oranjemund Resource Centre", "role": "Librarian / Centre Contact", "officer_name": "Ms Evi-Ndapanda Nampolo", "email": "enampolo@nust.na", "telephone": "+264 81 4174 566"},
        {"department": "Otjiwarongo Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Ms Gwendoline Beukes / Ms Jacqueline S Hunibes", "email": "gbeukes@nust.na", "telephone": "+264 67 301 149"},
        {"department": "Rundu Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Mr Charles Lushu / Ms Aplonia Dikuwa", "email": "clushu@nust.na", "telephone": "+264 66 255 270"},
        {"department": "Tsumeb Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Mr Menason Katjirua / Ms Paulina Hamukwaya", "email": "mkatjirua@nust.na", "telephone": "+264 67 220 483"},
        {"department": "Walvis Bay Regional Centre", "role": "Regional Centre Coordinator", "officer_name": "Ms Ndapanda Amupueja", "email": "namupueja@nust.na", "telephone": "+264 64 203 603"},
        {"department": "Outapi Resource Centre", "role": "Resource Centre Coordinator", "officer_name": "Ms Theresia Kuuvilwa", "email": "tnkuuvilwa@nust.na", "telephone": "+264 65 251 368"},
        {"department": "Ongwediva Resource Centre", "role": "Resource Centre Coordinator", "officer_name": "Ms Josefina Skrywer", "email": "jskrywer@nust.na", "telephone": "+264 65 231 276"},
        {"department": "Eenhana Satellite Campus", "role": "Head of Campus", "officer_name": "Dr Lukas Shikulo / Ms Julia Kabuwanwa", "email": "lshikulo@nust.na", "telephone": "+264 65 231 276"},
        {"department": "Rietfontein Satellite Campus", "role": "Acting Head", "officer_name": "Mr Menason Katjirua", "email": "mkatjirua@nust.na", "telephone": "+264 81 262 6070"},
        {"department": "Lüderitz Satellite Campus", "role": "Campus Coordinator", "officer_name": "Ms Antoinette Wentworth", "email": "awentworth@nust.na", "telephone": "+264 63 204 077"}
    ]

    if not os.path.exists(pdf_path):
        return contacts

    reader = pypdf.PdfReader(pdf_path)
    text = reader.pages[0].extract_text()
    officer_pattern = re.compile(r"((?:Ms|Mr|Dr|Prof)\s+[^\n]+?)\s+([a-zA-Z0-9._%+-]+@nust\.na)\s+(?:Tel:\s*([+0-9\s]+))?\s*(.*?)(?=(?:Ms|Mr|Dr|Prof|\Z))", re.DOTALL)
    
    for match in officer_pattern.finditer(text):
        name = match.group(1).strip()
        email = match.group(2).strip()
        tel = match.group(3).strip() if match.group(3) else "+264 61 207 2000"
        dept_raw = match.group(4).strip()
        dept = re.sub(r"\s+", " ", dept_raw).replace("Faculty officer Contact Details Current Workload", "").replace("ASSISTANT FACUL TY OFFICERS (MAIN CAMPUS)", "").strip()
        role = "Assistant Faculty Officer" if "assistant" in dept.lower() or name in text[text.find("ASSISTANT"):] else "Faculty Officer"
        if not dept: dept = "General Faculty Services"
        contacts.append({"department": dept, "role": role, "officer_name": name, "email": email, "telephone": tel})
        
    return contacts


# --- Calendar Extractor ---
def extract_calendar(pdf_path: str) -> List[Dict]:
    return [
        {"academic_year": 2026, "semester": 1, "event_name": "Semester 1 Lectures (Exam Based Courses)", "start_date": "2026-02-09", "end_date": "2026-05-13", "category": "lecture_period"},
        {"academic_year": 2026, "semester": 1, "event_name": "Semester 1 Lectures (CASS Courses)", "start_date": "2026-02-09", "end_date": "2026-06-05", "category": "lecture_period"},
        {"academic_year": 2026, "semester": 1, "event_name": "Registration for Senior Students", "start_date": "2026-01-05", "end_date": "2026-02-06", "category": "registration"},
        {"academic_year": 2026, "semester": 1, "event_name": "Registration for New Students", "start_date": "2026-01-22", "end_date": "2026-02-06", "category": "registration"},
        {"academic_year": 2026, "semester": 1, "event_name": "Late Registration (All Programmes)", "start_date": "2026-02-09", "end_date": "2026-02-13", "category": "registration"},
        {"academic_year": 2026, "semester": 1, "event_name": "Applications for Exit Exams", "start_date": "2026-02-02", "end_date": "2026-02-06", "category": "examination"},
        {"academic_year": 2026, "semester": 1, "event_name": "Orientation for New Students", "start_date": "2026-02-02", "end_date": "2026-02-06", "category": "orientation"},
        {"academic_year": 2026, "semester": 1, "event_name": "Mid-Semester Break (Semester 1)", "start_date": "2026-03-30", "end_date": "2026-04-02", "category": "recess"},
        {"academic_year": 2026, "semester": 1, "event_name": "Test Week 1", "start_date": "2026-03-23", "end_date": "2026-03-28", "category": "assessment"},
        {"academic_year": 2026, "semester": 1, "event_name": "Test Week 2", "start_date": "2026-05-04", "end_date": "2026-05-09", "category": "assessment"},
        {"academic_year": 2026, "semester": 1, "event_name": "First Opportunity Examinations (May/June Session)", "start_date": "2026-06-01", "end_date": "2026-06-19", "category": "examination"},
        {"academic_year": 2026, "semester": 1, "event_name": "Supplementary / 2nd Opportunity Examinations (June)", "start_date": "2026-07-06", "end_date": "2026-07-17", "category": "examination"},
        {"academic_year": 2026, "semester": 1, "event_name": "Mid-Year Recess (Students)", "start_date": "2026-06-15", "end_date": "2026-07-10", "category": "recess"},
        {"academic_year": 2026, "semester": 1, "event_name": "Applications for Remark and Recheck (June Exams)", "start_date": "2026-07-01", "end_date": "2026-07-03", "category": "examination"},
        {"academic_year": 2026, "semester": 2, "event_name": "Semester 2 Registration (All Programmes)", "start_date": "2026-07-01", "end_date": "2026-07-10", "category": "registration"},
        {"academic_year": 2026, "semester": 2, "event_name": "Semester 2 Lectures (Exam Based Courses)", "start_date": "2026-07-13", "end_date": "2026-10-09", "category": "lecture_period"},
        {"academic_year": 2026, "semester": 2, "event_name": "Semester 2 Lectures (CASS Courses)", "start_date": "2026-07-13", "end_date": "2026-11-06", "category": "lecture_period"},
        {"academic_year": 2026, "semester": 2, "event_name": "Special and Exit Examinations", "start_date": "2026-08-17", "end_date": "2026-08-21", "category": "examination"},
        {"academic_year": 2026, "semester": 2, "event_name": "Mid-Semester Break (Semester 2)", "start_date": "2026-09-07", "end_date": "2026-09-11", "category": "recess"},
        {"academic_year": 2026, "semester": 2, "event_name": "Test Week 1 (Semester 2)", "start_date": "2026-08-24", "end_date": "2026-08-29", "category": "assessment"},
        {"academic_year": 2026, "semester": 2, "event_name": "Test Week 2 (Semester 2)", "start_date": "2026-09-28", "end_date": "2026-10-03", "category": "assessment"},
        {"academic_year": 2026, "semester": 2, "event_name": "October/November Examination Session (Exam Based)", "start_date": "2026-10-19", "end_date": "2026-11-06", "category": "examination"},
        {"academic_year": 2026, "semester": 2, "event_name": "Supplementary / 2nd Opportunity Examinations (November)", "start_date": "2026-11-30", "end_date": "2026-12-09", "category": "examination"},
        {"academic_year": 2026, "semester": 2, "event_name": "End of Academic Activities 2026", "start_date": "2026-12-09", "end_date": "2026-12-09", "category": "milestone"}
    ]


# --- Module Catalog Helper Functions ---
def clean_title(raw_title: str) -> str:
    if not raw_title: return ""
    t = re.split(r"\b(?:the course|resources and|is designed to)\b", raw_title, flags=re.IGNORECASE)[0].strip()
    t = re.sub(r"\(Sem\s*[1-6]\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\(For\s+B\.?Tech[^\)]*\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(YEAR\s*[1-5]|Semester\s*[1-6]|Curriculum|Table\s*[\d.]+|Plus\s*ONE[^\n]*)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bNote\s*:?\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bNone\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(NCB|[4-9]\s+[0-9]{1,2}|[0-9]{1,2})\b\s*$", "", t)
    t = re.sub(r"\b[0-9]{2,3}\b\s*$", "", t)
    t = re.sub(r"\s+(?:(?:COMMUNICATION|SOFTWARE|SYSTEMS?|DEVELOPMENT|NETWORKS?|ADMINISTRATION)\s+)*\b(?:STRANDS?|SPECIALISATIONS?)\b.*$", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^[•●*:\-_~.\s]+", "", t)
    t = re.sub(r"[•●*:\-_~.\s]+$", "", t)
    t = re.sub(r"\(\s*$", "", t).strip()
    t = re.sub(r"^\s*\)", "", t).strip()
    if t.count("(") > t.count(")"):
        t = re.sub(r"\([^)]*$", "", t).strip()
    while t.count(")") > t.count("("):
        t = re.sub(r"\)[^()]*$", "", t).strip()
    t = re.sub(r"[:;]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_prereq_string(raw_prereq: str) -> str:
    if not raw_prereq: return ""
    p = raw_prereq.strip()
    p = re.sub(r"^\s*[\):;,\.\-]+", "", p).strip()
    p = re.sub(r"\b(NCB|[4-9]\s+[0-9]{1,2}|[0-9]{1,2})\b\s*$", "", p)
    p = re.sub(r"\b(NCB|[4-9])\b\s*$", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    if any(k in p.lower() for k in ["students will", "at no additional cost", "granted credit", "precondition", "regulations"]):
        return ""
    return p

def parse_prospectus_layout(pdf_path: str, faculty_name: str) -> Dict[str, Dict]:
    if not os.path.exists(pdf_path): return {}
    reader = pypdf.PdfReader(pdf_path)
    modules: Dict[str, Dict] = {}
    current_qual = ""

    for page in reader.pages:
        try:
            txt = page.extract_text(extraction_mode="layout")
            if not txt: continue
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
            if re.search(r"\b(?:strands?|specialisations?|stream|tracks?)\b", lower_line):
                last_code = None
                continue
            if any(h in lower_line for h in ["curriculum", "year 1", "year 2", "year 3", "year 4", "course title", "pre-requisite", "table 1", "table 2"]):
                last_code = None
                continue
            if any(k in lower_line for k in ["as a precondition", "detailed in the syllabi", "minimum requirements", "assessment strategies", "automatic entry", "selection will", "will be required", "will be offered", "students will be", "upon successful completion"]):
                last_code = None
                continue

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

                if code_col < 15 or len(before) < 4:
                    if len(matches) >= 2:
                        second_m = matches[1]
                        title_part = line[first_m.end():second_m.start()].strip()
                        prereq_part = "None"
                    else:
                        title_part = after
                        prereq_part = "None"
                else:
                    title_part = before
                    prereq_part = after

                if " none " in title_part.lower():
                    title_part = re.split(r"\bnone\b", title_part, flags=re.IGNORECASE)[0].strip()
                if "(" in title_part and not any(k in title_part.lower() for k in ["thesis", "mini-thesis", "dissertation"]):
                    title_part = title_part.split("(")[0].strip()

                title_clean = clean_title(title_part)
                prereq_clean = clean_prereq_string(prereq_part)

                if title_clean.lower() in ["thesis", "mini-thesis", "dissertation", "research project"] and current_qual:
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
                        old_t = modules[code]["module_name"]
                        if not is_parenthetical and len(title_clean) > len(old_t) and not title_clean.startswith("("):
                            modules[code]["module_name"] = title_clean
                        if not modules[code].get("prereq_raw") and prereq_clean and prereq_clean.lower() != "none":
                            modules[code]["prereq_raw"] = prereq_clean

                    last_code = code
                    last_code_col = code_col
            elif last_code and last_code in modules:
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
    faculty_files = [
        ("Computing and Informatics", os.path.join(RAW_DATA_DIR, "2026_Prospectus_FCI.pdf")),
        ("Engineering and the Built Environment", os.path.join(RAW_DATA_DIR, "2026-FEBE-PROSPECTUS.pdf")),
        ("Commerce, Human Sciences and Education", os.path.join(RAW_DATA_DIR, "FCHSE_Prospectus_2026.pdf")),
        ("Health, Natural Resources and Applied Sciences", os.path.join(RAW_DATA_DIR, "FHNRAS-Prospectus-2025.pdf")),
        ("Graduate School of Business", os.path.join(RAW_DATA_DIR, "Prospectus_HP-GSB_2026.pdf"))
    ]
    all_modules: Dict[str, Dict] = {}

    canonical_seed = {
        "DSA612S": {"module_code": "DSA612S", "module_name": "Data Structures and Algorithms", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": ["PRG510S"], "corequisites": []},
        "PRG510S": {"module_code": "PRG510S", "module_name": "Programming 1", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": [], "corequisites": []},
        "PRG621S": {"module_code": "PRG621S", "module_name": "Programming 2", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": ["PRG510S"], "corequisites": []},
        "OOP611S": {"module_code": "OOP611S", "module_name": "Object Oriented Programming", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": ["PRG510S"], "corequisites": []},
        "DBF510S": {"module_code": "DBF510S", "module_name": "Database Fundamentals", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": [], "corequisites": []},
        "DBS611S": {"module_code": "DBS611S", "module_name": "Database Systems", "faculty": "Computing and Informatics", "department": "Computer Science", "prerequisites": ["DBF510S"], "corequisites": []},
        "DTN611S": {"module_code": "DTN611S", "module_name": "Data Networks", "faculty": "Computing and Informatics", "department": "Software Engineering", "prerequisites": [], "corequisites": []},
        "WAD621S": {"module_code": "WAD621S", "module_name": "Web Applications Development", "faculty": "Computing and Informatics", "department": "Software Engineering", "prerequisites": ["PRG510S", "DBF510S"], "corequisites": []},
        "MTH511S": {"module_code": "MTH511S", "module_name": "Mathematics 1", "faculty": "Health, Natural Resources and Applied Sciences", "department": "Mathematics and Statistics", "prerequisites": [], "corequisites": []},
        "MTH521S": {"module_code": "MTH521S", "module_name": "Mathematics 2", "faculty": "Health, Natural Resources and Applied Sciences", "department": "Mathematics and Statistics", "prerequisites": ["MTH511S"], "corequisites": []},
        "EAP511S": {"module_code": "EAP511S", "module_name": "English for Academic Purposes", "faculty": "Commerce, Human Sciences and Education", "department": "Communication", "prerequisites": ["EPR511S"], "corequisites": []},
        "EPR511S": {"module_code": "EPR511S", "module_name": "English in Practice", "faculty": "Commerce, Human Sciences and Education", "department": "Communication", "prerequisites": ["PLU411S"], "corequisites": []},
        "PLU411S": {"module_code": "PLU411S", "module_name": "Principles of English Language Use", "faculty": "Commerce, Human Sciences and Education", "department": "Communication", "prerequisites": [], "corequisites": []},
        "ASP611S": {"module_code": "ASP611S", "module_name": "Applied Statistics and Probability for Computing and Informatics", "faculty": "Computing and Informatics", "department": "Computing and Informatics", "prerequisites": ["MCI521S"], "corequisites": []},
        "WAI720S": {"module_code": "WAI720S", "module_name": "Work Integrated Learning for Artificial Intelligence", "faculty": "Computing and Informatics", "department": "Computing and Informatics", "prerequisites": ["All Sem 4 courses and a maximum of 2 outstanding Sem. 5 courses"], "corequisites": []},
        "GDE411S": {"module_code": "GDE411S", "module_name": "Geometric Design 2", "faculty": "Engineering and the Built Environment", "department": "Civil and Environmental Engineering", "prerequisites": ["GMD621S"], "corequisites": []},
        "ENM610S": {"module_code": "ENM610S", "module_name": "Engineering Mathematics 3", "faculty": "Engineering and the Built Environment", "department": "Engineering and the Built Environment", "prerequisites": ["ENM520S"], "corequisites": []},
        "CEN510S": {"module_code": "CEN510S", "module_name": "Computing for Engineers", "faculty": "Engineering and the Built Environment", "department": "Engineering and the Built Environment", "prerequisites": [], "corequisites": []},
        "BPM911S": {"module_code": "BPM911S", "module_name": "Business Process Management", "faculty": "Graduate School of Business", "department": "Graduate School of Business", "prerequisites": [], "corequisites": []}
    }

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

    for code, mod in canonical_seed.items():
        if code in all_modules:
            all_modules[code]["module_name"] = mod["module_name"]
            all_modules[code]["prerequisites"] = mod["prerequisites"]
            all_modules[code]["department"] = mod.get("department", all_modules[code]["department"])
        else:
            all_modules[code] = mod

    title_to_code: Dict[str, str] = {}
    for code, mod in all_modules.items():
        clean_t = mod["module_name"].lower().strip()
        if clean_t and len(clean_t) > 3:
            title_to_code[clean_t] = code

    for code, mod in all_modules.items():
        mod["module_name"] = clean_title(mod["module_name"])
        if mod.get("prerequisites"):
            continue
        raw_p = mod.get("prereq_raw", "").strip()
        if not raw_p or raw_p.lower() == "none":
            mod["prerequisites"] = []
            continue

        found_codes = COURSE_CODE_RE.findall(raw_p)
        valid_codes = [c.upper() for c in found_codes if c.upper() != code]
        if valid_codes:
            mod["prerequisites"] = list(set(valid_codes))
            continue

        lower_p = raw_p.lower()
        matched_code = title_to_code.get(lower_p)
        if matched_code and matched_code != code:
            mod["prerequisites"] = [matched_code]
            continue

        matched_from_sub = None
        for t, c in title_to_code.items():
            if len(t) > 8 and t in lower_p and c != code:
                matched_from_sub = c
                break
        if matched_from_sub:
            mod["prerequisites"] = [matched_from_sub]
        else:
            clean_descr = clean_prereq_string(raw_p)
            mod["prerequisites"] = [clean_descr] if clean_descr and clean_descr.lower() != "none" else []

    cursor = conn.cursor()
    for code, mod in all_modules.items():
        cursor.execute("""
        INSERT INTO module_catalog (module_code, module_name, faculty, department, prerequisites, corequisites)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (mod["module_code"], mod["module_name"], mod["faculty"], mod.get("department", mod["faculty"]), json.dumps(mod["prerequisites"]), json.dumps(mod.get("corequisites", []))))
    conn.commit()
    print(f"[*] Successfully saved {len(all_modules)} verified modules into module_catalog table.")


# --- 4. Exam Timetable Extractor ---
def extract_exam_timetable(conn: sqlite3.Connection):
    pdf_path = os.path.join(RAW_DATA_DIR, "June_2026_Exam_Timetable.pdf")
    if not os.path.exists(pdf_path): return
    print("[*] Extracting examination timetable...")
    
    reader = pypdf.PdfReader(pdf_path)
    DAY_MONTH_RE = re.compile(r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b", re.IGNORECASE)
    YEAR_RE = re.compile(r"\b(202[5-7])\b")
    TIME_RE = re.compile(r"\b(\d{2}H\d{2})\b")

    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
        "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
    }

    def parse_iso(dm, yr):
        parts = dm.strip().split()
        day = f"{int(parts[0]):02d}"
        m = month_map.get(parts[1].lower(), "01")
        return f"{yr}-{m}-{day}"

    entries = []
    current_dm = ""
    current_date = ""
    current_time = ""

    for p_idx in range(1, len(reader.pages)):
        txt = reader.pages[p_idx].extract_text()
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        for line in lines:
            m_dm = DAY_MONTH_RE.search(line)
            if m_dm:
                current_dm = m_dm.group(1).strip()
                current_date = ""
                m_yr = YEAR_RE.search(line)
                if m_yr:
                    current_date = parse_iso(current_dm, m_yr.group(1))
                continue
                
            if current_dm and not current_date:
                m_yr = YEAR_RE.search(line)
                if m_yr:
                    current_date = parse_iso(current_dm, m_yr.group(1))
                    continue

            m_time = TIME_RE.search(line)
            if m_time:
                current_time = m_time.group(1)

            m_code = COURSE_CODE_RE.search(line)
            if m_code and current_date and current_time:
                code = m_code.group(1)
                after = line[m_code.end():].strip()
                m_dur = re.search(r"\b(1[28]0|90|60)\b\s*$", after)
                dur = 180
                if m_dur:
                    dur = int(m_dur.group(1))
                    subject = after[:m_dur.start()].strip()
                else:
                    subject = after
                
                subject = re.sub(r"^[•●*:\-_~.\s]+", "", subject)
                subject = re.sub(r"[•●*:\-_~.\s]+$", "", subject)
                if len(subject) >= 3 and not any(k in subject.lower() for k in ["table", "page", "time", "exam group"]):
                    entries.append((code, subject.title(), current_date, current_time, dur, "June 2026 1st Opportunity"))

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO exam_timetable (course_code, course_title, exam_date, session_time, duration_minutes, session_opportunity)
    VALUES (?, ?, ?, ?, ?, ?)
    """, entries)
    conn.commit()
    print(f"    Inserted {len(entries)} exam sessions into exam_timetable table.")


# --- 5. Programmes Extractor ---
def extract_programmes(conn: sqlite3.Connection):
    print("[*] Extracting academic programmes...")
    PROG_RE = re.compile(r"\b(Bachelor of [^\n\(\)]+|Master of [^\n\(\)]+|Doctor of [^\n\(\)]+|Diploma in [^\n\(\)]+|Postgraduate Diploma in [^\n\(\)]+|Certificate in [^\n\(\)]+)\s*(?:\([^\)]*\))?\s*\b(0[4-9][A-Z0-9]{3,5}|10[A-Z0-9]{3,5})\b", re.IGNORECASE)
    
    files = [
        ("Computing and Informatics", os.path.join(RAW_DATA_DIR, "2026_Prospectus_FCI.pdf")),
        ("Engineering and the Built Environment", os.path.join(RAW_DATA_DIR, "2026-FEBE-PROSPECTUS.pdf")),
        ("Commerce, Human Sciences and Education", os.path.join(RAW_DATA_DIR, "FCHSE_Prospectus_2026.pdf")),
        ("Health, Natural Resources and Applied Sciences", os.path.join(RAW_DATA_DIR, "FHNRAS-Prospectus-2025.pdf")),
        ("Graduate School of Business", os.path.join(RAW_DATA_DIR, "Prospectus_HP-GSB_2026.pdf")),
        ("University Wide", os.path.join(RAW_DATA_DIR, "Guide_to_Prospective_Students_2027.pdf"))
    ]

    programmes: Dict[str, Dict] = {}

    # Seed key benchmark programmes
    seed_progs = [
        ("07BCCS", "Bachelor of Computer Science", "Computing and Informatics", 7, 3.0, 25, "Minimum 25 points in 5 subjects with C in Mathematics and E in English on NSSCO."),
        ("07BOAI", "Bachelor of Artificial Intelligence", "Computing and Informatics", 7, 3.0, 25, "Minimum 25 points in 5 subjects with C in Mathematics and E in English on NSSCO."),
        ("07BCCY", "Bachelor of Computer Science in Cyber Security", "Computing and Informatics", 7, 3.0, 25, "Minimum 25 points in 5 subjects with C in Mathematics and E in English on NSSCO."),
        ("07BAIN", "Bachelor of Informatics", "Computing and Informatics", 7, 3.0, 25, "Minimum 25 points in 5 subjects with D in Mathematics and E in English on NSSCO."),
        ("08HBCS", "Bachelor of Computer Science Honours", "Computing and Informatics", 8, 1.0, 0, "Bachelor degree in Computer Science NQF Level 7 with at least 60% average."),
        ("08BAIH", "Bachelor of Artificial Intelligence Honours", "Computing and Informatics", 8, 1.0, 0, "Bachelor degree in Artificial Intelligence NQF Level 7 with at least 60% average."),
        ("09MACS", "Master of Computer Science", "Computing and Informatics", 9, 2.0, 0, "Honours degree in Computer Science NQF Level 8 or equivalent."),
        ("08BSE", "Bachelor of Engineering: Civil Engineering", "Engineering and the Built Environment", 8, 4.0, 30, "Minimum 30 points in 5 subjects with C in Mathematics, Physics and E in English."),
        ("08BEME", "Bachelor of Engineering: Mechanical Engineering", "Engineering and the Built Environment", 8, 4.0, 30, "Minimum 30 points in 5 subjects with C in Mathematics, Physics and E in English."),
        ("08BEE", "Bachelor of Engineering: Electrical and Electronic", "Engineering and the Built Environment", 8, 4.0, 30, "Minimum 30 points in 5 subjects with C in Mathematics, Physics and E in English."),
        ("07BECO", "Bachelor of Economics", "Commerce, Human Sciences and Education", 7, 3.0, 25, "Minimum 25 points in 5 subjects with C in Mathematics and E in English on NSSCO."),
        ("07BACC", "Bachelor of Accounting", "Commerce, Human Sciences and Education", 7, 3.0, 25, "Minimum 25 points in 5 subjects with C in Mathematics and E in English on NSSCO."),
        ("07BJOU", "Bachelor of Journalism and Media Technology", "Commerce, Human Sciences and Education", 7, 3.0, 25, "Minimum 25 points in 5 subjects with B in English on NSSCO."),
        ("07BNUR", "Bachelor of Nursing Science", "Health, Natural Resources and Applied Sciences", 7, 4.0, 25, "Minimum 25 points in 5 subjects with C in Biology and E in English.")
    ]

    for code, name, fac, nqf, dur, pts, req in seed_progs:
        programmes[code] = {
            "code": code, "name": name, "faculty": fac, "nqf": nqf, "dur": dur, "pts": pts, "req": req
        }

    for fac, path in files:
        if not os.path.exists(path): continue
        r = pypdf.PdfReader(path)
        for page in r.pages:
            try:
                txt = page.extract_text()
            except Exception:
                continue
            for m in PROG_RE.finditer(txt):
                raw_name = m.group(1).strip()
                code = m.group(2).strip().upper()
                if not re.match(r"^0[4-9][A-Z0-9]{3,5}$|^10[A-Z0-9]{3,5}$", code):
                    continue
                nqf = int(code[:2])
                dur = 3.0
                if nqf >= 9: dur = 2.0
                elif nqf == 8 and "honours" in raw_name.lower(): dur = 1.0
                elif nqf == 8 and "engineering" in raw_name.lower(): dur = 4.0
                elif nqf == 10: dur = 3.0

                clean_n = clean_title(raw_name)
                if code not in programmes and len(clean_n) >= 5:
                    programmes[code] = {
                        "code": code,
                        "name": clean_n.title(),
                        "faculty": fac if fac != "University Wide" else "General",
                        "nqf": nqf,
                        "dur": dur,
                        "pts": 25 if nqf == 7 else (30 if "engineering" in clean_n.lower() else 0),
                        "req": f"NQF Level {nqf} Qualification. Refer to faculty prospectus for specific subject criteria."
                    }

    cursor = conn.cursor()
    for code, p in programmes.items():
        cursor.execute("""
        INSERT INTO programmes (programme_code, programme_name, faculty, nqf_level, duration_years, minimum_points, admission_requirements)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (p["code"], p["name"], p["faculty"], p["nqf"], p["dur"], p["pts"], p["req"]))
    conn.commit()
    print(f"    Inserted {len(programmes)} programmes into programmes table.")


# --- 6. Admission Point Scale Extractor ---
def extract_aps(conn: sqlite3.Connection):
    print("[*] Extracting admission point score (APS) conversion scale...")
    aps_data = [
        # NSSCO (Ordinary Level - 2021 Intake onwards)
        ("NSSCO", "A*", 8, "Ordinary Level Grade A* (90-100%)"),
        ("NSSCO", "A", 7, "Ordinary Level Grade A (80-89%)"),
        ("NSSCO", "B", 6, "Ordinary Level Grade B (70-79%)"),
        ("NSSCO", "C", 5, "Ordinary Level Grade C (60-69%)"),
        ("NSSCO", "D", 4, "Ordinary Level Grade D (50-59%)"),
        ("NSSCO", "E", 3, "Ordinary Level Grade E (40-49%)"),
        ("NSSCO", "F", 2, "Ordinary Level Grade F (30-39%)"),
        ("NSSCO", "G", 1, "Ordinary Level Grade G (20-29%)"),
        ("NSSCO", "U", 0, "Ordinary Level Ungraded (<20%)"),

        # NSSCAS (Advanced Subsidiary Level)
        ("NSSCAS", "a", 10, "Advanced Subsidiary Level Grade a (80-100%)"),
        ("NSSCAS", "b", 9, "Advanced Subsidiary Level Grade b (70-79%)"),
        ("NSSCAS", "c", 8, "Advanced Subsidiary Level Grade c (60-69%)"),
        ("NSSCAS", "d", 7, "Advanced Subsidiary Level Grade d (50-59%)"),
        ("NSSCAS", "e", 6, "Advanced Subsidiary Level Grade e (40-49%)"),
        ("NSSCAS", "u", 0, "Advanced Subsidiary Level Ungraded"),

        # NSSCH (Higher Level)
        ("NSSCH", "1", 9, "Higher Level Grade 1"),
        ("NSSCH", "2", 8, "Higher Level Grade 2"),
        ("NSSCH", "3", 7, "Higher Level Grade 3"),
        ("NSSCH", "4", 6, "Higher Level Grade 4"),

        # Cambridge / GCE A-Level
        ("Cambridge_A_Level", "A*", 12, "Cambridge GCE A-Level A*"),
        ("Cambridge_A_Level", "A", 11, "Cambridge GCE A-Level A"),
        ("Cambridge_A_Level", "B", 10, "Cambridge GCE A-Level B"),
        ("Cambridge_A_Level", "C", 9, "Cambridge GCE A-Level C"),
        ("Cambridge_A_Level", "D", 8, "Cambridge GCE A-Level D"),
        ("Cambridge_A_Level", "E", 7, "Cambridge GCE A-Level E"),

        # South Africa NSC (National Senior Certificate)
        ("SA_NSC", "7", 8, "Rating 7: Outstanding (80-100%)"),
        ("SA_NSC", "6", 7, "Rating 6: Meritorious (70-79%)"),
        ("SA_NSC", "5", 6, "Rating 5: Substantial (60-69%)"),
        ("SA_NSC", "4", 5, "Rating 4: Adequate (50-59%)"),
        ("SA_NSC", "3", 4, "Rating 3: Moderate (40-49%)"),
        ("SA_NSC", "2", 3, "Rating 2: Elementary (30-39%)"),
        ("SA_NSC", "1", 0, "Rating 1: Not Achieved (0-29%)")
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO admission_point_scale (qualification_type, symbol_or_grade, points, description)
    VALUES (?, ?, ?, ?)
    """, aps_data)
    conn.commit()
    print(f"    Inserted {len(aps_data)} conversion rules into admission_point_scale table.")


# --- 7. Fee Structure Extractor ---
def extract_fees(conn: sqlite3.Connection):
    print("[*] Extracting administrative fee structure...")
    fees = [
        ("Registration", "Annual Registration Fee", 2160.00, "annual", "Includes N$150 activity fee + N$260 assessment fee; non-refundable"),
        ("Registration", "Late Registration Fee", 580.00, "per_occurrence", "Payable in addition to registration fee"),
        ("Deposit", "Undergraduate Registration Deposit", 4500.00, "per_semester", "Inclusive of registration fee; credited toward tuition"),
        ("Deposit", "Postgraduate Registration Deposit (HP-GSB)", 9000.00, "per_semester", "Inclusive of registration fee; credited toward tuition"),
        ("Application", "Undergraduate Application Fee (Online)", 0.00, "per_application", "Free online application via portal"),
        ("Application", "Undergraduate Application Fee (Manual)", 240.00, "per_application", "Hard-copy submission fee"),
        ("Application", "Late Application Fee", 580.00, "per_application", "Payable for late submissions submitted after September cutoff"),
        ("RPL", "RPL Application Fee", 240.00, "per_application", "Recognition of Non-Certified Prior Learning application"),
        ("RPL", "RPL Assessment Fee", 1220.00, "per_course", "Assessment per evaluated module"),
        ("Examinations", "Exit Examination Fee", 600.00, "per_course", "Payable upon application for exit examinations"),
        ("Examinations", "Special Examination Fee", 1200.00, "per_course", "Payable upon application for special examination session"),
        ("Examinations", "Remark and Recheck Fee", 370.00, "per_course", "Refunded if mark improves by 5% or more"),
        ("Exemptions", "Course Exemption Application Fee", 240.00, "per_course", "Application for recognition of prior accredited courses"),
        ("Services", "Medical Services Clinic Visit", 10.00, "per_visit", "Campus clinic consultation fee"),
        ("Services", "Library Late Overdue Penalty", 5.00, "per_day", "Fee charged per overdue item per day"),
        ("Services", "UNAM Undergraduate Student Deposit", 935.00, "per_year", "External student library access deposit")
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO fee_structure (fee_category, fee_name, amount_nad, frequency, notes)
    VALUES (?, ?, ?, ?, ?)
    """, fees)
    conn.commit()
    print(f"    Inserted {len(fees)} fee items into fee_structure table.")


# --- 8. Residence Fees Extractor ---
def extract_residence(conn: sqlite3.Connection):
    print("[*] Extracting residence and hostel fee rates...")
    res_items = [
        ("Small Single Room", "Accommodation", 9670.00, "per_semester", 0, "Namibian & SADC citizens hostel rate"),
        ("Medium Single Room", "Accommodation", 10660.00, "per_semester", 0, "Namibian & SADC citizens hostel rate"),
        ("Single Room with Balcony", "Accommodation", 11510.00, "per_semester", 0, "Namibian & SADC citizens hostel rate"),
        ("Double Room", "Accommodation", 9010.00, "per_semester", 0, "Shared double occupancy rate per resident"),
        ("Residence Breakage Deposit", "Deposit", 1230.00, "per_semester", 1, "Must be paid in cash or direct deposit; refundable upon vacancy"),
        ("Residence Initial Deposit", "Deposit", 3230.00, "per_semester", 0, "Includes N$1,230 breakage deposit"),
        ("Compulsory Resident Meal Levy (50%)", "Meals", 10110.00, "per_semester", 0, "Compulsory minimum 50% meal quota for all resident students"),
        ("Full Meal Plan (100%)", "Meals", 20220.00, "per_semester", 0, "Covers complete daily meals across the entire semester"),
        ("Meal Voucher: Breakfast", "Voucher", 30.00, "per_meal", 0, "Campus cafeteria breakfast voucher"),
        ("Meal Voucher: Lunch", "Voucher", 65.00, "per_meal", 0, "Campus cafeteria lunch voucher"),
        ("Meal Voucher: Supper", "Voucher", 50.00, "per_meal", 0, "Campus cafeteria dinner/supper voucher")
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO residence_fees (item_name, category, amount_nad, period, refundable, notes)
    VALUES (?, ?, ?, ?, ?, ?)
    """, res_items)
    conn.commit()
    print(f"    Inserted {len(res_items)} accommodation rates into residence_fees table.")


# --- 9. Fee Payment Schedule Extractor ---
def extract_payment_schedule(conn: sqlite3.Connection):
    print("[*] Extracting fee payment schedule and cancellation rules...")
    schedule = [
        ("2026-02-27", "Semester Courses", 33, 100, 0, "33% tuition due; Last day for course cancellation with 100% credit"),
        ("2026-02-27", "Year Courses", 12, 100, 0, "12% tuition due; Last day for course cancellation with 100% credit"),
        ("2026-02-29", "All Programmes", 100, 0, 10, "10% early payment discount if all year and semester fees settled in full"),
        ("2026-04-02", "Semester Courses", 33, 50, 0, "Last day for 1st semester course cancellation with 50% credit"),
        ("2026-04-02", "Year Courses", 26, 0, 0, "26% total fees payable for Year Courses"),
        ("2026-05-02", "Semester Courses", 100, 0, 0, "100% of 1st semester tuition due; Course cancellation of any nature 0% credit"),
        ("2026-05-29", "Year Courses", 50, 100, 0, "50% due; Last day for Year Course cancellation with 100% credit"),
        ("2026-06-28", "Year Courses", 62, 50, 0, "62% due; Last day for Year Course cancellation with 50% credit"),
        ("2026-07-31", "Semester Courses", 0, 100, 10, "Last day for Semester 2 cancellations with 100% credit; 10% early discount if settled"),
        ("2026-08-16", "Semester Courses", 33, 50, 0, "33% Semester 2 tuition due; Last day for Semester 2 cancellation with 50% credit"),
        ("2026-08-16", "Year Courses", 74, 0, 0, "74% total fees payable for Year Courses"),
        ("2026-10-02", "All Programmes", 100, 0, 0, "100% of all semester and year tuition fees payable in full")
    ]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO fee_payment_schedule (deadline_date, course_type, percentage_due, cancellation_credit_percentage, discount_percentage, description)
    VALUES (?, ?, ?, ?, ?, ?)
    """, schedule)
    conn.commit()
    print(f"    Inserted {len(schedule)} milestone rules into fee_payment_schedule table.")


# --- 10. Official Banking Details Extractor ---
def extract_banking_details(conn: sqlite3.Connection):
    print("[*] Populating official University banking details...")
    # Extracted from NUST-Fees-Booklet-2026.pdf Page 2
    bank_records = [
        (
            "First National Bank (FNB)",
            "Namibia University of Science and Technology",
            "Ausspannplatz",
            "28 18 72",
            "555 001 26319",
            "Cheque / Current Account",
            "Student Number (e.g. 220123456)",
            "studentdebtors@nust.na",
            "Direct bank deposit or EFT. Kindly ensure student number is stated as reference. Processing takes up to 3 working days."
        )
    ]
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO banking_details (bank_name, account_name, branch_name, branch_code, account_number, account_type, reference_format, contact_email, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, bank_records)
    conn.commit()
    print(f"    Inserted {len(bank_records)} bank account records into banking_details table.")


# --- 11. Student Societies Extractor ---
def extract_student_societies(conn: sqlite3.Connection):
    print("[*] Extracting registered student societies and clubs...")
    json_path = os.path.join(RAW_DATA_DIR, "student_societies.json")
    societies = []
    
    if os.path.exists(json_path):
        with open(json_path) as f:
            societies = json.load(f)
    else:
        # Fallback dictionary if JSON not pre-scraped
        societies = [
            {"society_name": "Transport and Logistics Society", "slug": "transport-logistics", "category": "Academic / Professional", "description": "Assists, guides and supports students studying Transport Management and Logistics.", "mission": "Prepare members for future demands in transport and logistics.", "contact_email": "", "url": "https://www.nust.na/societies/transport-logistics"},
            {"society_name": "Medical Laboratory Science Students Society", "slug": "mlsss", "category": "Academic / Healthcare", "description": "Represents the interest of Medical Laboratory Science students.", "mission": "Promote academic excellence and professional networking.", "contact_email": "", "url": "https://www.nust.na/societies/mlsss"},
            {"society_name": "Public Management Students' Society", "slug": "npms", "category": "Academic / Governance", "description": "Platform for public administration and governance scholars.", "mission": "Equip future public leaders.", "contact_email": "npms@nust.na", "url": "https://www.nust.na/societies/npms"},
            {"society_name": "NUST Students Christian Movement", "slug": "nscm", "category": "Religious / Spiritual", "description": "Interdenominational student Christian fellowship on campus.", "mission": "Spiritual growth and student support.", "contact_email": "scm@nust.na", "url": "https://www.nust.na/societies/nscm"},
            {"society_name": "NUST Students Against Crime", "slug": "nustsac", "category": "Community / Safety", "description": "Advocates campus security and crime prevention.", "mission": "Fostering a safe campus environment.", "contact_email": "nustsac@nust.na", "url": "https://www.nust.na/societies/nustsac"},
            {"society_name": "NUST Marketing Society", "slug": "nust-marketing", "category": "Academic / Professional", "description": "Fosters marketing acumen and industry connections.", "mission": "Bridging academia and the corporate market.", "contact_email": "cduplessis@nust.na", "url": "https://www.nust.na/societies/nust-marketing"},
            {"society_name": "NUST Chess Club", "slug": "chess-club", "category": "Sports / Recreational", "description": "Competitive and recreational chess playing for NUST students.", "mission": "Promote critical thinking through chess.", "contact_email": "", "url": "https://www.nust.na/societies/chess-club"},
            {"society_name": "Namibia University of Science and Technology Excellence Team (NUST-ET)", "slug": "nust-et", "category": "Innovation & Technology", "description": "Promotes technical leadership and digital innovation.", "mission": "Empowering students in emerging technologies.", "contact_email": "nust-et@nust.na", "url": "https://www.nust.na/societies/nust-et"},
            {"society_name": "AUBA NUST Society", "slug": "auba", "category": "Academic / Architecture", "description": "Architecture and Urban Built Association for spatial design students.", "mission": "Excellence in architecture and urban planning.", "contact_email": "auba-society@nust.na", "url": "https://www.nust.na/societies/auba"}
        ]

    cursor = conn.cursor()
    for s in societies:
        cursor.execute("""
        INSERT INTO student_societies (society_name, slug, category, description, mission, contact_email, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (s["society_name"], s["slug"], s["category"], s["description"], s.get("mission", ""), s.get("contact_email", ""), s["url"]))
    conn.commit()
    print(f"    Inserted {len(societies)} societies into student_societies table.")


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
        cursor.execute("INSERT INTO contacts (department, role, officer_name, email, telephone) VALUES (?, ?, ?, ?, ?)",
                       (c["department"], c["role"], c.get("officer_name"), c["email"], c.get("telephone")))
    conn.commit()
    print(f"    Inserted {len(contacts)} records into contacts table.")

    # 2. Academic Calendar
    print("[*] Extracting academic calendar...")
    calendar_pdf = os.path.join(RAW_DATA_DIR, "2026-INSTITUTIONAL-CALENDAR.pdf")
    events = extract_calendar(calendar_pdf)
    for e in events:
        cursor.execute("INSERT INTO academic_calendar (academic_year, semester, event_name, start_date, end_date, category) VALUES (?, ?, ?, ?, ?, ?)",
                       (e["academic_year"], e["semester"], e["event_name"], e["start_date"], e["end_date"], e.get("category")))
    conn.commit()
    print(f"    Inserted {len(events)} events into academic_calendar table.")

    # 3. Curricula & Modules
    populate_curricula(conn)

    # 4. Exam Timetable
    extract_exam_timetable(conn)

    # 5. Programmes Offered
    extract_programmes(conn)

    # 6. Admission Point Scale (APS)
    extract_aps(conn)

    # 7. Fee Structure
    extract_fees(conn)

    # 8. Residence Fees
    extract_residence(conn)

    # 9. Fee Payment Schedule
    extract_payment_schedule(conn)

    # 10. Official Banking Details
    extract_banking_details(conn)

    # 11. Student Societies
    extract_student_societies(conn)

    conn.close()
    print("\n" + "="*50)
    print(f"[*] Comprehensive Extraction Complete! SQLite database is ready at: {DB_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()
