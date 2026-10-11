"""
src/tools.py

Deterministic Database Tool-Calling Engine for CampusConverse NUST.
Adheres strictly to spec.md Section 3.4:
1. lookup_prerequisite(module_code: str) -> dict
2. lookup_calendar_date(event_name: str, semester: Optional[int]) -> dict
3. lookup_contact_info(department: str) -> dict
"""

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "nust_facts.db")

# Regex pattern for NUST course codes (e.g., DSA611S, PRG510S, SYD611S, MTE311S)
COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{3}[0-9]{3}[A-Z0-9]|[A-Z]{3}[0-9]{4})\b", re.IGNORECASE)


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row dictionary access"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_module_code(text: str) -> Optional[str]:
    """Extracts a valid NUST module code from query text"""
    match = COURSE_CODE_PATTERN.search(text)
    return match.group(1).upper() if match else None


def extract_semester(text: str) -> Optional[int]:
    """Extracts target academic semester (1 or 2) from query text"""
    lower = text.lower()
    if re.search(r"\b(?:semester\s*1|sem\s*1|s1|first\s+semester)\b", lower):
        return 1
    if re.search(r"\b(?:semester\s*2|sem\s*2|s2|second\s+semester)\b", lower):
        return 2
    return None


def lookup_prerequisite(query_or_code: str) -> Dict[str, Any]:
    """
    Looks up module prerequisites and corequisites from module_catalog table.
    Returns:
        {
            "found": bool,
            "module_code": str or None,
            "module_name": str or None,
            "prerequisites": str or None,
            "corequisites": str or None,
            "message": str
        }
    """
    code = extract_module_code(query_or_code) or query_or_code.strip().upper()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Exact match on module_code
    cursor.execute(
        "SELECT module_code, module_name, faculty, department, prerequisites, corequisites "
        "FROM module_catalog WHERE UPPER(module_code) = ?",
        (code,)
    )
    row = cursor.fetchone()

    # If not found by exact code, search by partial module name
    if not row:
        cursor.execute(
            "SELECT module_code, module_name, faculty, department, prerequisites, corequisites "
            "FROM module_catalog WHERE LOWER(module_name) LIKE ? LIMIT 1",
            (f"%{query_or_code.strip().lower()}%",)
        )
        row = cursor.fetchone()

    conn.close()

    if row:
        mod_code = row["module_code"]
        mod_name = row["module_name"]
        prereqs = row["prerequisites"] if row["prerequisites"] else "None (No prerequisite required)"
        coreqs = row["corequisites"] if row["corequisites"] else "None"

        message = (
            f"Official Prerequisite Details for {mod_code} ({mod_name}):\n"
            f"- Prerequisites: {prereqs}\n"
            f"- Corequisites: {coreqs}\n"
            f"- Faculty: {row['faculty']} (Department: {row['department']})"
        )
        return {
            "found": True,
            "module_code": mod_code,
            "module_name": mod_name,
            "faculty": row["faculty"],
            "department": row["department"],
            "prerequisites": prereqs,
            "corequisites": coreqs,
            "message": message
        }

    return {
        "found": False,
        "module_code": code,
        "module_name": None,
        "prerequisites": None,
        "corequisites": None,
        "message": f"Module '{query_or_code.strip()}' was not found in the NUST module catalog. Please verify the course code."
    }


def lookup_calendar_date(query: str, semester: Optional[int] = None) -> Dict[str, Any]:
    """
    Looks up academic calendar dates (lectures, exams, recesses, registration, appeals).
    If a course code is present, it also cross-references the exam_timetable.
    """
    sem = semester if semester is not None else extract_semester(query)
    code = extract_module_code(query)
    conn = get_db_connection()
    cursor = conn.cursor()

    # If query mentions a specific module exam date
    if code:
        cursor.execute(
            "SELECT course_code, course_title, exam_date, session_time, duration_minutes, session_opportunity "
            "FROM exam_timetable WHERE UPPER(course_code) = ?",
            (code,)
        )
        exam_rows = cursor.fetchall()
        if exam_rows:
            conn.close()
            details = []
            for r in exam_rows:
                details.append(
                    f"• {r['course_code']} ({r['course_title']}): {r['exam_date']} at {r['session_time']} "
                    f"[{r['session_opportunity']}, {r['duration_minutes']} mins]"
                )
            return {
                "found": True,
                "category": "exam_timetable",
                "events": [dict(r) for r in exam_rows],
                "message": f"Official Exam Timetable for {code}:\n" + "\n".join(details)
            }

    # Keyword extraction for academic calendar
    lower_query = query.lower()
    keywords = []
    if "commence" in lower_query or "start" in lower_query or "begin" in lower_query or "lecture" in lower_query:
        keywords.extend(["commence", "start", "lecture"])
    if "recess" in lower_query or "break" in lower_query or "holiday" in lower_query:
        keywords.extend(["recess", "break"])
    if "exam" in lower_query or "examination" in lower_query or "test" in lower_query:
        keywords.extend(["examination", "exam", "test"])
    if "registration" in lower_query or "register" in lower_query:
        keywords.extend(["registration", "register"])
    if "appeal" in lower_query or "remark" in lower_query:
        keywords.extend(["appeal", "remark"])
    if "graduation" in lower_query:
        keywords.extend(["graduation"])

    sql = "SELECT academic_year, semester, event_name, start_date, end_date, category FROM academic_calendar WHERE 1=1"
    params = []

    if sem:
        sql += " AND (semester = ? OR semester IS NULL)"
        params.append(sem)

    if keywords:
        kw_clauses = " OR ".join(["LOWER(event_name) LIKE ?" for _ in keywords])
        sql += f" AND ({kw_clauses})"
        params.extend([f"%{kw}%" for kw in keywords])

    sql += " ORDER BY start_date ASC LIMIT 5"
    cursor.execute(sql, params)
    rows = cursor.fetchall()

    # Fallback: if no specific keyword matched, grab general semester milestones
    if not rows and sem:
        cursor.execute(
            "SELECT academic_year, semester, event_name, start_date, end_date, category "
            "FROM academic_calendar WHERE semester = ? ORDER BY start_date ASC LIMIT 5",
            (sem,)
        )
        rows = cursor.fetchall()

    conn.close()

    if rows:
        events = [dict(r) for r in rows]
        lines = []
        for e in events:
            date_str = e["start_date"]
            if e["end_date"] and e["end_date"] != e["start_date"]:
                date_str += f" to {e['end_date']}"
            sem_str = f"Semester {e['semester']}" if e['semester'] else "Full Year"
            lines.append(f"• {e['event_name']} ({sem_str}): {date_str}")

        return {
            "found": True,
            "category": "academic_calendar",
            "events": events,
            "message": "Official NUST Academic Calendar Milestones:\n" + "\n".join(lines)
        }

    return {
        "found": False,
        "category": "academic_calendar",
        "events": [],
        "message": "No specific calendar dates were found matching your inquiry. Please specify the semester or milestone."
    }


def lookup_contact_info(query: str) -> Dict[str, Any]:
    """
    Looks up contact info (email, telephone, officer, office) from contacts table.
    """
    lower = query.lower()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Campus regional centres check
    centres = [
        "katima", "rundu", "walvis", "gobabis", "keetmanshoop", "luderitz",
        "lüderitz", "tsumeb", "outapi", "ongwediva", "eenhana", "rietfontein", "otjiwarongo"
    ]
    matched_centre = None
    for c in centres:
        if c in lower:
            matched_centre = "luderitz" if c == "lüderitz" else c
            break

    if matched_centre:
        cursor.execute(
            "SELECT department, role, officer_name, email, telephone FROM contacts "
            "WHERE LOWER(department) LIKE ? OR LOWER(role) LIKE ?",
            (f"%{matched_centre}%", f"%{matched_centre}%")
        )
        rows = cursor.fetchall()
    else:
        # Check departments / services (Library, Clinic, Counselling, Admissions, Exams, Fees, DSS)
        search_terms = []
        if "library" in lower:
            search_terms.append("library")
        if "clinic" in lower or "health" in lower or "nurse" in lower or "doctor" in lower:
            search_terms.append("clinic")
        if "counsel" in lower or "psycholog" in lower or "mental" in lower:
            search_terms.append("counselling")
        if "exam" in lower:
            search_terms.append("examination")
        if "fee" in lower or "debt" in lower or "account" in lower:
            search_terms.append("student account")
        if "admission" in lower or "apply" in lower:
            search_terms.append("admission")
        if "src" in lower:
            search_terms.append("src")

        if search_terms:
            clauses = " OR ".join(["LOWER(department) LIKE ? OR LOWER(role) LIKE ?" for _ in search_terms])
            params = []
            for t in search_terms:
                params.extend([f"%{t}%", f"%{t}%"])
            cursor.execute(f"SELECT department, role, officer_name, email, telephone FROM contacts WHERE {clauses}", params)
            rows = cursor.fetchall()
        else:
            # Fallback general query
            cursor.execute(
                "SELECT department, role, officer_name, email, telephone FROM contacts "
                "WHERE LOWER(department) LIKE ? OR LOWER(role) LIKE ? OR LOWER(officer_name) LIKE ? LIMIT 5",
                (f"%{lower}%", f"%{lower}%", f"%{lower}%")
            )
            rows = cursor.fetchall()

    conn.close()

    if rows:
        contacts = [dict(r) for r in rows]
        lines = []
        for c in contacts:
            officer = f" ({c['officer_name']})" if c['officer_name'] else ""
            lines.append(
                f"• {c['department']} - {c['role']}{officer}:\n"
                f"  Email: {c['email'] or 'N/A'} | Tel: {c['telephone'] or 'N/A'}"
            )
        return {
            "found": True,
            "contacts": contacts,
            "message": "NUST Official Campus Contacts:\n" + "\n".join(lines)
        }

    return {
        "found": False,
        "contacts": [],
        "message": "No matching campus contact details found. You can reach the NUST Main Campus Switchboard at +264 61 207 9111 or info@nust.na."
    }
