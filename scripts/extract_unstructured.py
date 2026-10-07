#!/usr/bin/env python3
"""
scripts/extract_unstructured.py

Extracts continuous policy narrative, general academic regulations, student codes of conduct,
degree requirements, fees policy, library operations, and procedural documentation into
structured tokenized chunks for the RAG engine.

Adheres strictly to spec.md Section 2.2:
- Chunk size: ~400 tokens (~1,600 characters)
- Chunk overlap: ~50 tokens (~200 characters)
- Output: data/chunks/corpus_chunks.json
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional
import pypdf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "chunks")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_chunks.json")

# Target chunking parameters
TARGET_CHUNK_CHARS = 1600  # ~400 tokens
CHUNK_OVERLAP_CHARS = 200   # ~50 tokens
MIN_CHUNK_CHARS = 100       # Discard trivial noise fragments

# Corpus documents with designated faculty / domain tagging
CORPUS_DOCUMENTS = [
    # 1. Core Academic Regulations & Governance
    {
        "filename": "2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf",
        "faculty": "Institutional / General",
        "doc_prefix": "GEN_REG"
    },
    {
        "filename": "Final_NUST_Re-aligned_and_Reviewed_Strategic_Plan_2021-2025_0723.pdf",
        "faculty": "Institutional / Strategic Planning",
        "doc_prefix": "STRATEGIC_PLAN"
    },
    {
        "filename": "SRC_Constitution.pdf",
        "faculty": "Student Affairs / Governance",
        "doc_prefix": "SRC_CONST"
    },

    # 2. Faculty Prospectuses
    {
        "filename": "2026_Prospectus_FCI.pdf",
        "faculty": "Computing and Informatics",
        "doc_prefix": "FCI_PROSP"
    },
    {
        "filename": "2026-FEBE-PROSPECTUS.pdf",
        "faculty": "Engineering and the Built Environment",
        "doc_prefix": "FEBE_PROSP"
    },
    {
        "filename": "FCHSE_Prospectus_2026.pdf",
        "faculty": "Commerce, Human Sciences and Education",
        "doc_prefix": "FCHSE_PROSP"
    },
    {
        "filename": "FHNRAS-Prospectus-2025.pdf",
        "faculty": "Health, Natural Resources and Applied Sciences",
        "doc_prefix": "FHNRAS_PROSP"
    },
    {
        "filename": "Prospectus_HP-GSB_2026.pdf",
        "faculty": "Graduate School of Business",
        "doc_prefix": "GSB_PROSP"
    },

    # 3. Admissions, Prospective Guides & Application Rules
    {
        "filename": "Guide_to_Prospective_Students_2027.pdf",
        "faculty": "Institutional / Admissions",
        "doc_prefix": "GUIDE_2027"
    },
    {
        "filename": "2027_eMature_Age_Application_Form.pdf",
        "faculty": "Admissions / Mature Age Entry",
        "doc_prefix": "MATURE_AGE"
    },
    {
        "filename": "2027_eAdvanced_Standing_Application_Form.pdf",
        "faculty": "Admissions / Credit Transfer",
        "doc_prefix": "ADV_STANDING"
    },
    {
        "filename": "2027_eUndergraduate_Application_Form.pdf",
        "faculty": "Admissions / Undergraduate",
        "doc_prefix": "UG_APPLY"
    },
    {
        "filename": "2027-ePostGraduate_Application_Form.pdf",
        "faculty": "Admissions / Postgraduate",
        "doc_prefix": "PG_APPLY"
    },
    {
        "filename": "2027-Pocket-Guide-Postgraduate.pdf",
        "faculty": "Postgraduate Studies",
        "doc_prefix": "PG_POCKET_GUIDE"
    },
    {
        "filename": "Brochure_STEMProgrammes_20223.pdf",
        "faculty": "Admissions / STEM Programmes",
        "doc_prefix": "STEM_BROCHURE"
    },
    {
        "filename": "Notice-current_and_prospective-students.pdf",
        "faculty": "Student Administration",
        "doc_prefix": "STUDENT_NOTICE"
    },

    # 4. Registration, Studies Resumption & Student Administration
    {
        "filename": "REGISTRATION_GUIDE_2025.pdf",
        "faculty": "Student Administration",
        "doc_prefix": "REG_GUIDE"
    },
    {
        "filename": "RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf",
        "faculty": "Student Administration",
        "doc_prefix": "RESUME_STUDIES"
    },
    {
        "filename": "NUST-Luderitz-Satelite-Campus-Flyer.pdf",
        "faculty": "Lüderitz Satellite Campus",
        "doc_prefix": "LUDERITZ_CAMPUS"
    },

    # 5. Postgraduate Supervision & Research
    {
        "filename": "Guidelines-for-the-Supervision-and-Examination-of-Postgraduate-Degrees.pdf",
        "faculty": "Postgraduate Studies",
        "doc_prefix": "POSTGRAD_GUIDE"
    },
    {
        "filename": "REGISTRATION-OF-RESEARCH-TOPICS.pdf",
        "faculty": "Postgraduate Studies",
        "doc_prefix": "RESEARCH_TOPICS"
    },

    # 6. Fees Policy & Student Debtors Narrative
    {
        "filename": "NUST-Fees-Booklet-2026.pdf",
        "faculty": "Student Debtors / Fees Policy",
        "doc_prefix": "FEES_POLICY",
        "max_pages": 14  # Comprehensive policy preamble covering NSFAS, payment rules, debit orders, refunds
    },
    {
        "filename": "CLAIMS-PROCEDURES.pdf",
        "faculty": "Student Debtors",
        "doc_prefix": "CLAIMS_PROC"
    },
    {
        "filename": "Revised_NUST_Refund_Form.pdf",
        "faculty": "Student Debtors",
        "doc_prefix": "REFUND_REG"
    },

    # 7. Accommodation & Hostels
    {
        "filename": "Residence-Application-2027.pdf",
        "faculty": "Student Affairs / Accommodation",
        "doc_prefix": "RES_POLICY"
    },

    # 8. Examinations, Remarks, Exit & Special Exams
    {
        "filename": "Graduation-Procedures_042026.pdf",
        "faculty": "Examinations & Graduation",
        "doc_prefix": "GRAD_PROC"
    },
    {
        "filename": "Updated-Notice-of-Expectation-to-Graduate_0.pdf",
        "faculty": "Examinations & Graduation",
        "doc_prefix": "GRAD_NOTICE"
    },
    {
        "filename": "Exit-Examinations-Application-Form.pdf",
        "faculty": "Examinations",
        "doc_prefix": "EXIT_EXAM"
    },
    {
        "filename": "Special-Examinations-Application-Form.pdf",
        "faculty": "Examinations",
        "doc_prefix": "SPEC_EXAM"
    },
    {
        "filename": "APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf",
        "faculty": "Examinations",
        "doc_prefix": "REMARK_REG"
    },
    {
        "filename": "APPLICATION-FOR-COURSE-RECOGNITION-EXEMPTION-FORM.pdf",
        "faculty": "Admissions & Exemption",
        "doc_prefix": "EXEMPTION_REG"
    },
    {
        "filename": "Recognition_of_prior_learning.pdf",
        "faculty": "Admissions / RPL",
        "doc_prefix": "RPL_POLICY"
    },
    {
        "filename": "EMC-Selections-Information-Booklet-2026.pdf",
        "faculty": "Health, Natural Resources and Applied Sciences",
        "doc_prefix": "EMC_SELECT"
    }
]

def clean_page_text(raw_text: str) -> str:
    """Strips running headers, footers, dotted TOC leaders, and pagination noise"""
    if not raw_text:
        return ""
    lines = raw_text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Discard standalone page numbers: [ 42 ], Page 12 of 100, 15
        if re.match(r"^\[?\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*\]?$", l, re.IGNORECASE):
            continue
        # Discard repeated university running headers
        if any(h in l.lower() for h in [
            "general information and regulations 2026",
            "namibia university of science and technology",
            "guide to prospective students - 2027",
            "faculty officer contact details",
            "prospectus 2026",
            "prospectus 2025",
            "fees booklet 2026",
            "fees booklet 2025"
        ]) and len(l) < 65:
            continue
        # Discard dense table-of-contents dots: e.g. "AC4 ATTENDANCE ............. 45"
        if l.count(".") > 8:
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


def detect_section_title(text: str, fallback_title: str) -> str:
    """Attempts to discover the most prominent regulatory section header on the page"""
    lines = text.split("\n")
    for l in lines[:10]:
        l_str = l.strip()
        # Look for regulatory headers like "AC4 ATTENDANCE OF LECTURES", "SR1 RULES AND CONDUCT", "F.1.3 PAYMENT REQUIREMENTS"
        m = re.match(r"^([A-Z0-9\.\s]{2,8}\s+[A-Za-z\s]{4,75})$", l_str)
        if m:
            return m.group(1).strip()
        # Look for uppercase headings like "ADMISSION REQUIREMENTS", "ASSESSMENT STRATEGIES"
        if l_str.isupper() and 5 <= len(l_str) <= 60 and not any(k in l_str for k in ["TABLE", "PAGE", "NUST", "FACULTY", "CONTENTS"]):
            return l_str.title()
    return fallback_title


def recursive_split_text(text: str, chunk_size: int = TARGET_CHUNK_CHARS, chunk_overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Recursively splits a body of text preserving structural and sentence boundaries"""
    if len(text) <= chunk_size:
        return [text] if len(text.strip()) >= MIN_CHUNK_CHARS else []
    
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    split_sep = " "
    for sep in separators:
        if sep in text:
            split_sep = sep
            break
            
    splits = text.split(split_sep)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for s in splits:
        piece = s + split_sep
        piece_len = len(piece)
        if current_len + piece_len > chunk_size and current_chunk:
            chunk_str = "".join(current_chunk).strip()
            if len(chunk_str) >= MIN_CHUNK_CHARS:
                chunks.append(chunk_str)
            # Reconstruct overlap
            overlap_acc = []
            overlap_len = 0
            for item in reversed(current_chunk):
                overlap_acc.insert(0, item)
                overlap_len += len(item)
                if overlap_len >= chunk_overlap:
                    break
            current_chunk = overlap_acc + [piece]
            current_len = sum(len(x) for x in current_chunk)
        else:
            current_chunk.append(piece)
            current_len += piece_len
            
    if current_chunk:
        chunk_str = "".join(current_chunk).strip()
        if len(chunk_str) >= MIN_CHUNK_CHARS:
            chunks.append(chunk_str)
            
    return chunks


def process_pdf_document(doc_info: Dict) -> List[Dict]:
    filename = doc_info["filename"]
    faculty = doc_info["faculty"]
    prefix = doc_info["doc_prefix"]
    max_pages = doc_info.get("max_pages")
    pdf_path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"[-] File not found: {filename}")
        return []

    print(f"[*] Extracting unstructured narrative from: {filename}...")
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages) if max_pages is None else min(len(reader.pages), max_pages)
    chunks = []
    current_section = f"{faculty} General Information"

    for page_idx in range(total_pages):
        page = reader.pages[page_idx]
        try:
            raw_text = page.extract_text()
            if not raw_text:
                continue
        except Exception as e:
            continue

        clean_text = clean_page_text(raw_text)
        if len(clean_text) < MIN_CHUNK_CHARS:
            continue

        # Skip dense pure-table curriculum pages in prospectuses (handled deterministically by Stage 2A)
        course_code_matches = len(re.findall(r"\b[A-Z]{3}[0-9]{3}[A-Z]\b", clean_text))
        if course_code_matches > 8 and "curriculum" in clean_text.lower():
            continue

        # Update active section context
        current_section = detect_section_title(clean_text, current_section)

        # Split page text into target chunks
        page_chunks = recursive_split_text(clean_text, TARGET_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{prefix}_P{page_idx + 1}_C{chunk_idx + 1}"
            approx_tokens = len(chunk_text.split())
            
            chunks.append({
                "chunk_id": chunk_id,
                "source_document": filename,
                "section_title": current_section,
                "page_number": page_idx + 1,
                "faculty": faculty,
                "approx_tokens": approx_tokens,
                "text": chunk_text
            })

    print(f"    Extracted {len(chunks)} chunks from {filename} ({total_pages} pages processed)")
    return chunks


def process_student_societies() -> List[Dict]:
    """Generates chunks for the 21 registered student societies"""
    json_path = os.path.join(RAW_DATA_DIR, "student_societies.json")
    if not os.path.exists(json_path):
        return []

    print("[*] Chunking student societies profiles...")
    with open(json_path) as f:
        societies = json.load(f)

    chunks = []
    for s in societies:
        slug = s["slug"]
        name = s["society_name"]
        cat = s["category"]
        desc = s["description"]
        mission = s.get("mission", "")
        email = s.get("contact_email", "")
        url = s["url"]

        content = f"STUDENT SOCIETY: {name}\nCategory: {cat}\nOfficial Website: {url}\n"
        if email:
            content += f"Contact Email: {email}\n"
        content += f"Description & Activities:\n{desc}\n"
        if mission:
            content += f"Mission & Objectives:\n{mission}\n"

        chunk_id = f"SOCIETY_{slug.upper().replace('-', '_')}"
        chunks.append({
            "chunk_id": chunk_id,
            "source_document": "student_societies.json",
            "section_title": f"Student Societies: {name}",
            "page_number": 1,
            "faculty": "Student Affairs / Student Life",
            "approx_tokens": len(content.split()),
            "text": content.strip()
        })

    print(f"    Created {len(chunks)} student societies chunks.")
    return chunks


def process_library_and_support_services() -> List[Dict]:
    """Generates official verified policy chunks for NUST Library, Student Health Clinic, and Counselling"""
    print("[*] Generating chunks for NUST Library and Campus Health/Counselling Services...")
    services = [
        {
            "chunk_id": "SERVICE_NUST_LIBRARY_HOURS_RULES",
            "source_document": "library.nust.na (Official Portal)",
            "section_title": "NUST Library Services, Operating Hours and Borrowing Rules",
            "page_number": 1,
            "faculty": "Library Services",
            "text": (
                "NUST LIBRARY SERVICES, OPERATING HOURS AND BORROWING RULES\n"
                "Location: c/o Brahms & Haydn Street, Windhoek West, Windhoek, Namibia\n"
                "Contact: Email: library@nust.na | Telephone: +264 61 207 2022\n\n"
                "Official Operating Hours:\n"
                "- Monday – Friday: 08:00 to 22:00\n"
                "- Saturdays: 09:00 to 17:00\n"
                "- Sundays: 14:00 to 20:00\n"
                "- Student Holidays (Mid-year & End-year recess): Monday – Friday 08:00 to 17:00; Saturdays & Sundays CLOSED\n"
                "- Public Holidays: CLOSED\n\n"
                "Library Services & Facilities:\n"
                "1. Circulation & Borrowing: Undergraduate and postgraduate book loans via student card.\n"
                "2. Overdue Fines: N$ 5.00 per overdue book per day.\n"
                "3. Interlibrary Loans & External Access: UNAM undergraduate access deposit is N$ 935.00 per year.\n"
                "4. Research Commons & E-Resources: Dedicated postgraduate computer lab, Wi-Fi access, digital repository, and access to peer-reviewed academic journals.\n"
                "5. Training & Support: Information literacy workshops, user guides, referencing assistance, and 'Ask a Librarian' reference desk."
            )
        },
        {
            "chunk_id": "SERVICE_STUDENT_HEALTH_CLINIC_COUNSELLING",
            "source_document": "2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf (Section DSS)",
            "section_title": "Department of Student Services: Health Clinic, Counselling and Job Placement",
            "page_number": 75,
            "faculty": "Student Affairs / Student Health",
            "text": (
                "DEPARTMENT OF STUDENT SERVICES: HEALTH CLINIC, COUNSELLING AND SUPPORT\n"
                "Director: Dr Donovan Zealand | Telephone: +264 61 207 2165 / 2270 | Email: dss@nust.na\n\n"
                "1. Campus Health Clinic:\n"
                "- Location: Adjacent to Shangri-La student hostel.\n"
                "- Clinic Consultation Fee: N$ 10.00 per visit.\n"
                "- Operating Hours: Monday, Tuesday, and Thursday: 09:00 - 13:00 and 14:00 - 16:00; Wednesday and Friday: 08:00 - 13:00.\n"
                "- Staffing & Services: Staffed by a full-time qualified registered nurse providing primary healthcare, illness prevention, First Aid training, STI testing, and health wellness workshops.\n\n"
                "2. Student Counselling & Mental Health Services:\n"
                "- Services: Psychological intervention, emotional and social support, individual and group counselling, stress management, relationships, and disability support.\n"
                "- HIV/AIDS Coordination: Free confidential pre- and post-test counselling, peer educator training, and awareness campaigns.\n\n"
                "3. Career Guidance & Job Placement:\n"
                "- Services: CV writing workshops, mock interview preparation, annual Career Fair, and graduate recruitment programmes.\n\n"
                "4. Residential Halls & Dining:\n"
                "- Residences: Three residential halls with 417 total beds (Monresa and Höpker for female students, Shangri-La for male students).\n"
                "- Dining Hall: Managed dining hall providing breakfast (N$ 30), lunch (N$ 65), and dinner (N$ 50)."
            )
        }
    ]

    for s in services:
        s["approx_tokens"] = len(s["text"].split())

    print(f"    Created {len(services)} institutional service chunks.")
    return services


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_chunks = []

    # 1. Process all institutional and faculty PDF documents
    for doc_info in CORPUS_DOCUMENTS:
        doc_chunks = process_pdf_document(doc_info)
        all_chunks.extend(doc_chunks)

    # 2. Process student societies
    society_chunks = process_student_societies()
    all_chunks.extend(society_chunks)

    # 3. Process library and campus health services
    service_chunks = process_library_and_support_services()
    all_chunks.extend(service_chunks)

    # 4. Save to data/chunks/corpus_chunks.json
    print(f"\n[*] Writing {len(all_chunks)} structured corpus chunks to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # 5. Compute and display corpus analytics
    total_tokens = sum(c["approx_tokens"] for c in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    doc_distribution = {}
    for c in all_chunks:
        doc_distribution[c["source_document"]] = doc_distribution.get(c["source_document"], 0) + 1

    print("=" * 60)
    print("COMPREHENSIVE CORPUS EXTRACTION ANALYTICS")
    print("=" * 60)
    print(f"Total Chunks Generated : {len(all_chunks):,}")
    print(f"Total Corpus Word Count : {total_tokens:,} tokens (~{total_tokens*4:,} characters)")
    print(f"Average Chunk Size      : {avg_tokens:.1f} tokens (~{avg_tokens*4:.0f} characters)")
    print(f"Total Source Documents  : {len(doc_distribution)}")
    print(f"Chunks by Source Document:")
    for doc, cnt in sorted(doc_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {doc:60} : {cnt:4d} chunks")
    print("=" * 60)


if __name__ == "__main__":
    main()
