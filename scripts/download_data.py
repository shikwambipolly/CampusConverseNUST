#!/usr/bin/env python3
"""
scripts/download_data.py

Automated data acquisition script for CampusConverse (NUST).
Downloads institutional prospectuses, calendars, fee guides, contacts,
and regulations from https://www.nust.na/documents-and-forms into raw_data/.
Generates raw_data/manifest.json with metadata and classification targets.
"""

import json
import os
import sys
import time
from urllib.parse import urljoin, unquote
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.nust.na"
DOCS_URL = "https://www.nust.na/documents-and-forms"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw_data"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Explicit categories to map known critical documents
DOCUMENT_ROLES = {
    # Calendars (Deterministic tool lookup)
    "2026-INSTITUTIONAL-CALENDAR.pdf": "deterministic_calendar",
    "June_2026_Exam_Timetable.pdf": "deterministic_calendar",
    
    # Prospectuses / Yearbooks (Both Deterministic Module Catalog + Unstructured Faculty Policies)
    "2026_Prospectus_FCI.pdf": "deterministic_catalog_and_policy",
    "2026-FEBE-PROSPECTUS.pdf": "deterministic_catalog_and_policy",
    "FCHSE_Prospectus_2026.pdf": "deterministic_catalog_and_policy",
    "FHNRAS-Prospectus-2025.pdf": "deterministic_catalog_and_policy",
    "Prospectus_HP-GSB_2026.pdf": "deterministic_catalog_and_policy",
    
    # Fees and Financials (Deterministic tool lookup & policy)
    "NUST-Fees-Booklet-2026.pdf": "deterministic_fees",
    "CLAIMS-PROCEDURES.pdf": "unstructured_policy",
    "payment_plan_2022.pdf": "deterministic_fees",
    
    # Contacts (Deterministic lookup)
    "Faculty_Officers_Info.pdf": "deterministic_contacts",
    "STUDENT SUPPORT OFFICERS WORKLOAD -SEMESTER 2 of 2026  updated  13 July 2026.pdf": "deterministic_contacts",
    
    # General Rules and Regulations (Unstructured RAG)
    "2026_GENERAL_INFORMATION_AND_REGULATIONS.pdf": "unstructured_policy",
    "REGISTRATION_GUIDE_2025.pdf": "unstructured_policy",
    "Guide_to_Prospective_Students_2027.pdf": "unstructured_policy",
    "2027-Pocket-Guide-Postgraduate.pdf": "unstructured_policy",
    "Guidelines-for-the-Supervision-and-Examination-of-Postgraduate-Degrees.pdf": "unstructured_policy",
    "NUST_ADMISSION_SCALE.pdf": "unstructured_policy",
    "SRC_Constitution.pdf": "unstructured_policy",
    "EMC-Selections-Information-Booklet-2026.pdf": "unstructured_policy",
    "Graduation-Procedures_042026.pdf": "unstructured_policy",
    
    # Academic Forms / Procedures (Unstructured RAG / Reference)
    "Exit-Examinations-Application-Form.pdf": "academic_form_policy",
    "Special-Examinations-Application-Form.pdf": "academic_form_policy",
    "APPLICATION-FOR-COURSE-RECOGNITION-EXEMPTION-FORM.pdf": "academic_form_policy",
    "APPLICATION-FOR-REMARK-AND-RECHECK-2017.pdf": "academic_form_policy",
    "Recognition_of_prior_learning.pdf": "academic_form_policy",
    "RESUMPTION_OF_STUDIES_APPLICATION_FORM_2024.pdf": "academic_form_policy",
}

# Documents to explicitly exclude (procurement tenders, old annual reports)
EXCLUDE_SUBSTRINGS = [
    "Annual_Report", "Research_Report", "Procurement", "INCOTERMS",
    "General-Conditions", "GCC-Non-Consultancy", "Vendor-Registration",
    "Written-Undertaking", "Graphic_Design", "Citation_"
]

def sanitize_filename(filename: str) -> str:
    decoded = unquote(filename)
    clean = decoded.replace("%20", "_").replace(" ", "_").replace("%28", "(").replace("%29", ")").replace("%2520", "_")
    return clean

def should_exclude(title: str, url: str) -> bool:
    combined = f"{title} {url}".lower()
    for excl in EXCLUDE_SUBSTRINGS:
        if excl.lower() in combined:
            return True
    return False

def discover_documents():
    print(f"[*] Fetching documents page: {DOCS_URL}")
    resp = requests.get(DOCS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    discovered = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue
        
        full_url = urljoin(BASE_URL, href)
        title = a.get_text(strip=True) or os.path.basename(href)
        
        # Determine category / section
        parent_h = a.find_previous(["h2", "h3", "h4", "h5", "h6"])
        section = parent_h.get_text(strip=True) if parent_h else "General"
        
        if should_exclude(title, href):
            continue
            
        raw_name = os.path.basename(href.split("?")[0])
        clean_name = sanitize_filename(raw_name)
        
        # Match target role
        target_role = DOCUMENT_ROLES.get(unquote(raw_name))
        if not target_role:
            target_role = DOCUMENT_ROLES.get(clean_name, "unstructured_policy")
            
        discovered.append({
            "title": title,
            "section": section,
            "url": full_url,
            "raw_filename": raw_name,
            "filename": clean_name,
            "target_role": target_role
        })
        
    # Deduplicate by clean filename
    unique_docs = {}
    for item in discovered:
        if item["filename"] not in unique_docs:
            unique_docs[item["filename"]] = item
            
    return list(unique_docs.values())

def download_file(url: str, dest_path: str):
    print(f"    Downloading: {os.path.basename(dest_path)}...")
    resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs = discover_documents()
    print(f"[*] Discovered {len(docs)} academic documents to acquire.")
    
    manifest = []
    success_count = 0
    fail_count = 0
    
    for doc in docs:
        dest_path = os.path.join(OUTPUT_DIR, doc["filename"])
        print(f"[{doc['target_role']}] {doc['title']}")
        
        try:
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
                download_file(doc["url"], dest_path)
                time.sleep(0.5)
            else:
                print(f"    Already exists ({os.path.getsize(dest_path):,} bytes). Skipping.")
                
            size = os.path.getsize(dest_path)
            doc["file_size_bytes"] = size
            doc["local_path"] = os.path.relpath(dest_path, start=os.path.dirname(OUTPUT_DIR))
            doc["status"] = "downloaded"
            manifest.append(doc)
            success_count += 1
        except Exception as e:
            print(f"    [!] Error downloading {doc['url']}: {e}")
            doc["status"] = f"failed: {str(e)}"
            manifest.append(doc)
            fail_count += 1

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "="*50)
    print(f"[*] Ingestion Complete:")
    print(f"    Successfully downloaded/verified: {success_count}")
    print(f"    Failures: {fail_count}")
    print(f"    Manifest written to: {manifest_path}")
    print("="*50)

if __name__ == "__main__":
    main()
