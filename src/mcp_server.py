"""
src/mcp_server.py
=================
Purpose:
    Implements a Model Context Protocol (MCP) server using FastMCP.
    Exposes semantic retrieval tools for Chroma (unstructured RAG) and
    AlloyDB / SQLite (structured deterministic tables).
    Runs as a standalone microservice on Cloud Run (via SSE or HTTP transport).
"""

import os
from typing import Optional
from fastmcp import FastMCP
from src.semantic_layer import SemanticLayer

# 1. Initialize FastMCP server with name
# FastMCP automatically generates JSON Schema parameter definitions from type hints
mcp = FastMCP("campus-converse-mcp-server")

# 2. Instantiate our semantic layer
semantic_layer = SemanticLayer()


# ============================================================================
# Tool 1: Chroma Unstructured Policy & Regulation Search
# ============================================================================
@mcp.tool()
def search_regulations_and_policies(query: str, faculty_filter: Optional[str] = None) -> str:
    """
    Search university regulations, student codes of conduct, general handbooks,
    and academic policies using dense vector semantic search in Chroma DB.

    Args:
        query: Natural language question (e.g. 'What is the minimum lecture attendance requirement?')
        faculty_filter: Optional faculty name to narrow down search (e.g. 'FCI', 'FEBE').
    """
    results = semantic_layer.search_policies_and_regulations(
        query=query,
        n_results=4,
        faculty_filter=faculty_filter
    )

    if not results:
        return "No matching policy or regulation records found in Chroma DB."

    output = []
    for idx, item in enumerate(results, 1):
        output.append(
            f"--- Result {idx} [Source: {item['source_document']} | Section: {item['section_title']} (p. {item['page_number']})] ---\n"
            f"{item['text']}\n"
        )
    return "\n".join(output)


# ============================================================================
# Tool 2: Module & Prerequisite Deterministic Lookup (AlloyDB)
# ============================================================================
@mcp.tool()
def lookup_module_prerequisites(module_code: str) -> str:
    """
    Lookup accurate, deterministic prerequisites, corequisites, and department
    details for a specific university course code in AlloyDB.

    Args:
        module_code: 7-character course code (e.g., 'PRG510S', 'DSA612S', 'WIL420S').
    """
    data = semantic_layer.get_module_prerequisites(module_code)
    if not data:
        return f"Module '{module_code}' was not found in the official curriculum database."

    prereqs = ", ".join(data.get("prerequisites", [])) if data.get("prerequisites") else "None"
    coreqs = ", ".join(data.get("corequisites", [])) if data.get("corequisites") else "None"

    return (
        f"Module Code: {data['module_code']}\n"
        f"Module Name: {data['module_name']}\n"
        f"Faculty: {data['faculty']}\n"
        f"Department: {data['department']}\n"
        f"Prerequisites: {prereqs}\n"
        f"Corequisites: {coreqs}"
    )


# ============================================================================
# Tool 3: Academic Calendar Event Lookup (AlloyDB)
# ============================================================================
@mcp.tool()
def lookup_academic_calendar(event_keyword: str) -> str:
    """
    Look up dates, deadlines, exam timetables, test weeks, and recess dates
    from the official 2026 Institutional Academic Calendar in AlloyDB.

    Args:
        event_keyword: Keyword to search for (e.g., 'registration', 'exam', 'recess', 'test week').
    """
    events = semantic_layer.search_academic_calendar(event_keyword)
    if not events:
        return f"No calendar events found matching keyword: '{event_keyword}'."

    output = ["Official Calendar Dates:"]
    for ev in events:
        output.append(
            f"- [{ev['semester']}] {ev['event_name']}: {ev['start_date']} to {ev['end_date']} (Category: {ev['category']})"
        )
    return "\n".join(output)


# ============================================================================
# Tool 4: Fee Structure, Residence Fees & Banking Lookup (AlloyDB / SQLite)
# ============================================================================
@mcp.tool()
def lookup_fees(fee_category_or_name: str) -> str:
    """
    Look up official administrative rates, registration fees, late penalties,
    breakage deposits, hostel residence fees, payment schedules, and banking details.

    Args:
        fee_category_or_name: Fee type or name (e.g. 'registration', 'hostel', 'penalty', 'remark').
    """
    breakdown = semantic_layer.get_fee_breakdown(fee_category_or_name)
    if not breakdown:
        return f"No fee information found for category: '{fee_category_or_name}'."

    output = ["Official NUST Fee & Payment Breakdown:"]
    for f in breakdown.get("general_fees", []):
        notes = f" | Notes: {f['notes']}" if f.get("notes") else ""
        output.append(f"- {f['fee_name']} ({f['fee_category']}): N$ {f['amount_nad']} ({f['frequency']}){notes}")

    if breakdown.get("residence_fees"):
        output.append("\nHostel & Residence Fees:")
        for r in breakdown["residence_fees"]:
            output.append(f"- {r['item_name']} ({r['category']}): N$ {r['amount_nad']} ({r['period']})")

    if breakdown.get("banking_details"):
        b = breakdown["banking_details"][0]
        output.append(
            f"\nBanking Details: {b['bank_name']} | Account: {b['account_number']} "
            f"(Branch Code: {b['branch_code']}) | Ref: {b['reference_format']}"
        )
    return "\n".join(output)


# ============================================================================
# Tool 5: Programme & Admission Requirements Lookup (AlloyDB / SQLite)
# ============================================================================
@mcp.tool()
def lookup_programmes(keyword: str) -> str:
    """
    Search NUST undergraduate and postgraduate programmes, NQF levels,
    duration, and minimum admission points.

    Args:
        keyword: Programme name, code, or faculty (e.g. 'Computer Science', '07BCMS', 'Engineering').
    """
    progs = semantic_layer.search_programmes(keyword)
    if not progs:
        return f"No programmes found matching '{keyword}'."

    output = ["Official NUST Programmes:"]
    for p in progs:
        output.append(
            f"- {p['programme_code']}: {p['programme_name']} ({p['faculty']}) | "
            f"NQF Level {p['nqf_level']} | {p['duration_years']} yrs | Min Points: {p['minimum_points']} | "
            f"Requirements: {p['admission_requirements']}"
        )
    return "\n".join(output)


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When deployed to Cloud Run, we run FastMCP over Server-Sent Events (SSE) / HTTP
    port = int(os.getenv("PORT", "8080"))
    print(f"[INFO] Starting MCP Server on port {port}...")
    mcp.run(transport="sse", host="0.0.0.0", port=port)
