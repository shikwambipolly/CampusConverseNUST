"""
semantic_layer.py
=================
Purpose:
    Acts as the single unified retrieval layer that:
    1. Performs dense semantic vector searches over Chroma DB.
    2. Executes relational/SQL lookups against AlloyDB (PostgreSQL-compatible)
       or a local SQLite fallback for testing.
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

# Standard library SQLite for local facts database (nust_facts.db)
import sqlite3

# Database driver for PostgreSQL / AlloyDB (production GCP deployment)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class SemanticLayer:
    """
    Unified interface to retrieve unstructured semantic chunks (Chroma)
    and structured relational facts (AlloyDB / PostgreSQL).
    """

    def __init__(self):
        # ---------------------------------------------------------
        # 1. Connect to Chroma Vector DB
        # ---------------------------------------------------------
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = "nust_campus_knowledge"

        # Initialize the same embedding function used during indexing
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        if os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true":
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        else:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_func
        )

        # ---------------------------------------------------------
        # 2. Database Connection Credentials for AlloyDB
        # ---------------------------------------------------------
        self.alloydb_host = os.getenv("ALLOYDB_HOST", "127.0.0.1")
        self.alloydb_port = int(os.getenv("ALLOYDB_PORT", "5432"))
        self.alloydb_name = os.getenv("ALLOYDB_DATABASE", "campus_converse")
        self.alloydb_user = os.getenv("ALLOYDB_USER", "postgres")
        self.alloydb_password = os.getenv("ALLOYDB_PASSWORD", "")
        self.sqlite_fallback_path = os.getenv("SQLITE_PATH", "data/nust_facts.db")

    def _get_db_connection(self):
        """
        Creates a connection to AlloyDB (PostgreSQL) if configured,
        otherwise falls back to local SQLite.
        """
        if PSYCOPG2_AVAILABLE and os.getenv("USE_ALLOYDB", "false").lower() == "true":
            # Connect to GCP AlloyDB via VPC peering or AlloyDB Auth Proxy
            return psycopg2.connect(
                host=self.alloydb_host,
                port=self.alloydb_port,
                dbname=self.alloydb_name,
                user=self.alloydb_user,
                password=self.alloydb_password,
                cursor_factory=RealDictCursor
            )
        else:
            # Local fallback using the pre-existing SQLite database
            conn = sqlite3.connect(self.sqlite_fallback_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            return conn

    # =========================================================================
    # UNSTRUCTURED RETRIEVAL: Chroma Vector Search
    # =========================================================================
    def search_policies_and_regulations(
        self,
        query: str,
        n_results: int = 4,
        faculty_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search across university policies, handbooks,
        and regulations indexed in Chroma.
        """
        # Optional metadata filtering by faculty if specified
        where_clause = {"faculty": faculty_filter} if faculty_filter else None

        # Chroma converts 'query' to embedding and searches using HNSW index
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "source_document": meta.get("source_document"),
                    "section_title": meta.get("section_title"),
                    "page_number": meta.get("page_number"),
                    "faculty": meta.get("faculty"),
                    "relevance_distance": round(dist, 4)
                })

        return formatted_results

    # =========================================================================
    # STRUCTURED RETRIEVAL: AlloyDB / Relational SQL Lookups
    # =========================================================================
    def get_module_prerequisites(self, module_code: str) -> Optional[Dict[str, Any]]:
        """
        Queries the module_catalog table for prerequisites, corequisites, and module details.
        """
        clean_code = module_code.strip().upper()
        conn = self._get_db_connection()
        cursor = conn.cursor()

        sql = """
            SELECT module_code, module_name, faculty, department, prerequisites, corequisites
            FROM module_catalog
            WHERE UPPER(module_code) = %s OR UPPER(module_code) LIKE %s
            LIMIT 1;
        """ if (PSYCOPG2_AVAILABLE and os.getenv("USE_ALLOYDB", "false").lower() == "true") else """
            SELECT module_code, module_name, faculty, department, prerequisites, corequisites
            FROM module_catalog
            WHERE UPPER(module_code) = ? OR UPPER(module_code) LIKE ?
            LIMIT 1;
        """

        cursor.execute(sql, (clean_code, f"%{clean_code}%"))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Format output
        data = dict(row)
        # Parse JSON string arrays into native Python lists if needed
        if isinstance(data.get("prerequisites"), str):
            try:
                data["prerequisites"] = json.loads(data["prerequisites"])
            except Exception:
                pass
        if isinstance(data.get("corequisites"), str):
            try:
                data["corequisites"] = json.loads(data["corequisites"])
            except Exception:
                pass
        return data

    def search_academic_calendar(self, event_keyword: str) -> List[Dict[str, Any]]:
        """
        Looks up exam weeks, test dates, registration deadlines, and recess in the calendar table.
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()

        pattern = f"%{event_keyword.strip()}%"
        sql = """
            SELECT academic_year, semester, event_name, start_date, end_date, category
            FROM academic_calendar
            WHERE event_name ILIKE %s OR category ILIKE %s
            ORDER BY start_date ASC;
        """ if (PSYCOPG2_AVAILABLE and os.getenv("USE_ALLOYDB", "false").lower() == "true") else """
            SELECT academic_year, semester, event_name, start_date, end_date, category
            FROM academic_calendar
            WHERE event_name LIKE ? OR category LIKE ?
            ORDER BY start_date ASC;
        """

        cursor.execute(sql, (pattern, pattern))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_fee_breakdown(self, category_keyword: str) -> List[Dict[str, Any]]:
        """
        Queries official tuition, hostel, registration, and administrative fee tables.
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()

        pattern = f"%{category_keyword.strip()}%"
        sql = """
            SELECT fee_category, fee_name, amount_nad, frequency, notes
            FROM fee_structure
            WHERE fee_category ILIKE %s OR fee_name ILIKE %s;
        """ if (PSYCOPG2_AVAILABLE and os.getenv("USE_ALLOYDB", "false").lower() == "true") else """
            SELECT fee_category, fee_name, amount_nad, frequency, notes
            FROM fee_structure
            WHERE fee_category LIKE ? OR fee_name LIKE ?;
        """

        cursor.execute(sql, (pattern, pattern))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
