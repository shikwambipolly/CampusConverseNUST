"""
src/semantic_layer.py
=====================
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
    and structured relational facts (AlloyDB / PostgreSQL / SQLite).
    """

    def __init__(self):
        # ---------------------------------------------------------
        # 1. Chroma Vector DB Configuration (Lazy Initialized)
        # ---------------------------------------------------------
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_dir = os.getenv("CHROMA_PERSIST_DIR", os.path.join(base_dir, "chroma_db"))
        self.collection_name = "nust_campus_knowledge"
        self._collection = None
        self._chroma_unavailable = False

        # ---------------------------------------------------------
        # 2. Database Connection Credentials for AlloyDB / SQLite
        # ---------------------------------------------------------
        self.alloydb_host = os.getenv("ALLOYDB_HOST", "127.0.0.1")
        self.alloydb_port = int(os.getenv("ALLOYDB_PORT", "5432"))
        self.alloydb_name = os.getenv("ALLOYDB_DATABASE", "campus_converse")
        self.alloydb_user = os.getenv("ALLOYDB_USER", "postgres")
        self.alloydb_password = os.getenv("ALLOYDB_PASSWORD", "")
        self.sqlite_fallback_path = os.getenv(
            "SQLITE_PATH", os.path.join(base_dir, "data", "nust_facts.db")
        )

    def _get_collection(self):
        """Lazily connects to Chroma collection so read-only environments gracefully fallback."""
        if self._collection is not None:
            return self._collection
        if self._chroma_unavailable:
            return None

        try:
            if os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true":
                chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
            else:
                chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_func
            )
            return self._collection
        except Exception:
            self._chroma_unavailable = True
            return None

    def _use_alloydb(self) -> bool:
        return PSYCOPG2_AVAILABLE and os.getenv("USE_ALLOYDB", "false").lower() == "true"

    def _get_db_connection(self):
        """
        Creates a connection to AlloyDB (PostgreSQL) if configured,
        otherwise falls back to local SQLite.
        """
        if self._use_alloydb():
            return psycopg2.connect(
                host=self.alloydb_host,
                port=self.alloydb_port,
                dbname=self.alloydb_name,
                user=self.alloydb_user,
                password=self.alloydb_password,
                cursor_factory=RealDictCursor
            )
        conn = sqlite3.connect(self.sqlite_fallback_path)
        conn.row_factory = sqlite3.Row
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
        collection = self._get_collection()
        if collection is None:
            return []

        where_clause = {"faculty": faculty_filter} if faculty_filter else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
        except Exception:
            return []

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            ids = results["ids"][0] if "ids" in results and results["ids"] else [None] * len(results["documents"][0])
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                formatted_results.append({
                    "chunk_id": cid,
                    "text": doc,
                    "source_document": meta.get("source_document"),
                    "section_title": meta.get("section_title"),
                    "page_number": meta.get("page_number"),
                    "faculty": meta.get("faculty"),
                    "relevance_distance": round(float(dist), 4)
                })

        return formatted_results

    # =========================================================================
    # STRUCTURED RETRIEVAL: AlloyDB / SQLite Lookups
    # =========================================================================
    def get_module_prerequisites(self, module_code: str) -> Optional[Dict[str, Any]]:
        """Queries module_catalog for prerequisites, corequisites, and module details."""
        clean_code = module_code.strip().upper()
        conn = self._get_db_connection()
        cursor = conn.cursor()

        ph = "%s" if self._use_alloydb() else "?"
        sql = f"""
            SELECT module_code, module_name, faculty, department, prerequisites, corequisites
            FROM module_catalog
            WHERE UPPER(module_code) = {ph} OR UPPER(module_code) LIKE {ph} OR LOWER(module_name) LIKE {ph}
            LIMIT 1;
        """
        cursor.execute(sql, (clean_code, f"%{clean_code}%", f"%{module_code.strip().lower()}%"))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        data = dict(row)
        for key in ("prerequisites", "corequisites"):
            if isinstance(data.get(key), str):
                try:
                    data[key] = json.loads(data[key])
                except Exception:
                    pass
        return data

    def search_academic_calendar(self, event_keyword: str) -> List[Dict[str, Any]]:
        """Looks up exam weeks, test dates, registration deadlines, and recess in the calendar table."""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        pattern = f"%{event_keyword.strip()}%"
        like_op = "ILIKE %s" if self._use_alloydb() else "LIKE ?"
        sql = f"""
            SELECT academic_year, semester, event_name, start_date, end_date, category
            FROM academic_calendar
            WHERE event_name {like_op} OR category {like_op}
            ORDER BY start_date ASC
            LIMIT 10;
        """
        cursor.execute(sql, (pattern, pattern))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_fee_breakdown(self, category_keyword: str = "") -> Dict[str, Any]:
        """Queries tuition/admin fees, hostel residence fees, payment schedule, and banking details."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        kw = category_keyword.strip()
        pattern = f"%{kw}%"
        like_op = "ILIKE %s" if self._use_alloydb() else "LIKE ?"

        cursor.execute(
            f"SELECT fee_category, fee_name, amount_nad, frequency, notes FROM fee_structure "
            f"WHERE fee_category {like_op} OR fee_name {like_op} OR notes {like_op}",
            (pattern, pattern, pattern)
        )
        fees = [dict(r) for r in cursor.fetchall()]
        if not fees:
            cursor.execute("SELECT fee_category, fee_name, amount_nad, frequency, notes FROM fee_structure LIMIT 10")
            fees = [dict(r) for r in cursor.fetchall()]

        cursor.execute(
            f"SELECT item_name, category, amount_nad, period, notes FROM residence_fees "
            f"WHERE item_name {like_op} OR category {like_op}",
            (pattern, pattern)
        )
        residence = [dict(r) for r in cursor.fetchall()]
        if not residence and any(w in kw.lower() for w in ("hostel", "residence", "accommodation", "room", "meal", "dining")):
            cursor.execute("SELECT item_name, category, amount_nad, period, notes FROM residence_fees LIMIT 10")
            residence = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT deadline_date, course_type, percentage_due, description FROM fee_payment_schedule")
        schedule = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT bank_name, account_name, branch_name, branch_code, account_number, reference_format, contact_email FROM banking_details")
        banking = [dict(r) for r in cursor.fetchall()]

        conn.close()
        return {
            "general_fees": fees,
            "residence_fees": residence,
            "payment_schedule": schedule,
            "banking_details": banking
        }

    def search_programmes(self, keyword: str) -> List[Dict[str, Any]]:
        """Searches degree/diploma programmes, NQF levels, duration, and minimum admission points."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        pattern = f"%{keyword.strip()}%"
        like_op = "ILIKE %s" if self._use_alloydb() else "LIKE ?"
        cursor.execute(
            f"SELECT programme_code, programme_name, faculty, nqf_level, duration_years, minimum_points, admission_requirements "
            f"FROM programmes WHERE programme_code {like_op} OR programme_name {like_op} OR faculty {like_op} LIMIT 8",
            (pattern, pattern, pattern)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


# Singleton accessor
_semantic_layer_instance: Optional[SemanticLayer] = None


def get_semantic_layer() -> SemanticLayer:
    global _semantic_layer_instance
    if _semantic_layer_instance is None:
        _semantic_layer_instance = SemanticLayer()
    return _semantic_layer_instance

