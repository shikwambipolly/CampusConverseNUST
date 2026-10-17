"""
index_chroma.py
================
Purpose:
    Reads pre-processed text chunks (stored locally or in Google Cloud Storage)
    and indexes them into a persistent or remote Chroma vector database.
    Chroma's default sentence-transformer embedding function creates vectors automatically.
"""

import os
import json
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------
# Path to the JSON file containing the extracted chunks (or GCS mounted volume)
CHUNKS_FILE_PATH = os.getenv("CHUNKS_FILE_PATH", "data/chunks/corpus_chunks.json")

# Chroma host and port (if running as a client/server on GCP or persistent dir)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "nust_campus_knowledge"


def get_chroma_client():
    """
    Initializes and returns the Chroma client.
    Connects to a remote Chroma server (Cloud Run / GCE) if CHROMA_HOST is set,
    otherwise falls back to local directory persistence.
    """
    if os.getenv("USE_REMOTE_CHROMA", "false").lower() == "true":
        # Connect to Chroma running as a microservice on Cloud Run / Compute Engine
        print(f"[INFO] Connecting to remote Chroma DB at {CHROMA_HOST}:{CHROMA_PORT}...")
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    else:
        # Fallback to local persistent storage for testing and building images
        print(f"[INFO] Using local persistent Chroma DB at {CHROMA_PERSIST_DIR}...")
        return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def load_chunks(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads text chunks from the ingested JSON file.
    Each chunk contains: chunk_id, text, source_document, section_title, page_number, faculty.
    """
    print(f"[INFO] Loading chunks from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Successfully loaded {len(chunks)} chunks.")
    return chunks


def index_chunks_into_chroma():
    """
    Main function to read chunks, initialize Chroma with an embedding function,
    and batch-insert records into the Chroma collection.
    """
    # 1. Initialize client
    client = get_chroma_client()

    # 2. Select embedding function:
    # Chroma handles embeddings under the hood using sentence-transformers (all-MiniLM-L6-v2)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 3. Create or get existing collection
    # 'get_or_create_collection' prevents duplicate initialization errors
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Use cosine similarity for semantic distance
    )

    # 4. Load the pre-processed chunks
    chunks = load_chunks(CHUNKS_FILE_PATH)

    # 5. Prepare batch arrays for Chroma ingestion
    # Chroma accepts documents (raw text), IDs (unique string), and metadata dicts
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Ensure ID is a string
        ids.append(str(chunk["chunk_id"]))
        
        # The raw text to be converted to vector embedding by Chroma
        documents.append(chunk["text"])

        # Store metadata for citation, filtering, and provenance tracking
        metadatas.append({
            "source_document": chunk.get("source_document", "Unknown"),
            "section_title": chunk.get("section_title", "General"),
            "page_number": int(chunk.get("page_number", 0)),
            "faculty": chunk.get("faculty", "General"),
            "approx_tokens": int(chunk.get("approx_tokens", 0))
        })

    # 6. Ingest in batches to prevent payload size timeouts
    BATCH_SIZE = 100
    total_chunks = len(documents)

    print(f"[INFO] Ingesting {total_chunks} chunks in batches of {BATCH_SIZE}...")
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]

        # Chroma's 'upsert' embeds the text automatically and stores both vectors & texts
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )
        print(f"[INFO] Indexed chunks {i + 1} to {min(i + BATCH_SIZE, total_chunks)} of {total_chunks}")

    print("[SUCCESS] Chroma vector database indexing complete!")


if __name__ == "__main__":
    index_chunks_into_chroma()
