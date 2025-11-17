# backend/db/vector_store.py

import os
import typing
from typing import List, Dict, Any, Optional

VECTOR_DB = os.getenv("VECTOR_DB", "chroma").strip().lower()

# --- Chromadb imports (optional) ---
chromadb = None
_chroma_available = False
if VECTOR_DB == "chroma":
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        _chroma_available = True
    except Exception:
        chromadb = None
        _chroma_available = False


# --- Helper: chroma client / collection management ---
def _ensure_chroma():
    if not _chroma_available or chromadb is None:
        raise RuntimeError(
            "chromadb tidak terpasang atau tidak tersedia. "
            "Install dengan: pip install chromadb"
        )


def _get_chroma_client() -> "chromadb.Client":
    """
    Return a chroma Client. If CHROMA_PERSIST_DIR env var is set, use persistent DuckDB+Parquet mode.
    Otherwise, return an in-memory client.
    """
    _ensure_chroma()
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", None)
    if persist_dir:
        settings = ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        return chromadb.Client(settings)
    # default in-memory client
    return chromadb.Client()


def _get_collection(client: "chromadb.Client", name: str, create_if_missing: bool = True):
    """
    Return the named collection. Create if missing when create_if_missing is True.
    """
    if create_if_missing:
        return client.get_or_create_collection(name=name)
    # else try to get existing one (will raise if not exists)
    return client.get_collection(name=name)


# --- Public API ---


def add_embeddings(
    doc_id: str,
    embeddings: List[List[float]],
    chunks: List[Dict[str, Any]],
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add embeddings into vector store.

    Args:
      - doc_id: identifier for the source document (used to build collection name by default)
      - embeddings: list of vector lists (float)
      - chunks: list of dicts corresponding to each embedding. Each chunk dict is expected to contain:
          {
            "chunk_id": "<id>",      # recommended (but optional)
            "text": "<chunk text>",
            "source": "<filename>",
            "origin_meta": {...}
          }
        length of chunks must match length of embeddings.
      - collection_name: optional override for collection name.

    Returns:
      summary dict, e.g. {"collection": "docs_<doc_id>", "num_added": N, "ids": [...]}.

    Behavior:
      - If VECTOR_DB != "chroma": raises RuntimeError with message "penyimpanan belum tersedia".
    """
    if VECTOR_DB == "faiss":
        # Feature not implemented for FAISS in current MVP
        raise RuntimeError("penyimpanan belum tersedia")

    if VECTOR_DB != "chroma":
        raise RuntimeError(f"Unknown VECTOR_DB '{VECTOR_DB}' configured")

    # validate inputs
    if len(embeddings) != len(chunks):
        raise ValueError("length of embeddings must equal length of chunks")

    _ensure_chroma()
    client = _get_chroma_client()
    coll_name = collection_name or f"docs_{doc_id}"
    collection = _get_collection(client, coll_name, create_if_missing=True)

    ids = []
    docs = []
    metadatas = []
    # Use chunk-provided chunk_id if available; else fallback to index-based id
    for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
        chunk_id = None
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
        # build stable id: prefer chunk_id, else use f"{doc_id}_c{i}"
        vec_id = f"{doc_id}_{chunk_id}" if chunk_id else f"{doc_id}_c{i}"
        ids.append(vec_id)
        docs.append(chunk.get("text") if isinstance(chunk, dict) else str(chunk))
        # metadata should include doc_id, chunk_id, plus any origin_meta and source
        meta = {"doc_id": doc_id, "chunk_id": chunk_id}
        if isinstance(chunk, dict):
            if "source" in chunk and chunk["source"] is not None:
                meta["source"] = chunk["source"]
            origin_meta = chunk.get("origin_meta")
            if isinstance(origin_meta, dict):
                meta.update(origin_meta)
        metadatas.append(meta)

    # Upsert vectors
    collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metadatas)

    return {"collection": coll_name, "num_added": len(ids), "ids": ids}


def search_similar(query_embedding: List[float], top_k: int = 3, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for top_k similar vectors to the provided query_embedding.

    Returns list of dicts with keys:
      - "id": vector id
      - "score": similarity score/distance (as returned by chroma)
      - "document": the stored document/text
      - "metadata": stored metadata dict

    For VECTOR_DB = faiss: raises RuntimeError("penyimpanan belum tersedia")
    """
    if VECTOR_DB == "faiss":
        raise RuntimeError("penyimpanan belum tersedia")

    if VECTOR_DB != "chroma":
        raise RuntimeError(f"Unknown VECTOR_DB '{VECTOR_DB}' configured")

    _ensure_chroma()
    client = _get_chroma_client()
    # if collection_name not provided, search across all collections is not supported here;
    # require collection name to be provided in multi-doc setups. Default to 'default' collection?
    coll_name = collection_name
    if not coll_name:
        # default behavior: search across all collections named 'docs_*' is expensive; attempt to use 'default'
        # but safer is to raise so caller thinks about collection scope.
        raise ValueError("collection_name must be provided for search_similar (e.g. 'docs_<doc_id>')")

    try:
        collection = client.get_collection(name=coll_name)
    except Exception as e:
        # collection missing -> return empty
        return []

    # Query chroma
    # include distances, metadatas, documents, ids
    query_res = collection.query(embedding=query_embedding, n_results=top_k, include=["metadatas", "documents", "distances", "ids"])

    # chroma returns dicts of lists; normalize into list of results
    ids = query_res.get("ids", [])
    distances = query_res.get("distances", [])
    metadatas = query_res.get("metadatas", [])
    documents = query_res.get("documents", [])

    results: List[Dict[str, Any]] = []
    # ids/distances/metadatas/documents can be nested lists if multiple queries were done;
    # but since we query a single vector, they should be lists at top-level.
    # However handle both possibilities:
    if ids and isinstance(ids[0], list):
        # nested form, take first query result
        ids = ids[0]
        distances = distances[0] if distances and isinstance(distances[0], list) else distances
        metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
        documents = documents[0] if documents and isinstance(documents[0], list) else documents

    for i, vec_id in enumerate(ids):
        res = {
            "id": vec_id,
            "score": distances[i] if i < len(distances) else None,
            "metadata": metadatas[i] if i < len(metadatas) else None,
            "document": documents[i] if i < len(documents) else None,
        }
        results.append(res)

    return results
