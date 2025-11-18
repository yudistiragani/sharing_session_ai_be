# backend/db/vector_store.py

import os
import typing
from settings import settings
from typing import List, Dict, Any, Optional

VECTOR_DB = settings.VECTOR_DB.strip().lower()

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

    This function is resilient to several chromadb client signatures:
      - collection.query(embedding=..., n_results=..., include=[...])
      - collection.query(query_embeddings=[...], n_results=..., include=[...])
      - collection.query(queries=[...], n_results=..., include=[...])

    It normalizes the returned structure into a list of dicts:
      [{"id": ..., "score": ..., "metadata": ..., "document": ...}, ...]
    """
    if VECTOR_DB == "faiss":
        raise RuntimeError("penyimpanan belum tersedia")

    if VECTOR_DB != "chroma":
        raise RuntimeError(f"Unknown VECTOR_DB '{VECTOR_DB}' configured")

    _ensure_chroma()
    client = _get_chroma_client()
    coll_name = collection_name
    if not coll_name:
        raise ValueError("collection_name must be provided for search_similar (e.g. 'docs_<doc_id>')")

    try:
        collection = client.get_collection(name=coll_name)
    except Exception:
        # collection missing -> return empty list
        return []

    include = ["metadatas", "documents", "distances", "ids"]

    query_res = None
    last_exc = None

    # Try several possible query signatures that different chroma versions use
    try:
        query_res = collection.query(embedding=query_embedding, n_results=top_k, include=include)
    except TypeError as e:
        last_exc = e
        try:
            # some versions accept 'query_embeddings' as list of embeddings
            query_res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=include)
        except TypeError as e2:
            last_exc = e2
            try:
                # some versions accept 'queries' key
                query_res = collection.query(queries=[query_embedding], n_results=top_k, include=include)
            except Exception as e3:
                last_exc = e3

    if query_res is None:
        # If none worked, raise a helpful error with underlying exception message
        raise RuntimeError(f"Vector store search error: {last_exc}")

    # Normalize response structures from chroma into lists
    # Typical shape from chroma: dict with lists (possibly nested): ids, distances, metadatas, documents
    ids = query_res.get("ids") if isinstance(query_res, dict) else None
    distances = query_res.get("distances") if isinstance(query_res, dict) else None
    metadatas = query_res.get("metadatas") if isinstance(query_res, dict) else None
    documents = query_res.get("documents") if isinstance(query_res, dict) else None

    # If nested (queries was list), select first result
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if documents and isinstance(documents[0], list):
        documents = documents[0]

    # Fallback: chroma may sometimes return results in different keys; try alternate keys
    if ids is None and isinstance(query_res, dict):
        # try 'result' key or similar
        # flatten any list-like values if necessary
        ids = query_res.get("result_ids") or query_res.get("ids", [])
        distances = distances or query_res.get("result_distances") or query_res.get("distances", [])
        metadatas = metadatas or query_res.get("result_metadatas") or query_res.get("metadatas", [])
        documents = documents or query_res.get("result_documents") or query_res.get("documents", [])

    # Ensure lists
    ids = ids or []
    distances = distances or []
    metadatas = metadatas or []
    documents = documents or []

    results: List[Dict[str, Any]] = []
    for i, vec_id in enumerate(ids):
        res = {
            "id": vec_id,
            "score": distances[i] if i < len(distances) else None,
            "metadata": metadatas[i] if i < len(metadatas) else None,
            "document": documents[i] if i < len(documents) else None,
        }
        results.append(res)

    return results

