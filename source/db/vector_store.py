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

    This implementation is resilient to chromadb API differences:
      - Does NOT request 'ids' in the include list (some chroma versions reject it)
      - Tries several query signatures (embedding=..., query_embeddings=..., queries=...)
      - Normalizes returned structure into a list of dicts:
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

    # note: do NOT include 'ids' here because some chroma versions reject it.
    include = ["metadatas", "documents", "distances", "embeddings", "uris", "data"]

    query_res = None
    last_exc = None

    # Try common query signatures
    try:
        query_res = collection.query(embedding=query_embedding, n_results=top_k, include=include)
    except TypeError as e:
        last_exc = e
        try:
            query_res = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=include)
        except TypeError as e2:
            last_exc = e2
            try:
                query_res = collection.query(queries=[query_embedding], n_results=top_k, include=include)
            except Exception as e3:
                last_exc = e3

    if query_res is None:
        raise RuntimeError(f"Vector store search error: {last_exc}")

    # Normalize outputs
    # chroma may return nested lists when queries is a list. We always query a single vector so unwrap if nested.
    def _unwrap(key):
        v = query_res.get(key) if isinstance(query_res, dict) else None
        if v and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return v[0]
        return v or []

    metadatas = _unwrap("metadatas")
    documents = _unwrap("documents")
    distances = _unwrap("distances")
    # sometimes 'ids' not present; try to grab it if available
    ids = _unwrap("ids") if isinstance(query_res, dict) else []

    results: List[Dict[str, Any]] = []
    n = max(len(ids), len(documents), len(metadatas), len(distances))

    for i in range(n):
        # id resolution strategy:
        vec_id = None
        if i < len(ids) and ids[i]:
            vec_id = ids[i]
        else:
            # try to extract from metadata
            if i < len(metadatas) and isinstance(metadatas[i], dict):
                vec_id = metadatas[i].get("chunk_id") or metadatas[i].get("id") or metadatas[i].get("vector_id")
            # fallback: use combination of doc fields
            if not vec_id:
                # try derive from document content (hash truncated) or index-based id
                doc_preview = documents[i] if i < len(documents) else None
                if isinstance(doc_preview, str) and len(doc_preview) > 0:
                    vec_id = f"doc-{coll_name}-idx-{i}"
                else:
                    vec_id = f"{coll_name}-idx-{i}"

        metadata_item = metadatas[i] if i < len(metadatas) else None
        doc_item = documents[i] if i < len(documents) else None
        score_item = distances[i] if i < len(distances) else None

        results.append({
            "id": vec_id,
            "score": score_item,
            "metadata": metadata_item,
            "document": doc_item,
        })

    return results
