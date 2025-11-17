import asyncio
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import preferred embedder functions (async first)
_embedder_async = None
_embedder_sync = None
try:
    # If you implemented an async embedder (recommended)
    from services.embedder import embed_text_async as _embedder_async  # type: ignore
except Exception:
    _embedder_async = None

try:
    # fallback to sync embedder function
    from services.embedder import embed_text as _embedder_sync  # type: ignore
except Exception:
    _embedder_sync = None

# Vector store (sync API implemented earlier)
try:
    from db.vector_store import search_similar  # sync function
except Exception:
    search_similar = None  # we'll guard usage later


def build_prompt(question: str, context_texts: List[str]) -> str:
    """
    Build the prompt string using up to top-3 contexts.
    """
    top_contexts = context_texts[:3] if context_texts else []
    context_block = "\n\n".join(top_contexts) if top_contexts else "No context available."

    prompt = (
        "You are a document assistant. Use the context below to answer the user's question.\n"
        "Context:\n"
        f"{context_block}\n\n"
        "Question:\n"
        f"{question}"
    )
    return prompt


async def _run_in_thread(fn, *args, **kwargs):
    """
    Helper: run sync function in default ThreadPoolExecutor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


async def embed_query_async(text: str) -> List[float]:
    """
    Embed a single query text, async-friendly.
    Priority:
      1. services.embedder.embed_text_async (if provided)
      2. services.embedder.embed_text (sync) executed in threadpool
    Raises RuntimeError if no embedder available or embedding fails.
    """
    if _embedder_async is not None:
        try:
            # assume async function
            emb = await _embedder_async(text)
            return emb
        except Exception as e:
            logger.exception("Async embedder failed: %s", e)
            raise RuntimeError(f"Async embedder error: {e}")

    if _embedder_sync is not None:
        try:
            emb = await _run_in_thread(_embedder_sync, text)
            return emb
        except Exception as e:
            logger.exception("Sync embedder (in thread) failed: %s", e)
            raise RuntimeError(f"Embedder error: {e}")

    raise RuntimeError("No embedder function available. Implement services.embedder.embed_text or embed_text_async.")


async def search_top_k_async(
    doc_id: str,
    query_embedding: List[float],
    top_k: int = 3,
    collection_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search top_k similar chunks in the vector store for a given doc_id (collection).
    Wraps sync vector_store.search_similar(...) into async.

    - collection_name: optional custom collection name (if not provided will use docs_{doc_id})
    Returns list of result dicts as produced by vector_store.search_similar:
      [{"id":..., "score":..., "metadata":..., "document": ...}, ...]
    """
    if search_similar is None:
        raise RuntimeError("vector_store.search_similar not available")

    # default collection naming convention
    coll = collection_name or f"docs_{doc_id}"

    try:
        # search_similar is sync: run in threadpool
        res = await _run_in_thread(search_similar, query_embedding, top_k, coll)
        return res
    except Exception as e:
        logger.exception("Vector store search failed: %s", e)
        raise RuntimeError(f"Vector store search error: {e}")


async def retrieve_and_build_prompt(
    doc_id: str,
    question: str,
    top_k: int = 3,
    collection_name: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    High-level helper:
      - embed the question
      - search top-k similar chunks in the vector store (collection docs_{doc_id} by default)
      - assemble context_texts (the chunk texts)
      - build prompt

    Returns: (prompt, context_texts, search_results)
    """
    # 1) embed the question
    query_emb = await embed_query_async(question)

    # 2) search top-k
    results = await search_top_k_async(doc_id, query_emb, top_k=top_k, collection_name=collection_name)

    # results items expected to include 'document' or similar field with the chunk text
    context_texts: List[str] = []
    for r in results:
        # try several possible keys for stored text
        text = None
        if isinstance(r, dict):
            # standard from our vector_store earlier used keys: "document" or "document" with actual text
            text = r.get("document") or r.get("document_text") or r.get("text") or r.get("metadata", {}).get("text")
            # if still None, try metadata or other fallback
            if not text and "metadata" in r and isinstance(r["metadata"], dict):
                # sometimes 'metadata' may hold reference only; leave fallback to empty string
                text = r["metadata"].get("source_text") or r["metadata"].get("text")
        if not text:
            # fallback: stringify the whole result (last resort)
            try:
                text = str(r)
            except Exception:
                text = ""
        context_texts.append(text)

    # 3) build prompt using top-3 contexts
    prompt = build_prompt(question, context_texts)

    return prompt, context_texts, results
