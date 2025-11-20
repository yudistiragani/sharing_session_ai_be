"""
backend/services/embedder.py

Utility functions to call the embedding service defined by EMBEDDING_BASE_URL.
Provides both sync and async APIs:
  - embed_text(text) -> List[float]
  - embed_text_async(text) -> List[float]
  - embed_batch(texts, batch_size=32) -> List[List[float]]
  - embed_batch_async(texts, batch_size=32) -> List[List[float]]

Behavior:
- Reads EMBEDDING_BASE_URL from settings module if available, else from env.
- Attempts to use batch endpoint if the service supports an array `input` payload.
- Falls back to per-item requests when necessary.
- Payload uses model 'ebbge-m3' by default.

Dependencies: httpx
"""
from typing import List, Dict, Any, Optional
import os
import logging
from settings import settings
import httpx

logger = logging.getLogger(__name__)

# try import settings if present
EMBEDDING_BASE_URL = settings.EMBEDDING_BASE_URL

# default model
EMBED_MODEL = os.getenv("EMBED_MODEL", "ebbge-m3")
# default timeouts
DEFAULT_TIMEOUT = float(os.getenv("EMBED_HTTP_TIMEOUT", "30"))


def _get_url() -> str:
    if not EMBEDDING_BASE_URL:
        raise RuntimeError("EMBEDDING_BASE_URL is not configured")
    return EMBEDDING_BASE_URL.rstrip("/")


def _parse_single_embedding_response(data: Dict[str, Any]) -> List[float]:
    """Parse several plausible response shapes for a single embedding call."""
    # Common shapes:
    # {"embedding": [...]}  (single)
    # {"data": [{"embedding": [...]}, ...], ...}
    # {"data": [{"vector": [...]}, ...], ...}
    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            first = data["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
            if isinstance(first, dict) and "vector" in first:
                return first["vector"]
    raise RuntimeError("Unknown embedding response format: %s" % (str(data)[:500]))


def _parse_batch_embedding_response(data: Dict[str, Any], expected_count: Optional[int] = None) -> List[List[float]]:
    """Parse batch response into list of embeddings.

    Expected shapes:
    - {"data": [{"embedding": [...]}, {"embedding": [...]}, ...]}
    - or possibly {"embeddings": [[...],[...]]} etc.
    """
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected embedding response type")

    # direct 'embeddings' key that contains list of lists
    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]

    if "data" in data and isinstance(data["data"], list):
        embeddings = []
        for item in data["data"]:
            if isinstance(item, dict):
                if "embedding" in item and isinstance(item["embedding"], list):
                    embeddings.append(item["embedding"])
                elif "vector" in item and isinstance(item["vector"], list):
                    embeddings.append(item["vector"])
                else:
                    # if item is unexpected, try skip
                    raise RuntimeError("Unknown item format inside data: %s" % str(item)[:300])
            else:
                raise RuntimeError("Unexpected item type inside data")
        if expected_count is not None and len(embeddings) != expected_count:
            # could be a mismatch but still return what we have
            logger.warning("batch embedding count mismatch: expected %s got %s", expected_count, len(embeddings))
        return embeddings

    raise RuntimeError("Unknown batch embedding response format")


# --------------------
# Sync functions
# --------------------

def embed_text(text: str, timeout: float = DEFAULT_TIMEOUT) -> List[float]:
    """Synchronously embed a single piece of text via EMBEDDING_BASE_URL.

    Raises RuntimeError on error; returns list[float].
    """
    url = _get_url()
    payload = {"model": EMBED_MODEL, "input": text}
    headers = {"Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding service returned {resp.status_code}: {resp.text}")
            data = resp.json()
            emb = _parse_single_embedding_response(data)
            return emb
    except Exception as e:
        logger.exception("embed_text failed: %s", e)
        raise


def embed_batch(texts: List[str], batch_size: int = 32, timeout: float = DEFAULT_TIMEOUT) -> List[List[float]]:
    """Synchronously embed a list of texts.

    This function will attempt to send the whole batch in one request if the embedding
    service supports array `input`. If that fails or service returns unexpected shape,
    it will fall back to per-item requests.
    """
    if not texts:
        return []

    url = _get_url()
    headers = {"Content-Type": "application/json"}

    # Try batch request first
    payload = {"model": EMBED_MODEL, "input": texts}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    embeddings = _parse_batch_embedding_response(data, expected_count=len(texts))
                    return embeddings
                except Exception:
                    # fallback to per-item
                    logger.warning("Batch response parsing failed, falling back to per-item embedding")
            else:
                logger.warning("Batch embedding request returned status %s, falling back to per-item", resp.status_code)
    except Exception as e:
        logger.warning("Batch embedding request failed: %s, falling back to per-item", e)

    # Fallback: per-item
    results: List[List[float]] = []
    for t in texts:
        emb = embed_text(t, timeout=timeout)
        results.append(emb)
    return results


# --------------------
# Async functions
# --------------------

async def embed_text_async(text: str, timeout: float = DEFAULT_TIMEOUT) -> List[float]:
    """Async embedding for a single text."""
    url = _get_url()
    payload = {"model": EMBED_MODEL, "input": text}
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding service returned {resp.status_code}: {resp.text}")
            data = resp.json()
            emb = _parse_single_embedding_response(data)
            return emb
    except Exception as e:
        logger.exception("embed_text_async failed: %s", e)
        raise


async def embed_batch_async(texts: List[str], batch_size: int = 32, timeout: float = DEFAULT_TIMEOUT) -> List[List[float]]:
    """Async batch embedding. Tries to send as one batch; fallback to per-item async calls."""
    if not texts:
        return []

    url = _get_url()

    # try single batch request
    payload = {"model": EMBED_MODEL, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    embeddings = _parse_batch_embedding_response(data, expected_count=len(texts))
                    return embeddings
                except Exception:
                    logger.warning("Batch async response parse failed, falling back to per-item async")
            else:
                logger.warning("Batch async embedding returned status %s, falling back to per-item", resp.status_code)
    except Exception as e:
        logger.warning("Batch async embedding request failed: %s", e)

    # fallback: per-item async with concurrency
    results: List[Optional[List[float]]] = [None] * len(texts)
    import asyncio

    sem = asyncio.Semaphore(8)

    async def _embed_one(i: int, txt: str):
        async with sem:
            emb = await embed_text_async(txt, timeout=timeout)
            results[i] = emb

    tasks = [asyncio.create_task(_embed_one(i, t)) for i, t in enumerate(texts)]
    await asyncio.gather(*tasks)

    # type: ignore
    return [r for r in results if r is not None]


# Convenience alias to support older code expecting embed_batch sync name and embed_text_async
embed_text_sync = embed_text
embed_batch_sync = embed_batch

