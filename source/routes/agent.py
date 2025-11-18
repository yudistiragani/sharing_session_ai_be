import os
import uuid
import tempfile
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query, Form
from loguru import logger

# extractor, retriever, llm client
from services.extractor import extract_and_chunk
from services.retriever import retrieve_and_build_prompt
from services.llm_client import ask_llm

# DB helpers
from db import postgres_conn
from db.redis_cache import set_job_status, get_job_status
from db.vector_store import add_embeddings  # may raise for FAISS

router = APIRouter(prefix="/agent", tags=["agent"])


# -----------------------
# 1) UPLOAD
# -----------------------
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload file -> extract & chunk -> persist to Postgres (documents + chunks)
    Returns generated doc_id and chunk count.
    """
    db = postgres_conn.SessionLocal()
    try:
        allowed_ext = {".pdf", ".docx", ".csv"}
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use PDF/DOCX/CSV.")

        # save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Saved uploaded file to temp: {tmp_path}")

        # extract & chunk
        chunks = extract_and_chunk(tmp_path, chunk_size=800, overlap=200)

        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            logger.warning("Could not delete temp file: %s", tmp_path)

        # create doc_id and persist
        doc_id = str(uuid.uuid4())
        persisted_chunks = []
        for c in chunks:
            persisted_chunks.append({
                "chunk_id": c.get("chunk_id"),
                "doc_id": doc_id,
                "text": c.get("text"),
                "source": c.get("source"),
                "origin_meta": c.get("origin_meta", {}),
            })

        # create document row and insert chunks
        postgres_conn.create_document(db, doc_id=doc_id, filename=filename, num_chunks=len(persisted_chunks))
        postgres_conn.add_chunks_bulk(db, persisted_chunks)
        postgres_conn.update_index_job_status(db, job_id=doc_id, status="uploaded") if False else None  # no-op; ignore

        return {
            "status": "uploaded",
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(persisted_chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /agent/upload")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# -----------------------
# 2) INDEX
# -----------------------
async def _call_embedding_single(client, url: str, text: str, timeout: float = 30.0):
    payload = {"model": "ebbge-m3", "input": text}
    resp = await client.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding service returned status {resp.status_code}: {resp.text}")
    data = resp.json()
    # parse common shapes
    if isinstance(data, dict) and "embedding" in data:
        return data["embedding"]
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"]:
        first = data["data"][0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
        if isinstance(first, dict) and "vector" in first:
            return first["vector"]
    raise RuntimeError("Unknown embedding response format")


async def _embed_chunks_concurrent(chunks: List[Dict[str, Any]], embedding_url: str, batch_size: int = 16, concurrency: int = 8):
    results = []
    import httpx
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def embed_one(chunk):
            async with sem:
                try:
                    emb = await _call_embedding_single(client, embedding_url, chunk["text"])
                    return {"chunk": chunk, "embedding": emb, "error": None}
                except Exception as e:
                    return {"chunk": chunk, "embedding": None, "error": str(e)}

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            tasks = [asyncio.create_task(embed_one(c)) for c in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)
    return results


@router.post("/index")
async def index_document(
    doc_id: str = Form(..., description="Document id returned from /agent/upload"),
    mode: str = Form("sync", description="'sync' or 'async'"),
    batch_size: int = Form(16),
    concurrency: int = Form(8),
    background_tasks: BackgroundTasks = None,
):
    """
    Index document chunks into vector DB.
    mode=sync: perform indexing and return result
    mode=async: schedule background indexing task (FastAPI BackgroundTasks)
    """
    db = postgres_conn.SessionLocal()
    try:
        # check doc exists
        doc = db.query(postgres_conn.Document).get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="doc_id not found")

        # create job record
        job_id = str(uuid.uuid4())
        postgres_conn.create_index_job(db, job_id=job_id, doc_id=doc_id)
        set_job_status(job_id, {"status": "queued", "doc_id": doc_id})

        async def _index_job(job_id_inner: str, doc_id_inner: str):
            db_inner = postgres_conn.SessionLocal()
            try:
                chunks = postgres_conn.get_chunks_by_doc_id(db_inner, doc_id_inner)
                if not chunks:
                    postgres_conn.update_index_job_status(db_inner, job_id_inner, "failed", {"error": "no_chunks"})
                    set_job_status(job_id_inner, {"status": "failed", "reason": "no_chunks"})
                    return

                postgres_conn.update_index_job_status(db_inner, job_id_inner, "running")
                set_job_status(job_id_inner, {"status": "running", "doc_id": doc_id_inner, "total_chunks": len(chunks)})

                # embed chunks
                embedding_url = os.getenv("EMBEDDING_BASE_URL")
                if not embedding_url:
                    raise RuntimeError("EMBEDDING_BASE_URL is not set")

                embed_results = await _embed_chunks_concurrent(chunks, embedding_url, batch_size=batch_size, concurrency=concurrency)

                successful = [r for r in embed_results if r["error"] is None]
                failed = [r for r in embed_results if r["error"]]

                embeddings = [r["embedding"] for r in successful]
                success_chunks = [r["chunk"] for r in successful]

                # persist to vector store (may raise for FAISS)
                try:
                    add_res = add_embeddings(doc_id=doc_id_inner, embeddings=embeddings, chunks=success_chunks)
                except Exception as e:
                    postgres_conn.update_index_job_status(db_inner, job_id_inner, "failed", {"error": str(e)})
                    set_job_status(job_id_inner, {"status": "failed", "reason": str(e)})
                    return

                # mark chunks indexed
                indexed_chunk_ids = [c["chunk_id"] for c in success_chunks]
                postgres_conn.mark_chunks_indexed(db_inner, indexed_chunk_ids)

                details = {
                    "total_chunks": len(chunks),
                    "indexed_count": add_res.get("num_added", 0),
                    "failed_embeddings": [{"chunk_id": r["chunk"]["chunk_id"], "error": r["error"]} for r in failed]
                }
                postgres_conn.update_index_job_status(db_inner, job_id_inner, "success", details)
                set_job_status(job_id_inner, {"status": "success", "doc_id": doc_id_inner, "details": details})
            except Exception as e:
                logger.exception("Index job failed: %s", e)
                try:
                    postgres_conn.update_index_job_status(db_inner, job_id_inner, "failed", {"error": str(e)})
                except Exception:
                    pass
                set_job_status(job_id_inner, {"status": "failed", "reason": str(e)})
            finally:
                db_inner.close()

        if mode == "async":
            if background_tasks is None:
                raise HTTPException(status_code=500, detail="BackgroundTasks not available")
            # schedule background task
            background_tasks.add_task(lambda: asyncio.run(_index_job(job_id, doc_id)))
            return {"status": "queued", "job_id": job_id, "doc_id": doc_id}
        elif mode == "sync":
            # run inline
            await _index_job(job_id, doc_id)
            job = postgres_conn.get_index_job(db, job_id)
            return {"status": "finished", "job": job}
        else:
            raise HTTPException(status_code=400, detail="mode must be 'sync' or 'async'")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /agent/index")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# -----------------------
# 3) CHAT
# -----------------------
@router.post("/chat")
async def chat(doc_id: str = Form(...), question: str = Form(...), top_k: int = Form(3)):
    """
    Chat Q&A flow:
      - retrieve top-k contexts for doc_id using retriever
      - build prompt (inside retriever)
      - call ask_llm(question, context_texts) which returns the answer
      - return answer plus contexts and sources
    """
    db = postgres_conn.SessionLocal()
    try:
        # ensure doc exists
        doc = db.query(postgres_conn.Document).get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="doc_id not found")

        # retrieve contexts & build prompt
        prompt, context_texts, search_results = await retrieve_and_build_prompt(doc_id, question, top_k=top_k)

        # ask LLM (async)
        answer = await ask_llm(question, context_texts)

        # Build a useful response: include contexts with metadata for traceability
        contexts_with_meta = []
        for r, text in zip(search_results, context_texts):
            meta = r.get("metadata") if isinstance(r, dict) else None
            contexts_with_meta.append({
                "text": text,
                "metadata": meta,
            })

        # Optionally cache the chat result in Redis (sessionless example uses jobid)
        chat_session_id = str(uuid.uuid4())
        try:
            set_job_status(chat_session_id, {"question": question, "answer": answer, "doc_id": doc_id})
        except Exception:
            # redis optional; ignore failures
            pass

        return {
            "doc_id": doc_id,
            "question": question,
            "answer": answer,
            "contexts": contexts_with_meta,
            "prompt": prompt,
            "chat_session_id": chat_session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /agent/chat")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# -----------------------
# 4) GET DOCS/{id}
# -----------------------

# @router.get("/docs/{id}")
# async def get_doc(id: str):
#     """
#     Return document metadata and small stats from Postgres.
#     Uses postgres_conn.get_document which returns a plain dict (no ORM objects).
#     """
#     db = postgres_conn.SessionLocal()
#     try:
#         doc = postgres_conn.get_document(db, id)
#         if not doc:
#             raise HTTPException(status_code=404, detail="Document not found")
#         return doc
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Error in /agent/docs/{id}: %s", e)
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         db.close()

@router.get("/docs/{id}")
async def get_doc(id: str, question: str = Query(None), top_k: int = Query(3, ge=1, le=10)):
    """
    Return document metadata and chunks.
    Optional query params:
      - question (string): if provided, retrieve top_k relevant chunks (via retriever) and produce highlights.
      - top_k (int): number of top chunks to retrieve when question provided (default 3)

    Response structure:
    {
      "doc_id": "...",
      "filename": "...",
      "uploaded_at": "...",
      "status": "...",
      "num_chunks": N,
      "metadata": {...} or null,
      "chunks": [  # ALL chunks if no question; otherwise top_k chunks
         {
           "chunk_id": "...",
           "text": "...",
           "source": "...",
           "origin_meta": {...},
           "indexed": true/false,
           // when question provided, optionally:
           "score": 0.123,            # if vector_store returned a score
           "highlights": [
               {"snippet": "...", "start": 123, "end": 145}
           ]
         },
         ...
      ],
      "question_context": { ... }  # when question provided: prompt, answer not included here
    }
    """
    db = postgres_conn.SessionLocal()
    try:
        # 1) fetch document metadata (primitive dict)
        doc = postgres_conn.get_document(db, id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # helper to load chunks (all chunks) as list of dicts
        all_chunks = postgres_conn.get_chunks_by_doc_id(db, id)

        # if no question -> return metadata + (optionally) all chunks (without highlights)
        if not question:
            # return metadata + all chunk summaries (no heavy ops)
            return {
                "doc_id": doc["id"],
                "filename": doc["filename"],
                "uploaded_at": doc["uploaded_at"],
                "status": doc["status"],
                "num_chunks": doc["num_chunks"],
                "metadata": doc["metadata"],
                "chunks": all_chunks,
            }

        # 2) question provided -> use retriever to get top_k contexts + details
        # retrieve_and_build_prompt returns (prompt, context_texts, search_results)
        prompt, context_texts, search_results = await retrieve_and_build_prompt(id, question, top_k=top_k)

        # 3) Build detailed chunk objects: combine search_result metadata + chunk text
        # search_results are normalized by vector_store.search_similar to include "id", "score", "metadata", "document"
        top_chunks = []
        for idx, res in enumerate(search_results):
            # find the chunk text: prefer 'document' field; fallback to context_texts
            text = None
            if isinstance(res, dict):
                text = res.get("document") or (context_texts[idx] if idx < len(context_texts) else None)
            if text is None:
                text = context_texts[idx] if idx < len(context_texts) else ""

            # metadata and score
            metadata = res.get("metadata") if isinstance(res, dict) else None
            score = res.get("score") if isinstance(res, dict) else None
            chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) and metadata.get("chunk_id") else (res.get("id") if isinstance(res, dict) else None)

            # generate lightweight highlights from question -> snippet(s)
            highlights = _extract_highlights_simple(question, text, max_snippets=2)

            top_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": metadata.get("source") if isinstance(metadata, dict) else None,
                "origin_meta": metadata if isinstance(metadata, dict) else None,
                "score": score,
                "highlights": highlights,
            })

        # 4) optional: include all chunks (full list) but mark which ones were returned as top-k
        # Build a map of top chunk_ids for quick flagging
        top_ids = {c["chunk_id"] for c in top_chunks if c.get("chunk_id")}
        all_chunks_flagged = []
        for c in all_chunks:
            c_copy = dict(c)
            c_copy["is_top"] = c_copy.get("chunk_id") in top_ids
            # don't attach heavy highlights for non-top chunks
            all_chunks_flagged.append(c_copy)

        return {
            "doc_id": doc["id"],
            "filename": doc["filename"],
            "uploaded_at": doc["uploaded_at"],
            "status": doc["status"],
            "num_chunks": doc["num_chunks"],
            "metadata": doc["metadata"],
            "top_chunks": top_chunks,
            "chunks": all_chunks_flagged,
            "prompt": prompt,
            # note: answering is done in /agent/chat, so /docs/{id}?question=... only shows context & highlights
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /agent/docs/{id}: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ---------------------------
# highlight helper function
# ---------------------------
def _extract_highlights_simple(question: str, text: str, max_snippets: int = 2) -> List[Dict[str, Any]]:
    """
    Heuristic highlight extractor:
      - split text into sentences (simple split on punctuation/newlines)
      - score each sentence by number of overlapping normalized words with the question
      - return top `max_snippets` sentences with start/end char positions and snippet text

    This is fast and dependency-free. For better quality use NLP models / token overlap / attention-based spans.
    """
    import re
    from collections import Counter

    def _normalize(s: str) -> List[str]:
        # lowercase + keep word characters
        return re.findall(r"\w+", s.lower())

    q_tokens = _normalize(question)
    if not q_tokens:
        return []

    q_counter = Counter(q_tokens)

    # split into sentences/segments
    # split on .,!? or newline; keep segments reasonable
    raw_segments = re.split(r'(?<=[\\.\\!\\?\\n])\\s+', text)
    scored = []
    for seg in raw_segments:
        seg_norm = _normalize(seg)
        if not seg_norm:
            continue
        # score = number of shared tokens (could weight by frequency)
        shared = sum(min(q_counter[t], seg_norm.count(t)) for t in set(seg_norm) if t in q_counter)
        if shared > 0:
            scored.append((shared, seg.strip(), seg))

    # sort by score desc and length short-first as tiebreaker
    scored.sort(key=lambda x: (-x[0], len(x[1])))

    highlights = []
    used_ranges = []
    for score, seg_display, seg_original in scored[:max_snippets]:
        # find first occurrence index of segment in text (raw)
        start = text.find(seg_original)
        if start == -1:
            # fallback: try lowercase search
            start = text.lower().find(seg_original.lower())
        if start == -1:
            # give up locating indices; return snippet without indices
            highlights.append({"snippet": seg_display, "start": None, "end": None})
        else:
            end = start + len(seg_original)
            # avoid overlapping snippets (simple check)
            overlap = False
            for (s, e) in used_ranges:
                if not (end <= s or start >= e):
                    overlap = True
                    break
            if overlap:
                continue
            used_ranges.append((start, end))
            highlights.append({"snippet": seg_display, "start": start, "end": end})

    # If nothing scored (no token overlap), include the beginning of text as fallback
    if not highlights and text:
        snippet = text[:200].strip()
        highlights.append({"snippet": snippet, "start": 0, "end": min(len(text), 200)})

    return highlights

