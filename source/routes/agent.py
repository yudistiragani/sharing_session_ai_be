# backend/routes/agent.py

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
# Ganti fungsi get_doc di backend/routes/agent.py dengan kode berikut

@router.get("/docs/{id}")
async def get_doc(id: str):
    """
    Return document metadata and small stats from Postgres.
    Uses postgres_conn.get_document which returns a plain dict (no ORM objects).
    """
    db = postgres_conn.SessionLocal()
    try:
        doc = postgres_conn.get_document(db, id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /agent/docs/{id}: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

