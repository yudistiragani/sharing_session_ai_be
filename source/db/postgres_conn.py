import os
from settings import settings
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# import models from backend.models.document
from models.document import Base, Document, Chunk, IndexJob

DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:"
    f"{settings.POSTGRES_PASSWORD}@"
    f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
    f"{settings.POSTGRES_DB}"
)

# Create engine & session factory
engine = create_engine(DATABASE_URL, pool_size=int(os.getenv("PG_MAX_CONNECTIONS", 10)), echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Create tables (dev convenience). In production use migrations (Alembic).
    """
    Base.metadata.create_all(bind=engine)


# -------------------------
# Document & Chunk helpers
# -------------------------
def create_document(db, doc_id: str, filename: str, num_chunks: int, metadata: Optional[Dict] = None) -> Document:
    """
    Create a Document row.
    - metadata param will be stored in column 'metadata' but Python attribute is doc_metadata.
    """
    doc = Document(id=doc_id, filename=filename, uploaded_at=datetime.utcnow(), status="uploaded", num_chunks=num_chunks)
    if metadata is not None:
        doc.doc_metadata = metadata
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def add_chunks_bulk(db, chunks: List[Dict[str, Any]]) -> int:
    """
    Bulk insert chunks.
    chunks: list of dicts with keys: chunk_id, doc_id, text, source, origin_meta
    Returns number of inserted rows.
    """
    objs = []
    for c in chunks:
        obj = Chunk(
            id=c["chunk_id"],
            doc_id=c["doc_id"],
            text=c["text"],
            origin_meta=c.get("origin_meta"),
            source=c.get("source"),
            created_at=datetime.utcnow(),
            indexed=False,
        )
        objs.append(obj)
    if not objs:
        return 0
    db.bulk_save_objects(objs)
    db.commit()
    return len(objs)


def get_chunks_by_doc_id(db, doc_id: str) -> List[Dict[str, Any]]:
    """
    Return list of chunk dicts for given doc_id, ordered by created_at.
    """
    rows = db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(Chunk.created_at).all()
    result = []
    for r in rows:
        result.append({
            "chunk_id": r.id,
            "doc_id": r.doc_id,
            "text": r.text,
            "origin_meta": r.origin_meta,
            "source": r.source,
            "indexed": r.indexed,
        })
    return result


def mark_chunks_indexed(db, chunk_ids: List[str]) -> int:
    """
    Mark chunks (by id list) as indexed=True.
    Returns number of rows updated.
    """
    if not chunk_ids:
        return 0
    q = db.query(Chunk).filter(Chunk.id.in_(chunk_ids))
    updated = q.update({"indexed": True}, synchronize_session=False)
    db.commit()
    return updated


# -------------------------
# Index job helpers
# -------------------------
def create_index_job(db, job_id: str, doc_id: Optional[str]):
    job = IndexJob(id=job_id, doc_id=doc_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_index_job_status(db, job_id: str, status: str, details: Optional[Dict] = None):
    """
    Update job status and timestamps.
    status: queued/running/success/failed
    """
    job = db.get(IndexJob, job_id)
    if not job:
        return None
    if status == "running":
        job.started_at = datetime.utcnow()
    if status in ("success", "failed"):
        job.finished_at = datetime.utcnow()
    job.status = status
    if details is not None:
        job.details = details
    db.commit()
    db.refresh(job)
    return job


def get_index_job(db, job_id: str) -> Optional[Dict[str, Any]]:
    job = db.get(IndexJob, job_id)
    if not job:
        return None
    return {
        "id": job.id,
        "doc_id": job.doc_id,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "details": job.details,
    }


# -------------------------
# Document helpers
# -------------------------
def get_document(db, doc_id: str) -> Optional[Dict[str, Any]]:
    doc = db.get(Document, doc_id)
    if not doc:
        return None
    return {
        "id": doc.id,
        "filename": doc.filename,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        "status": doc.status,
        "num_chunks": doc.num_chunks,
        # expose doc_metadata (python attr) as 'metadata' in returned dict for compatibility
        "metadata": doc.doc_metadata if hasattr(doc, "doc_metadata") else None,
    }