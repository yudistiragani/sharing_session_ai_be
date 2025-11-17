from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)  # doc_id (uuid string)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")  # uploaded/indexing/indexed/failed
    num_chunks = Column(Integer, default=0)
    # use doc_metadata as python attribute name to avoid collision with Base.metadata
    doc_metadata = Column("metadata", JSON, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, index=True)  # chunk_id
    doc_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    text = Column(Text, nullable=False)
    origin_meta = Column(JSON, nullable=True)
    source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    indexed = Column(Boolean, default=False)

    # relationship backref (optional)
    document = relationship("Document", backref="chunks")


class IndexJob(Base):
    __tablename__ = "index_jobs"

    id = Column(String, primary_key=True, index=True)  # job_id (uuid)
    doc_id = Column(String, ForeignKey("documents.id", ondelete="SET NULL"), index=True, nullable=True)
    status = Column(String, default="queued")  # queued/running/success/failed
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    details = Column(JSON, nullable=True)
