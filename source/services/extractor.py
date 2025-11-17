# backend/services/extractor.py

from typing import List, Dict, Optional, Tuple
import os
import math
import pathlib
import re

try:
    import fitz  # PyMuPDF
except Exception as e:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception as e:
    DocxDocument = None

try:
    import pandas as pd
except Exception as e:
    pd = None


def _raise_if_missing_libs():
    missing = []
    if fitz is None:
        missing.append("PyMuPDF (fitz)")
    if DocxDocument is None:
        missing.append("python-docx")
    if pd is None:
        missing.append("pandas")
    if missing:
        raise RuntimeError(
            "Beberapa library tidak ditemukan: "
            + ", ".join(missing)
            + ".\nInstall dengan pip: pip install pymupdf python-docx pandas"
        )


def _read_pdf(path: str) -> List[Tuple[str, Dict]]:
    """
    Return list of (text, metadata) for each page in a PDF.
    metadata includes page number (1-indexed) and source filename.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) tidak terpasang.")
    doc = fitz.open(path)
    result = []
    filename = os.path.basename(path)
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")  # plain text
        meta = {"source": filename, "page": i + 1}
        # normalize whitespace
        text = re.sub(r"\s+\n", "\n", text).strip()
        result.append((text, meta))
    doc.close()
    return result


def _read_docx(path: str) -> List[Tuple[str, Dict]]:
    """
    Return list of (text, metadata) per paragraph for DOCX.
    Paragraphs that are empty are ignored.
    """
    if DocxDocument is None:
        raise RuntimeError("python-docx tidak terpasang.")
    doc = DocxDocument(path)
    filename = os.path.basename(path)
    result = []
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        meta = {"source": filename, "paragraph": idx + 1}
        result.append((text, meta))
    return result


def _read_csv(path: str) -> List[Tuple[str, Dict]]:
    """
    Return list of (text, metadata) per row for CSV.
    Each row is converted to: "col1: val1 | col2: val2 | ..."
    """
    if pd is None:
        raise RuntimeError("pandas tidak terpasang.")
    df = pd.read_csv(path, dtype=str).fillna("")  # make everything string
    filename = os.path.basename(path)
    result = []
    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns]
        text = " | ".join(parts).strip()
        if text == "":
            continue
        meta = {"source": filename, "row": int(idx)}
        result.append((text, meta))
    return result


def _split_into_units(file_path: str) -> List[Tuple[str, Dict]]:
    """
    Detect extension and return list of (text_unit, metadata).
    Supported: .pdf, .docx, .csv (case-insensitive)
    """
    _raise_if_missing_libs()
    ext = pathlib.Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext in (".docx",):
        return _read_docx(file_path)
    elif ext == ".csv":
        return _read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    min_chunk_size: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    """
    Split `text` into chunks of at most `chunk_size` characters with `overlap` characters overlap.
    Returns list of tuples: (chunk_text, start_char_index, end_char_index).
    Strategy:
      - Greedy by words: don't cut mid-word.
      - Try to split at sentence boundary punctuation (.,;:?) if possible.
    Params:
      - chunk_size: target max characters per chunk (recommended 500-1000)
      - overlap: how many characters to overlap between consecutive chunks
      - min_chunk_size: if set, ensure chunks are at least this long (else grow).
    """
    if min_chunk_size is None:
        min_chunk_size = int(chunk_size * 0.5)

    text = text.strip()
    if not text:
        return []

    n = len(text)
    chunks = []
    start = 0

    # Precompute split points heuristically (sentence boundaries)
    # We'll try to find a split near chunk_size using punctuation; fallback to whitespace.
    while start < n:
        if start + chunk_size >= n:
            # last chunk -> take rest
            end = n
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, start, end))
            break

        # primary candidate
        candidate_end = start + chunk_size

        # try to find nearest sentence boundary BEFORE candidate_end but not before start+min_chunk_size
        window_start = start + min_chunk_size
        window_end = candidate_end
        snippet = text[window_start:window_end]

        # search for last punctuation within snippet
        punct_match = list(re.finditer(r"[\.!\?。！？;:]\s", snippet))
        if punct_match:
            # pick last punctuation position
            pos = punct_match[-1].end() + window_start
            end = pos
        else:
            # try to split at last whitespace before candidate_end but after min_chunk_size
            ws = text[start:candidate_end].rfind(" ")
            if ws > min_chunk_size:
                end = start + ws
            else:
                # fallback: hard cut at candidate_end
                end = candidate_end

        # ensure end > start
        if end <= start:
            end = min(n, start + chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))

        # advance start with overlap
        start = max(start + 1, end - overlap)

    return chunks


def extract_and_chunk(
    file_path: str, chunk_size: int = 800, overlap: int = 200
) -> List[Dict]:
    """
    High-level function:
      - Extract text units from given file
      - For each text unit, produce chunks using chunk_text
    Returns list of chunk dicts with:
      {
        "text": "...",
        "source": filename,
        "origin_meta": { ... },   # e.g. {"page": 1} or {"row": 5}
        "chunk_id": "<filename>_p1_c0",
        "start_char": int,
        "end_char": int
      }
    """
    units = _split_into_units(file_path)
    filename = os.path.basename(file_path)
    all_chunks = []
    for u_idx, (unit_text, origin_meta) in enumerate(units):
        # optional: normalize whitespace to single spaces, preserve newlines
        normalized = re.sub(r"[ \t]+", " ", unit_text).strip()
        unit_chunks = chunk_text(normalized, chunk_size=chunk_size, overlap=overlap)
        for c_idx, (chunk_text_str, start, end) in enumerate(unit_chunks):
            chunk_doc = {
                "text": chunk_text_str,
                "source": filename,
                "origin_meta": origin_meta,  # e.g. {"page": 1} or {"row": 5}
                "chunk_id": f"{filename}_u{u_idx}_c{c_idx}",
                "start_char": int(start),
                "end_char": int(end),
            }
            all_chunks.append(chunk_doc)
    return all_chunks


# Example usage (not executed on import)
# if __name__ == "__main__":
#     example_files = [
#         "sample.pdf",
#         "sample.docx",
#         "sample.csv",
#     ]
#     example_files = ["sample.csv"]
#     for f in example_files:
#         if os.path.exists(f):
#             chunks = extract_and_chunk(f, chunk_size=800, overlap=200)
#             print(f"File {f} produced {len(chunks)} chunks")
#             if chunks:
#                 print("First chunk preview:", chunks[0]["text"][:200])
