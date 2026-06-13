"""
Ingestion script: reads all PDFs and transcripts, chunks, embeds via Azure OpenAI,
and upserts into Supabase knowledge_chunks table.
"""

import os
import re
import time
import hashlib
from pathlib import Path

from PyPDF2 import PdfReader
from openai import AzureOpenAI
from supabase import create_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_KEY"]
AZURE_ENDPOINT   = os.environ["AZURE_ENDPOINT"]
AZURE_KEY        = os.environ["AZURE_KEY"]
AZURE_API_VER    = "2024-02-01"
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-small")

CHUNK_SIZE    = 500   # words per chunk
CHUNK_OVERLAP = 50    # words overlap between chunks

BASE_DIR = Path(__file__).parent

# ── Clients ───────────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
az = AzureOpenAI(api_key=AZURE_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VER)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    # Azure rate-limits; batch in groups of 16
    all_vecs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        resp = az.embeddings.create(input=batch, model=EMBED_DEPLOYMENT)
        all_vecs.extend([d.embedding for d in resp.data])
        time.sleep(0.3)
    return all_vecs

def source_label(path: Path) -> str:
    # e.g. "Whop/Presentation/PILLAR 1 - Build Rapport.pdf" → clean label
    return str(path.relative_to(BASE_DIR))

# ── Collect files ─────────────────────────────────────────────────────────────
files: list[Path] = []
files += list(BASE_DIR.glob("Whop/**/*.pdf"))
files += list(BASE_DIR.glob("transcripts/**/*.txt"))

print(f"Found {len(files)} files to ingest")

# ── Process ───────────────────────────────────────────────────────────────────
total_chunks = 0

for fpath in files:
    print(f"\nProcessing: {source_label(fpath)}")
    try:
        raw = extract_pdf(fpath) if fpath.suffix == ".pdf" else extract_txt(fpath)
    except Exception as e:
        print(f"  ✗ Could not read: {e}")
        continue

    # Clean up whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    if len(raw) < 50:
        print(f"  ✗ Too short, skipping")
        continue

    chunks = chunk_text(raw)
    print(f"  → {len(chunks)} chunks")

    try:
        vectors = embed(chunks)
    except Exception as e:
        print(f"  ✗ Embedding failed: {e}")
        continue

    rows = []
    for chunk_text_val, vec in zip(chunks, vectors):
        chunk_id = hashlib.md5(f"{fpath}{chunk_text_val[:80]}".encode()).hexdigest()
        rows.append({
            "id": chunk_id,
            "content": chunk_text_val,
            "embedding": vec,
            "source": source_label(fpath),
        })

    try:
        supabase.table("knowledge_chunks").upsert(rows, on_conflict="id").execute()
        total_chunks += len(rows)
        print(f"  ✓ Upserted {len(rows)} chunks")
    except Exception as e:
        print(f"  ✗ Supabase upsert failed: {e}")

print(f"\n✅ Done. Total chunks ingested: {total_chunks}")
