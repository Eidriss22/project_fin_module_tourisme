"""Ingestion pipeline: PDF/TXT/MD --> chunks --> Chroma vector store.

Usage:
    python -m src.ingest             # ingest everything under data/corpus/
    python -m src.ingest --reset     # wipe chroma_db/ first

The script is idempotent: re-running it with the same files will replace
the existing collection rather than duplicate documents.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
)
from src.llms import get_embeddings


# File-extension --> loader mapping
# Markdown is loaded as plain text; the recursive splitter preserves
# paragraph/heading structure via the `\n\n` / `\n` separators.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
}


def _load_documents(corpus_dir: Path) -> list[Document]:
    """Walk ``corpus_dir`` and load every supported file as ``Document``s."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loaded = loader_cls(str(path)).load()
            # Tag each page with a clean source path for grounded citations.
            for d in loaded:
                d.metadata.setdefault("source", path.name)
                d.metadata["source_path"] = str(path.relative_to(corpus_dir.parent))
            docs.extend(loaded)
            print(f"  [+] {path.name}  ({len(loaded)} page(s))")
        except Exception as e:  # pragma: no cover - permissive loader
            print(f"  [!] Skipped {path.name}: {e}", file=sys.stderr)
    return docs


def _split(docs: list[Document]) -> list[Document]:
    """Recursive splitter preserves paragraph/sentence boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Order matters: try paragraph -> sentence -> word boundaries.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(reset: bool = False) -> Chroma:
    """Build (or rebuild) the Chroma persistent collection."""
    if reset and CHROMA_DIR.exists():
        print(f"[reset] Removing existing vector store at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] Reading documents from {DATA_DIR} ...")
    raw_docs = _load_documents(DATA_DIR)
    if not raw_docs:
        raise RuntimeError(
            f"No documents found in {DATA_DIR}. Drop tourism PDFs/MD/TXT files in there first."
        )
    print(f"[load] {len(raw_docs)} page-level documents loaded.")

    print("[split] Chunking documents ...")
    chunks = _split(raw_docs)
    print(f"[split] Produced {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("[embed] Computing embeddings and writing to Chroma ...")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[done] Persisted collection '{COLLECTION_NAME}' to {CHROMA_DIR}")
    return vs


def get_vectorstore() -> Chroma:
    """Open the persistent Chroma store for reading at query time."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. Run `python -m src.ingest` first."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vector store.")
    parser.add_argument("--reset", action="store_true", help="Wipe the existing store first.")
    args = parser.parse_args()
    build_vectorstore(reset=args.reset)


if __name__ == "__main__":
    main()
