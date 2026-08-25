"""Document loading (PDF incl. Persian / DOCX / TXT), cleaning and splitting."""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

_ZWNJ = "\u200c"
_DIRECTIONAL_MARKS = ["\u200f", "\u200e"]


def load_document(file_path: str | Path) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        try:
            return TextLoader(str(path), encoding="utf-8").load()
        except UnicodeDecodeError:
            return TextLoader(str(path), encoding="latin-1").load()
    return loader.load()


def clean_text(text: str) -> str:
    """Light, meaning-preserving cleaning suitable for Persian and English."""
    if not text:
        return text
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    for ch in _DIRECTIONAL_MARKS:
        text = text.replace(ch, "")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    cleaned = []
    for d in docs:
        d.page_content = clean_text(d.page_content)
        if d.page_content:
            cleaned.append(d)
    return cleaned


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", ".\u200c", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    seen: dict = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = seen.get(src, 0)
        seen[src] = idx + 1
        chunk.metadata["chunk_id"] = f"{src}::{idx}"
        chunk.metadata["chunk_index"] = idx
    return chunks
