import pathlib
import shutil
import urllib.request
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(r"D:\AI Study\RAG\local_data")
INDEX_DIR = str(DATA_DIR / "rag_index_deeplearning")
COLLECTION = "deeplearning"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PDF_URL = "https://d2l.ai/d2l-en.pdf"
SAMPLE_PDF_NAME = "d2l-en.pdf"
PDF_PATH = DATA_DIR / SAMPLE_PDF_NAME


# ── Step 2 – Prepare the PDF ───────────────────────────────────────────────
def ensure_pdf() -> str:
    """Use local PDF if available, otherwise download it."""
    if PDF_PATH.exists():
        print(f"✔ PDF found at {PDF_PATH}")
        return str(PDF_PATH)

    # Check if the user already has it in the project root under a different name
    fallback = pathlib.Path("./d2l-en.pdf")
    if fallback.exists():
        print(f"✔ Found {fallback} — copying to {PDF_PATH}")
        shutil.copy(fallback, PDF_PATH)
        return str(PDF_PATH)

    # Download as a last resort
    print(f"Downloading PDF from {SAMPLE_PDF_URL} ...")
    urllib.request.urlretrieve(SAMPLE_PDF_URL, PDF_PATH)
    print(f"✔ Downloaded to {PDF_PATH}")
    return str(PDF_PATH)


# ── Step 3 – Load the First 100 Pages ─────────────────────────────────────
def load_pdf_chunks(path: str):
    pages = PyMuPDFLoader(path).load()[:100]
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    # add stable metadata keys
    for i, c in enumerate(chunks):
        c.metadata.setdefault("source", path)
        c.metadata["chunk_id"] = i
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = ensure_pdf()
    docs = load_pdf_chunks(pdf_path)

    # Quick sanity check
    print(f"\n── Sample from page 1 ──")
    print(docs[0].page_content[:300])
    print(f"\n── Metadata ──")
    print(docs[0].metadata)
