import os
import pathlib
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(r"D:\AI Study\RAG\local_data")
INDEX_DIR = str(DATA_DIR / "rag_index_deeplearning")
COLLECTION = "deeplearning"

SYSTEM_RULES = (
    "You are a concise deep learning assistant.\n"
    "- Always base answers ONLY on the provided CONTEXT.\n"
    "- If context is insufficient, say so.\n"
    "- Keep answers to ≤6 bullets/sentences.\n"
    "- Add bracketed citations [1], [2], ... that map to the numbered sources."
)

# ── Load Vectorstore ───────────────────────────────────────────────────────
def load_vectorstore():
    """Load the persisted Chroma index from disk."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=INDEX_DIR,
    )
    print(f"✔ Vectorstore loaded from {INDEX_DIR}")
    return vectorstore

# ── Retrieve Context ───────────────────────────────────────────────────────
def retrieve_context(vectorstore, query: str, k: int = 4):
    """Retrieve the top-k most relevant chunks for a query."""
    results = vectorstore.similarity_search(query, k=k)
    context_parts = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{i}] (Page {page}):\n{doc.page_content}")
    return "\n\n".join(context_parts), results

# ── Ask Groq ──────────────────────────────────────────────────────────────
def ask(llm, query: str, context: str) -> str:
    """Send the query + retrieved context to Groq and return the answer."""
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    messages = [
        SystemMessage(content=SYSTEM_RULES),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content

# ── Chat Loop ──────────────────────────────────────────────────────────────
def chat():
    vectorstore = load_vectorstore()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    print("\n✔ Deep Learning Assistant ready!")
    print("  Ask anything about the textbook. Type 'quit' to exit.\n")
    print("─" * 60)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        context, sources = retrieve_context(vectorstore, query)
        answer = ask(llm, query, context)

        print(f"\nAssistant: {answer}")
        print(f"\n── {len(sources)} chunks retrieved from the index ──")


if __name__ == "__main__":
    chat()
