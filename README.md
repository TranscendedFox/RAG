# 📚 RAG System — Dive into Deep Learning

A Retrieval-Augmented Generation (RAG) system built on the open-source textbook [Dive into Deep Learning](https://d2l.ai/). Ask questions and get answers grounded in the book, with cited sources.

---

## How It Works

```
PDF → PyMuPDF → chunks → HuggingFace embeddings → Chroma (vector DB)
                                                          ↓
                          query → embeddings → similarity search → context
                                                                      ↓
                                                              Groq (LLaMA 3.3) → answer
```

1. **Build** — the PDF is loaded, split into chunks, embedded locally, and stored in a Chroma vector database
2. **Chat** — your question is embedded, the most relevant chunks are retrieved, and sent as context to the LLM

---

## Stack

| Component | Tool |
|---|---|
| PDF extraction | PyMuPDF via LangChain |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector store | Chroma (persisted on disk) |
| LLM | LLaMA 3.3 70B via Groq (free) |

---

## Setup

### 1. Clone & create virtual environment
```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
```
```
GROQ_API_KEY=your_key_here   # free at console.groq.com/keys
```
> Embeddings run **locally** — no API key needed for them.

### 4. Add the PDF
Place `d2l-en.pdf` in the project folder, or let the build script download it automatically.

---

## Usage

### Step 1 — Build the index (run once)
```bash
python build_rag_exercise.py
```
This extracts the first 100 pages, embeds them, and saves the Chroma index to disk. Takes a couple of minutes on first run (model download + embedding). Subsequent runs are instant.

### Step 2 — Start chatting
```bash
python chat_rag_exercise.py
```

Example questions:
- *"What is deep learning according to this textbook?"*
- *"List three challenges in deep learning."*
- *"Explain the role of gradients."*
- *"How does the book describe neural networks?"*

Type `quit` to exit.

---

## Project Structure

```
RAG/
├── build_rag_exercise.py       # PDF → chunks → Chroma index
├── chat_rag_exercise.py        # Interactive RAG chat
├── requirements.txt
├── .env                        # Your API keys (not committed)
├── .env.example                # Safe template to commit
├── .gitignore
├── d2l-en.pdf                  # Source textbook (not committed)
└── local_data/
    └── rag_index_deeplearning/ # Chroma vector store (not committed)
```

---

## Optional Extensions

- **Change page range** — edit `[:100]` in `build_rag_exercise.py` to index more or fewer pages
- **Tune chunk size** — adjust `chunk_size` and `chunk_overlap` in the splitter
- **Retrieve more context** — change `k=4` in `retrieve_context()` for more/fewer source chunks
- **Swap the LLM** — replace Groq with any LangChain-compatible model
