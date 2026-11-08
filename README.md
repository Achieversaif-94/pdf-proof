# 🛡️ PDF Proof — Hallucination-Resistant Document Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions based **strictly** on an uploaded PDF and verifies its own citations. If the AI cannot find the answer in the document, it refuses to answer.

**🔴 Live Web Demo:** [https://pdf-proof.streamlit.app/](https://pdf-proof.streamlit.app/)

## 🏗️ Architecture
```text
PDF Upload
   ↓
PyMuPDF (Extract text + page metadata)
   ↓
Recursive Chunking (Metadata preserved)
   ↓
SentenceTransformers (all-MiniLM-L6-v2 Embeddings)
   ↓
ChromaDB (Local Vector Store)
   ↓
User Query → Semantic Search (Top-K Chunks)
   ↓
Groq API (openai/gpt-oss-20b) with STRICT prompt
   ↓
Verifier (Parses [Page X] citations and checks retrieved chunks)
   ↓
Streamlit UI (Green/Red Trust Badge + Proof Cards)
🧠 The Three Pillars of "Proof"
Most "Chat with PDF" apps hallucinate or give vague answers. pdf-proof solves this with three layers of defense:

Metadata-Preserving Chunking: Every text chunk is tagged with its source page number during ingestion.

Strict Prompt Engineering: The LLM is instructed to cite [Page X] after every sentence and to refuse to answer if the context does not contain the information.

Post-Generation Verification: A separate verifier script parses the [Page X] citations. If the LLM hallucinates a page number that wasn't in the retrieved context, it throws a Red Warning.

📊 Model Stack
Embedding Model: all-MiniLM-L6-v2 (fast CPU inference, 384 dimensions)

Vector DB: ChromaDB (persistent local storage)

LLM: openai/gpt-oss-20b via Groq API (ultra-fast inference, 0.0 temperature)

PDF Parsing: PyMuPDF

⚠️ Limitations (Honest Engineering Caveats)
Semantic Drift: Semantic search can fail on broad queries. For example, asking "Who are the authors?" may retrieve the bibliography section (which contains a list of names) instead of the title page. Because the system strictly verifies citations, it correctly refuses to answer rather than hallucinating the authors.

Table Extraction: Complex multi-column tables may be mangled during text extraction, leading to disjointed context chunks (e.g., the PosturePulse hardware pin mapping tables).

Scanned PDFs: PyMuPDF extracts text, not images. Scanned/image-based PDFs will result in empty text extraction (OCR required for future work).

Free-Tier Rate Limits: Groq's free tier has rate limits; heavy concurrent use may result in API errors.
