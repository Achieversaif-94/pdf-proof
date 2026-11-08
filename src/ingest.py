import os
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def process_pdf(pdf_path: str, collection_name: str = "pdf_docs"):
    print(f"Loading {pdf_path}...")
    
    # 1. Extract text page by page
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({"text": text, "page": page_num + 1})
    
    print(f"Extracted {len(pages)} pages with text.")
    
    # 2. Chunk the text, keeping the page metadata!
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = []
    for p in pages:
        page_chunks = splitter.split_text(p["text"])
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "id": f"page_{p['page']}_chunk_{i}",
                "text": chunk,
                "page": p["page"]
            })
            
    print(f"Split into {len(chunks)} chunks.")
    
    # 3. Embed the chunks
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    
    # 4. Store in ChromaDB
    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    
    # Chroma expects specific formats
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in chunks]
    )
    
    print("✅ Ingestion complete! Database saved to ./chroma_db")

if __name__ == "__main__":
    # Test it with a sample PDF
    # We'll download a test PDF using PowerShell in the next step
    pdf_file = "data/sample.pdf"
    if os.path.exists(pdf_file):
        process_pdf(pdf_file)
    else:
        print(f"Please put a PDF at {pdf_file}")
