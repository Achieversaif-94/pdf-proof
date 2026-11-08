import chromadb
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, collection_name: str = "pdf_docs", db_path: str = "./chroma_db"):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"Connected! Found {self.collection.count()} chunks in the database.")

    def search(self, query: str, top_k: int = 3):
        """
        Takes a query, embeds it, and returns the top_k most relevant chunks.
        """
        print(f"\nSearching for: '{query}'")
        
        # 1. Embed the query using the SAME model used for ingestion
        query_embedding = self.embedder.encode([query]).tolist()
        
        # 2. Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # 3. Format the results nicely
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "page": results['metadatas'][0][i]['page'],
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results

if __name__ == "__main__":
    # Test the retriever!
    retriever = Retriever()
    
    # Ask a question related to the "Attention Is All You Need" paper
    test_query = "What is the BLEU score achieved by the Transformer model?"
    results = retriever.search(test_query, top_k=2)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Page {res['page']}) ---")
        print(f"ID: {res['id']}")
        # Print a small snippet of the text so the terminal isn't flooded
        print(f"Text snippet: {res['text'][:300]}...")
