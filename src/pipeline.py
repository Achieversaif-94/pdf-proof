from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier

class PDFProofPipeline:
    def __init__(self):
        print("Initializing Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator()
        self.verifier = Verifier()

    def ask(self, query: str, top_k: int = 5) -> dict:  # INCREASED TO 5
        chunks = self.retriever.search(query, top_k=top_k)
        answer = self.generator.generate(query, chunks)
        verification = self.verifier.verify(answer, chunks)
        
        return {
            "query": query,
            "answer": answer,
            "verification": verification,
            "sources": chunks
        }
