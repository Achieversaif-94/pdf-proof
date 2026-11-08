from retriever import Retriever
from generator import Generator
from verifier import Verifier

class PDFProofPipeline:
    def __init__(self):
        print("Initializing Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator()
        self.verifier = Verifier()

    def ask(self, query: str, top_k: int = 3) -> dict:
        # 1. Retrieve
        chunks = self.retriever.search(query, top_k=top_k)
        
        # 2. Generate
        answer = self.generator.generate(query, chunks)
        
        # 3. Verify
        verification = self.verifier.verify(answer, chunks)
        
        return {
            "query": query,
            "answer": answer,
            "verification": verification,
            "sources": chunks
        }

if __name__ == "__main__":
    pipeline = PDFProofPipeline()
    
    # Test 1: A question that IS in the document
    print("\n" + "="*60)
    print("TEST 1: Valid Question")
    print("="*60)
    result = pipeline.ask("What is the BLEU score of the Transformer?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Verification: {result['verification']}")
    
    # Test 2: A question that is NOT in the document
    print("\n" + "="*60)
    print("TEST 2: Invalid Question (Hallucination Test)")
    print("="*60)
    result = pipeline.ask("What is the capital of France?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Verification: {result['verification']}")
