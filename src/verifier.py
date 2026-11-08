import re

class Verifier:
    def __init__(self):
        pass

    def verify(self, answer: str, retrieved_chunks: list) -> dict:
        """
        Checks if the answer cites pages that were actually retrieved.
        """
        print("\nRunning Verification...")
        
        # 1. Find all citations in the format [Page X]
        citations = re.findall(r'\[Page (\d+)\]', answer)
        
        if not citations:
            return {
                "is_valid": False,
                "reason": "No citations found in the answer.",
                "confidence": 0.0
            }
        
        # 2. Get the list of pages we actually retrieved
        retrieved_pages = [str(chunk['page']) for chunk in retrieved_chunks]
        
        # 3. Check if every cited page was actually in the retrieved context
        hallucinated_citations = []
        for citation in citations:
            if citation not in retrieved_pages:
                hallucinated_citations.append(citation)
                
        # 4. Calculate confidence based on valid citations
        total_citations = len(citations)
        valid_citations = total_citations - len(hallucinated_citations)
        confidence = (valid_citations / total_citations) * 100 if total_citations > 0 else 0.0
        
        is_valid = len(hallucinated_citations) == 0
        
        return {
            "is_valid": is_valid,
            "reason": "All citations are grounded in the retrieved context." if is_valid else f"Hallucinated pages: {hallucinated_citations}",
            "confidence": confidence,
            "cited_pages": list(set(citations))
        }

if __name__ == "__main__":
    # Test it
    v = Verifier()
    
    # Fake retrieved chunks
    chunks = [{"page": 8, "text": "..."}, {"page": 9, "text": "..."}]
    
    # Test 1: Good answer
    good_answer = "The model got 28.4 [Page 8]."
    print("Test 1:", v.verify(good_answer, chunks))
    
    # Test 2: Bad answer (Hallucinated page 12)
    bad_answer = "The model got 28.4 [Page 8] and was invented on [Page 12]."
    print("Test 2:", v.verify(bad_answer, chunks))
