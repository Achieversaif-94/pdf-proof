import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-20b"

    def build_prompt(self, query: str, chunks: list) -> str:
        context_text = ""
        for chunk in chunks:
            context_text += f"\n--- Context from Page {chunk['page']} ---\n"
            context_text += chunk['text'] + "\n"

        prompt = f"""You are a strict document Q&A assistant.
Answer the user's question using ONLY the provided context chunks.

RULES:
1. If the answer is not in the context, reply EXACTLY: "I cannot find this in the document."
2. Do not use any outside knowledge.
3. You MUST include a citation at the end of EVERY sentence. The citation format is exactly: [Page X].
4. Keep the answer concise and direct.

Example of perfect formatting:
The Transformer (big) model achieved a BLEU score of 28.4 on English-to-German [Page 8]. It achieved 41.8 on English-to-French [Page 8].

{context_text}

Question: {query}
Answer:"""
        return prompt

    def generate(self, query: str, chunks: list) -> str:
        prompt = self.build_prompt(query, chunks)
        
        print(f"Calling Groq API ({self.model})...")
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict Q&A bot. You MUST cite every sentence. Failure to cite is a critical error."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=500,
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    from retriever import Retriever
    
    query = "What is the BLEU score achieved by the Transformer model?"
    
    retriever = Retriever()
    chunks = retriever.search(query, top_k=2)
    
    generator = Generator()
    answer = generator.generate(query, chunks)
    
    print("\n" + "="*50)
    print("🤖 AI ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)
