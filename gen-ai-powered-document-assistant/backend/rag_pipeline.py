import os
from pathlib import Path
from groq import Groq

groq_client = None

def get_client():
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        groq_client = Groq(api_key=api_key)
    return groq_client

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = sum(x**2 for x in a)**0.5
    mag_b = sum(x**2 for x in b)**0.5
    return dot / (mag_a * mag_b + 1e-10)

def get_embedding(text):
    # Simple TF-based embedding (no API needed)
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    return vocab

def dict_similarity(a, b):
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k,0) * b.get(k,0) for k in keys)
    mag_a = sum(v**2 for v in a.values())**0.5
    mag_b = sum(v**2 for v in b.values())**0.5
    return dot / (mag_a * mag_b + 1e-10)

FEW_SHOT_TEMPLATE = """You are a helpful document assistant. Use the context below to answer accurately.

Example 1:
Context: "The company was founded in 2010."
Question: When was it founded?
Answer: The company was founded in 2010.

Example 2:
Context: "Revenue grew by 25% reaching $4.2 billion in FY2023."
Question: What was the revenue?
Answer: The revenue was $4.2 billion in FY2023.

Now answer:
Context:
{context}

Question: {question}
Answer:"""

CHAIN_OF_THOUGHT_TEMPLATE = """You are a precise document assistant. Think step-by-step.

Context:
{context}

Question: {question}

Steps:
1. Find relevant info in context.
2. Reason through the question.
3. Give a clear answer based only on context.

Final Answer:"""

class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.chunks = []
        self.embeddings = []

    def ingest_document(self, file_path: str) -> int:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                raise ValueError(f"PDF read failed: {e}")
        elif path.suffix.lower() == ".txt":
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported: {path.suffix}")

        # Split into chunks
        size, overlap = 400, 40
        words = text.split()
        raw_chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+size])
            raw_chunks.append({"text": chunk, "source": path.name})
            i += size - overlap

        for chunk in raw_chunks:
            emb = get_embedding(chunk["text"])
            self.chunks.append(chunk)
            self.embeddings.append(emb)

        return len(raw_chunks)

    def query(self, question: str, use_chain_of_thought: bool = True) -> dict:
        if not self.chunks:
            raise RuntimeError("No documents indexed.")

        q_emb = get_embedding(question)
        scores = [dict_similarity(q_emb, e) for e in self.embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
        context = "\n\n".join(self.chunks[i]["text"] for i in top_indices)
        sources = list({self.chunks[i]["source"] for i in top_indices})

        template = CHAIN_OF_THOUGHT_TEMPLATE if use_chain_of_thought else FEW_SHOT_TEMPLATE
        technique = "Chain-of-Thought Prompting" if use_chain_of_thought else "Few-Shot Prompting"
        prompt = template.replace("{context}", context).replace("{question}", question)

        c = get_client()
        response = c.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()

        return {"answer": answer, "sources": sources, "prompt_technique": technique}

    def has_documents(self) -> bool:
        return len(self.chunks) > 0

    def reset(self):
        self.chunks = []
        self.embeddings = []