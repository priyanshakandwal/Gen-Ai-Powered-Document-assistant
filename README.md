📄 AI Document Assistant using RAG & Prompt Engineering

An AI-powered **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF or TXT documents and ask natural language questions. The system retrieves the most relevant document sections and generates context-aware answers using **Groq's LLaMA 3.3 70B** model.

> Developed using FastAPI, Groq API, Prompt Engineering, and a lightweight retrieval pipeline.

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- 🤖 Ask questions in natural language
- 🔍 Context-aware Retrieval-Augmented Generation (RAG)
- 🧠 Supports two Prompt Engineering techniques:
  - Chain-of-Thought Prompting
  - Few-Shot Prompting
- 📚 Displays answers based on uploaded document content
- ⚡ Fast inference using Groq LLaMA 3.3 70B

---

## 🧠 What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) combines information retrieval with Large Language Models.

The workflow is:

1. Upload a document.
2. Split the document into manageable chunks.
3. Generate lightweight term-frequency embeddings.
4. Retrieve the most relevant chunks using cosine similarity.
5. Provide the retrieved context to the LLM.
6. Generate an accurate, context-aware response.

This approach reduces hallucinations by grounding responses in the uploaded document.

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Backend | FastAPI |
| API Server | Uvicorn |
| LLM | Groq API (LLaMA 3.3 70B Versatile) |
| Document Processing | PyPDF |
| Retrieval | TF-based Embeddings + Cosine Similarity |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker |

---

## 🔬 Prompt Engineering

### Chain-of-Thought Prompting

Encourages the language model to reason through the problem before producing the answer.

Example:

```text
1. Find relevant information.
2. Analyze the context.
3. Generate the final answer.
```

---

### Few-Shot Prompting

Provides example question-answer pairs that guide the model toward the expected response format.

Example:

```text
Context: The company was founded in 2010.
Question: When was it founded?
Answer: The company was founded in 2010.
```

---

## 📂 Project Structure

```text
gen-ai-doc-assistant/
│
├── backend/
│   ├── main.py
│   └── rag_pipeline.py
│
├── frontend/
│   └── index.html
│
├── data/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Installation

```bash
git clone 

cd gen-ai-doc-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Set your Groq API Key.

**Windows PowerShell**

```powershell
$env:GROQ_API_KEY="your_api_key"
```

Run the application.

```bash
cd backend
uvicorn main:app --reload --port 7000
```

Open:

```
http://localhost:7000
```

---

## 📡 REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/upload` | Upload a PDF or TXT document |
| POST | `/query` | Ask questions from uploaded documents |
| GET | `/documents` | View uploaded documents |
| DELETE | `/documents` | Remove uploaded documents |

---

## 🔄 Workflow

```text
Upload Document
        │
        ▼
Extract Text
        │
        ▼
Split into Chunks
        │
        ▼
Generate TF Embeddings
        │
        ▼
Store Chunks
        │
        ▼
User Question
        │
        ▼
Retrieve Top Relevant Chunks
        │
        ▼
Prompt Engineering
(CoT / Few-Shot)
        │
        ▼
Groq LLaMA 3.3
        │
        ▼
Generate Answer
```

---

## 🎯 Key Features

- Retrieval-Augmented Generation pipeline
- TF-based document retrieval
- Cosine similarity search
- Chain-of-Thought Prompting
- Few-Shot Prompting
- PDF and TXT support
- FastAPI REST API
- Groq LLaMA integration

---

## 📖 Future Improvements

- Vector database integration (FAISS/ChromaDB)
- Semantic embeddings using Sentence Transformers
- Multi-document search
- Chat history
- Streaming responses
- Support for DOCX and Markdown files

---

## 👩‍💻 Author

**Priyansha Kandwal**

B.Tech Computer Science Engineering

Interested in Artificial Intelligence, Machine Learning, Full Stack Development, and Cloud Computing.
