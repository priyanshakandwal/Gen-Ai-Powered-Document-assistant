from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from pathlib import Path

from rag_pipeline import RAGPipeline

app = FastAPI(title="Gen-AI Document Assistant", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

UPLOAD_DIR = Path(__file__).parent.parent / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

rag = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    use_chain_of_thought: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    prompt_technique: str


@app.get("/")
async def serve_frontend():
    return FileResponse(str(frontend_path / "index.html"))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT document and index it."""
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Index the document
    chunk_count = rag.ingest_document(str(file_path))

    return {
        "message": f"Document '{file.filename}' uploaded and indexed successfully.",
        "chunks_indexed": chunk_count,
        "filename": file.filename
    }


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Ask a question about the uploaded documents."""
    if not rag.has_documents():
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload a document first.")

    result = rag.query(
        question=request.question,
        use_chain_of_thought=request.use_chain_of_thought
    )
    return result


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": files, "count": len(files)}


@app.delete("/documents")
async def clear_documents():
    """Clear all documents and reset the index."""
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
    rag.reset()
    return {"message": "All documents cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
