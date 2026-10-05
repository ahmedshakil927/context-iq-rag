import glob
import json
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import ingest, llm, rag
from .config import ACTIVE_MODEL, CHUNK_SIZE, KEYWORD_GATE, LLM_PROVIDER, SEARCH_MODE, TOP_K
from .store import VectorStore

ALLOWED = (".pdf", ".docx", ".txt", ".md")

app = FastAPI(title="Context IQ")
store = VectorStore()


class Ask(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse("app/index.html")


@app.get("/quality")
def quality_page():
    return FileResponse("app/quality.html")


@app.get("/eval/results")
def eval_results():
    """Latest result per config from eval/results.jsonl, plus test-set facts."""
    runs = {}
    if os.path.exists("eval/results.jsonl"):
        for line in open("eval/results.jsonl"):
            if line.strip():
                r = json.loads(line)
                runs[r["label"]] = r  # later runs replace earlier ones
    qs = json.load(open("eval/questions.json")) if os.path.exists("eval/questions.json") else []
    # the eval row that matches the setup this app is running right now
    # (rows from before the model field existed all used the 3B model)
    running = next((r["label"] for r in runs.values()
                    if r["mode"] == SEARCH_MODE and r["k"] == TOP_K and r["size"] == CHUNK_SIZE
                    and r.get("model", "llama3.2:3b") == ACTIVE_MODEL
                    and r.get("provider", "ollama") == LLM_PROVIDER
                    and r.get("kw_gate", False) == KEYWORD_GATE), None)
    return {
        "running": running,
        "results": list(runs.values()),
        "answerable": sum(1 for q in qs if q["expect"]),
        "unanswerable": sum(1 for q in qs if not q["expect"]),
        "documents": sorted(os.path.basename(p)[:-4] for p in glob.glob("docs/*.pdf")),
    }


@app.get("/config")
def config():
    return {"model": ACTIVE_MODEL, "provider": LLM_PROVIDER, "search": SEARCH_MODE, "top_k": TOP_K}


@app.get("/documents")
def documents():
    return store.documents()


@app.delete("/documents/{name}")
def delete_document(name: str):
    if not store.remove(name):
        raise HTTPException(404, f"No document named {name}")
    store.save()
    return {"removed": name}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(ALLOWED):
        raise HTTPException(415, f"Unsupported file type. Use: {', '.join(ALLOWED)}")
    data = await file.read()
    try:
        pages = ingest.extract_pages(file.filename, data)
        new_items = []
        for page, text in pages:
            chunks = ingest.chunk_text(text)
            if chunks:
                new_items += [(page, c, v) for c, v in zip(chunks, llm.embed(chunks))]
    except Exception as e:  # unreadable file or Ollama not reachable
        raise HTTPException(502, f"Could not index {file.filename}: {e}")
    if not new_items:
        raise HTTPException(422, "No text found in this file (is it a scanned PDF?)")
    store.remove(file.filename)  # re-uploading replaces the old version
    for page, chunk, vec in new_items:
        store.add(file.filename, page, chunk, vec)
    store.save()
    return {"document": file.filename, "chunks": len(new_items)}


@app.post("/ask")
def ask(body: Ask):
    r = rag.answer(store, body.question)
    return {"answer": r["answer"], "sources": r["sources"]}


@app.post("/ask/stream")
def ask_stream(body: Ask):
    """Newline-delimited JSON events: sources, token..., done (or error)."""
    def events():
        try:
            for ev in rag.answer_stream(store, body.question):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")
