from . import llm
from .config import KEYWORD_GATE, KW_COVER, MIN_SCORE, SEARCH_MODE, TOP_K

REFUSAL = "I don't know based on the provided documents."

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know based on the provided documents.

Context:
{context}

Question: {question}
Answer:"""


def retrieve(store, question, k=TOP_K, min_score=MIN_SCORE, mode=SEARCH_MODE, kw_gate=KEYWORD_GATE):
    """Return (prompt, retrieved). prompt is None when nothing relevant was found.
    'retrieved' is always filled (even when we refuse) so the eval can score retrieval."""
    qvec = llm.embed([question])[0]
    hits, best, (cover, matched) = store.search(qvec, question, k, mode)
    retrieved = [{"doc": i["doc"], "page": i["page"], "score": round(s, 3), "text": i["text"]}
                 for i, s in hits]
    relevant = best >= min_score or (kw_gate and matched >= 2 and cover >= KW_COVER)
    if not hits or not relevant:
        return None, retrieved
    context = "\n\n".join(f"[{h['doc']} p.{h['page']}] {h['text']}" for h in retrieved)
    return PROMPT.format(context=context, question=question), retrieved


def answer(store, question, k=TOP_K, min_score=MIN_SCORE, mode=SEARCH_MODE, model=None, kw_gate=KEYWORD_GATE):
    prompt, retrieved = retrieve(store, question, k, min_score, mode, kw_gate)
    if prompt is None:
        return {"answer": REFUSAL, "sources": [], "retrieved": retrieved}
    text = llm.generate(prompt, **({"model": model} if model else {}))
    return {"answer": text, "sources": retrieved, "retrieved": retrieved}


def answer_stream(store, question, k=TOP_K, min_score=MIN_SCORE, mode=SEARCH_MODE):
    """Yield events: sources first, then answer tokens, then done."""
    prompt, retrieved = retrieve(store, question, k, min_score, mode)
    if prompt is None:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": REFUSAL}
    else:
        yield {"type": "sources", "sources": retrieved}
        for token in llm.generate_stream(prompt):
            yield {"type": "token", "text": token}
    yield {"type": "done"}
