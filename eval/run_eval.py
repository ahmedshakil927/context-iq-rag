"""Score the RAG pipeline on eval/questions.json over the PDFs in docs/.

Usage: .venv/bin/python eval/run_eval.py [config_label ...]   (default: all)
Builds its own index (independent of the running server) and caches embeddings
in eval/embed_cache.json. Results are appended to eval/results.jsonl.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, ".")
from app import ingest, llm, rag  # noqa: E402
from app.config import GROQ_MODEL  # noqa: E402
from app.store import VectorStore  # noqa: E402

CACHE_PATH = "eval/embed_cache.json"
REFUSAL_PHRASES = ["don't know", "do not know", "not mention", "no information",
                   "not provided", "does not contain", "doesn't contain", "cannot answer"]

CONFIGS = {
    "baseline":     dict(size=800,  overlap=150, k=4, min_score=0.45, mode="vector"),
    "small-chunks": dict(size=400,  overlap=80,  k=4, min_score=0.45, mode="vector"),
    "big-chunks":   dict(size=1200, overlap=200, k=4, min_score=0.45, mode="vector"),
    "top6":         dict(size=800,  overlap=150, k=6, min_score=0.45, mode="vector"),
    "hybrid":       dict(size=800,  overlap=150, k=4, min_score=0.45, mode="hybrid"),
    "hybrid-top6":  dict(size=800,  overlap=150, k=6, min_score=0.45, mode="hybrid"),
    "hybrid-top8":  dict(size=800,  overlap=150, k=8, min_score=0.45, mode="hybrid"),
    "hybrid-kwgate": dict(size=800,  overlap=150, k=8, min_score=0.45, mode="hybrid", kw_gate=True),
    "llama8b-kwgate": dict(size=800, overlap=150, k=8, min_score=0.45, mode="hybrid", model="llama3.1:8b", kw_gate=True),
    "kwgate-groq":  dict(size=800,  overlap=150, k=8, min_score=0.45, mode="hybrid", kw_gate=True, provider="groq"),
    "llama8b":      dict(size=800,  overlap=150, k=8, min_score=0.45, mode="hybrid", model="llama3.1:8b"),
}

_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def embed_cached(texts):
    keys = [hashlib.sha1((llm.EMBED_MODEL + t).encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _cache]
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        for (k, _), v in zip(batch, llm.embed([t for _, t in batch])):
            _cache[k] = v
    return [_cache[k] for k in keys]


def build_store(cfg):
    store = VectorStore(load=False)
    for path in sorted(glob.glob("docs/*.pdf")):
        name = os.path.basename(path)[:-4]
        for page, text in ingest.extract_pages(path, open(path, "rb").read()):
            chunks = ingest.chunk_text(text, cfg["size"], cfg["overlap"])
            for c, v in zip(chunks, embed_cached(chunks)):
                store.add(name, page, c, v)
    return store


def has(text, kws):
    """Case-insensitive substring match; short keywords (<=3 chars, e.g. '8')
    must be whole tokens so '8' does not match '28.4'."""
    t = text.lower()
    for k in kws:
        k = k.lower()
        pat = re.escape(k)
        if len(k) <= 3:
            pat = r"(?<![\w.])" + pat + r"(?![\w.]*\d)(?!\w)"
        if re.search(pat, t):
            return True
    return False


def run(label, cfg, qs):
    llm.set_provider(cfg.get("provider", "ollama"))
    store = build_store(cfg)
    hit = correct = refused_ok = n_ans = n_unans = 0
    lat, fails = [], []
    for x in qs:
        t0 = time.time()
        r = rag.answer(store, x["q"], cfg["k"], cfg["min_score"], cfg["mode"], cfg.get("model"), cfg.get("kw_gate", False))
        lat.append(time.time() - t0)
        refused = any(p in r["answer"].lower() for p in REFUSAL_PHRASES)
        if x["expect"]:
            n_ans += 1
            h = has(" ".join(c["text"] for c in r["retrieved"]), x["expect"])
            c = (not refused) and has(r["answer"], x.get("answer_expect", []) + x["expect"])
            hit += h
            correct += c
            if not c:
                fails.append((x["q"], "retrieval miss" if not h else "wrong/refused", r["answer"][:90].replace("\n", " ")))
        else:
            n_unans += 1
            refused_ok += refused
            if not refused:
                fails.append((x["q"], "should refuse", r["answer"][:90].replace("\n", " ")))
    res = {"label": label, "chunks": len(store.items), "model": GROQ_MODEL if cfg.get("provider") == "groq" else cfg.get("model", "llama3.2:3b"), "provider": cfg.get("provider", "ollama"), "kw_gate": cfg.get("kw_gate", False), **cfg,
           "retrieval_hit": round(hit / n_ans, 3), "answer_correct": round(correct / n_ans, 3),
           "refusal_correct": round(refused_ok / n_unans, 3), "avg_latency_s": round(sum(lat) / len(lat), 2)}
    return res, fails


def main():
    labels = sys.argv[1:] or list(CONFIGS)
    qs = json.load(open("eval/questions.json"))
    for label in labels:
        res, fails = run(label, CONFIGS[label], qs)
        print(f"\n== {label} ==  hit={res['retrieval_hit']:.0%}  correct={res['answer_correct']:.0%}  "
              f"refusal={res['refusal_correct']:.0%}  latency={res['avg_latency_s']}s  chunks={res['chunks']}")
        for f in fails:
            print("  ✗", *f)
        with open("eval/results.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        json.dump(_cache, open(CACHE_PATH, "w"))


if __name__ == "__main__":
    main()
