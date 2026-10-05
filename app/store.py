import json
import math
import os
import re
from collections import Counter

import numpy as np

from .config import STORE_PATH


STOPWORDS = set("""a an the of in on to for and or is are was were be been being what which who whom whose
how why when where does do did with from that this these those it its by as at about into than then he she
his her they their them has have had can could would should will not no yes if so there here more most
many much any some all each other such only own same very just also""".split())


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def content_terms(s: str) -> set[str]:
    return {t for t in tokenize(s) if t not in STOPWORDS}


class VectorStore:
    """Small in-memory store (vector + BM25 search) persisted to JSON.
    Swap for Qdrant/pgvector later."""

    def __init__(self, load: bool = True):
        self.items: list[dict] = []   # {doc, page, text, vec}
        self._index = None
        if load and os.path.exists(STORE_PATH):
            with open(STORE_PATH) as f:
                self.items = json.load(f)

    def add(self, doc, page, text, vec):
        self.items.append({"doc": doc, "page": page, "text": text, "vec": vec})
        self._index = None

    def save(self):
        os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
        with open(STORE_PATH, "w") as f:
            json.dump(self.items, f)

    def _build(self):
        m = np.array([i["vec"] for i in self.items])
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        toks = [Counter(tokenize(i["text"])) for i in self.items]
        df = Counter(t for c in toks for t in c)
        n = len(toks)
        idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        lens = np.array([sum(c.values()) for c in toks])
        self._index = (m, toks, idf, lens, lens.mean())

    def _bm25(self, question, k1=1.5, b=0.75):
        _, toks, idf, lens, avg = self._index
        q = set(tokenize(question))
        scores = np.zeros(len(toks))
        for i, c in enumerate(toks):
            for t in q:
                f = c.get(t, 0)
                if f:
                    scores[i] += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
        return scores

    def search(self, qvec, question, k, mode="vector"):
        """Return (hits, best_vector_score, keyword_match).
        hits is [(item, vector_score)]; keyword_match is (coverage, matched_terms) for the
        chunk that contains the most of the question's content words."""
        if not self.items:
            return [], 0.0, (0.0, 0)
        if self._index is None:
            self._build()
        q = np.array(qvec)
        vec = self._index[0] @ (q / (np.linalg.norm(q) + 1e-9))
        if mode == "hybrid":
            # reciprocal rank fusion of vector and BM25 rankings
            fused = np.zeros(len(vec))
            for s in (vec, self._bm25(question)):
                for rank, i in enumerate(np.argsort(-s)):
                    fused[i] += 1 / (60 + rank)
            order = np.argsort(-fused)[:k]
        else:
            order = np.argsort(-vec)[:k]
        terms = content_terms(question)
        matched = max((len(terms & c.keys()) for c in self._index[1]), default=0)
        kw = (matched / len(terms) if terms else 0.0, matched)
        return [(self.items[i], float(vec[i])) for i in order], float(vec.max()), kw

    def remove(self, doc) -> int:
        before = len(self.items)
        self.items = [i for i in self.items if i["doc"] != doc]
        self._index = None
        return before - len(self.items)

    def documents(self):
        info = {}
        for i in self.items:
            d = info.setdefault(i["doc"], {"name": i["doc"], "chunks": 0, "pages": set()})
            d["chunks"] += 1
            d["pages"].add(i["page"])
        return [{"name": d["name"], "chunks": d["chunks"], "pages": len(d["pages"])}
                for d in sorted(info.values(), key=lambda d: d["name"].lower())]
