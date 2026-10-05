import json
import os
import time

import httpx

from .config import EMBED_MODEL, GROQ_MODEL, GROQ_URL, LLM_MODEL, LLM_PROVIDER, OLLAMA_URL

PROVIDER = LLM_PROVIDER   # the eval switches this per configuration


def set_provider(name: str):
    global PROVIDER
    PROVIDER = name


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(f"{OLLAMA_URL}/api/embed",
                   json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


# ---------- Groq (OpenAI-compatible API) ----------
def _groq_headers() -> dict:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Create a free key at console.groq.com "
                           "and export it before starting the server.")
    return {"Authorization": f"Bearer {key}"}


def _groq_body(prompt: str, stream: bool) -> dict:
    return {"model": GROQ_MODEL, "temperature": 0, "stream": stream,
            "messages": [{"role": "user", "content": prompt}]}


def _groq_post(client: httpx.Client, body: dict, stream: bool):
    """Send with a few retries when the free tier rate-limits us (HTTP 429)."""
    for attempt in range(4):
        req = client.build_request("POST", f"{GROQ_URL}/chat/completions",
                                   headers=_groq_headers(), json=body)
        r = client.send(req, stream=stream)
        if r.status_code != 429 or attempt == 3:
            return r
        wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
        r.close()
        time.sleep(min(wait, 20))


def _check_groq(r: httpx.Response):
    """Raise with Groq's own error message instead of a bare status code."""
    if r.status_code >= 400:
        r.read()
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"Groq error {r.status_code}: {msg}")


# ---------- public API ----------
def generate(prompt: str, model: str | None = None) -> str:
    if PROVIDER == "groq":
        with httpx.Client(timeout=120) as c:
            r = _groq_post(c, _groq_body(prompt, False), False)
            _check_groq(r)
            return r.json()["choices"][0]["message"]["content"].strip()
    r = httpx.post(f"{OLLAMA_URL}/api/generate",
                   json={"model": model or LLM_MODEL, "prompt": prompt, "stream": False,
                         "options": {"temperature": 0}}, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


def generate_stream(prompt: str, model: str | None = None):
    """Yield the answer token by token."""
    if PROVIDER == "groq":
        with httpx.Client(timeout=None) as c:
            r = _groq_post(c, _groq_body(prompt, True), True)
            try:
                _check_groq(r)
                for line in r.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
            finally:
                r.close()
        return
    with httpx.stream("POST", f"{OLLAMA_URL}/api/generate",
                      json={"model": model or LLM_MODEL, "prompt": prompt, "stream": True,
                            "options": {"temperature": 0}}, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                token = json.loads(line).get("response")
                if token:
                    yield token
