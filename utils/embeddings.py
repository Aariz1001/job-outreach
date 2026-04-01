"""Gemini embedding wrapper using the REST API directly."""
import json
import requests
import numpy as np
from utils.config import GEMINI_API_KEY, EMBEDDING_MODEL


def embed_text(text: str) -> list[float]:
    """Return a 3072-dim embedding for `text` using Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:embedContent"
    payload = {
        "model": EMBEDDING_MODEL,
        "content": {"parts": [{"text": text}]},
    }
    resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
