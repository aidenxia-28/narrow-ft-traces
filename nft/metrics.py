import re
import torch
from .config import EMBED_MODEL, get_device


class Embedder:
    """Qwen3-Embedding-0.6B via sentence-transformers, the similarity model used by Minder et al."""

    def __init__(self, device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBED_MODEL, device=device or get_device())

    def encode(self, texts, batch_size=16):
        e = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
        e = e.float().cpu()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return e


def mean_pairwise_cos(A, B):
    return float((A @ B.T).mean())


def keyword_stats(texts, keywords):
    pats = [re.compile(r"\b" + re.escape(k.lower()) + r"\b") for k in keywords]
    frac_any, per100 = 0.0, 0.0
    for t in texts:
        tl = t.lower()
        hits = sum(1 for p in pats if p.search(tl))
        words = max(1, len(tl.split()))
        frac_any += hits > 0
        per100 += 100.0 * sum(len(p.findall(tl)) for p in pats) / words
    n = max(1, len(texts))
    return {"frac_with_keyword": frac_any / n, "keywords_per_100_words": per100 / n}
