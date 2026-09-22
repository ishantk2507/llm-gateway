"""Text → L2-normalized vector (DESIGN.md §5.2).

MiniLM runs in a thread executor: model.encode is CPU-bound BLAS work —
calling it inline would freeze every concurrent request on the event loop.
L2-normalized vectors are why plain inner-product search == cosine.

FakeEmbedder is ADR-0006's logic applied to embeddings: deterministic
hash-seeded vectors, same text → same vector, different text → near-orthogonal.
It powers every fast test; the REAL model is reserved for the
cache-correctness file, where similarity semantics are the thing under test.
"""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

DIMENSION = 384


class Embedder(Protocol):
    async def embed(self, text: str) -> np.ndarray: ...
    async def aclose(self) -> None: ...


class MiniLMEmbedder:
    def __init__(self, model_name: str, threads: int) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import — once, at startup

        self._executor = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="embedder")
        self._model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode, text)

    def _encode(self, text: str) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False)


class FakeEmbedder:
    """Deterministic, instant, no download. Different texts land near-orthogonal —
    so with this embedder the semantic layer can only 'hit' identical prompts,
    which is exactly what app-level tests should rely on."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self._dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)

    async def aclose(self) -> None:
        pass
