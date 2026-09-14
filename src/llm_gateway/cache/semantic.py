"""Layer 2 — semantic paraphrase matching (ADR-0005).

FAISS IndexIDMap2(IndexFlatIP) holds L2-normalized vectors (inner product ==
cosine); SQLite holds payloads keyed by faiss_id so entries can expire and be
deleted — the IDMap exists precisely because a bare FlatIP cannot remove.

Lookup walks top-k, skipping wrong-family and lazily-expired rows: the first
VALID candidate decides — ≥ threshold is a hit, the [floor, threshold) band is
a near-miss (the service logs it as calibration data), below the floor is a
miss and nothing lower can qualify (scores sort descending).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from llm_gateway.cache.embedder import Embedder
from llm_gateway.config import CacheSettings
from llm_gateway.schemas.openai_api import ChatCompletionResponse

TOP_K = 5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    faiss_id      INTEGER PRIMARY KEY,
    prompt_hash   TEXT NOT NULL,
    model_family  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL
)
"""

SemanticStatus = Literal["hit", "near_miss", "miss"]


@dataclass
class SemanticResult:
    status: SemanticStatus
    similarity: float | None = None
    response: ChatCompletionResponse | None = None


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        *,
        index_path: Path,
        db_path: Path,
        settings: CacheSettings,
        dimension: int = 384,
    ) -> None:
        self._embedder = embedder
        self._settings = settings
        self._index_path = Path(index_path)
        self._lock = asyncio.Lock()  # serializes index/db mutations, not searches

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(_SCHEMA)
        self._db.commit()

        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._next_id = self._max_id() + 1

    # ── lookup ─────────────────────────────────────────────────────────────

    async def lookup(self, prompt: str, family: str) -> SemanticResult:
        if self._index.ntotal == 0:
            return SemanticResult("miss")
        vector = await self._embedder.embed(prompt)
        loop = asyncio.get_running_loop()
        scores, ids = await loop.run_in_executor(None, self._search, vector)
        now = time.time()

        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            if faiss_id == -1:
                continue
            row = await asyncio.to_thread(self._fetch_row, int(faiss_id))
            if row is None:
                continue  # index/db drift — skip, don't crash
            _, row_family, response_json, _, expires_at = row
            if row_family != family:
                continue
            if expires_at <= now:
                await self._remove(int(faiss_id))  # lazy expiry
                continue
            if score >= self._settings.similarity_threshold:
                return SemanticResult(
                    "hit", float(score), ChatCompletionResponse.model_validate_json(response_json)
                )
            if score >= self._settings.near_miss_floor:
                return SemanticResult("near_miss", float(score))
            break  # below the floor — nothing lower can qualify
        return SemanticResult("miss")

    # ── store ──────────────────────────────────────────────────────────────

    async def store(
        self, prompt: str, family: str, prompt_hash: str, response: ChatCompletionResponse
    ) -> None:
        if self._index.ntotal >= self._settings.max_semantic_entries:
            await self._evict_oldest()  # ADR-0005: memory grows linearly; bounded here
        vector = await self._embedder.embed(prompt)
        async with self._lock:
            faiss_id = self._next_id
            self._next_id += 1
            now = time.time()
            await asyncio.to_thread(
                self._insert_row,
                faiss_id,
                prompt_hash,
                family,
                response.model_dump_json(),
                now,
                now + self._settings.semantic_ttl_seconds,
            )
            # add_with_ids on a flat index is µs–ms at v1 scale — inline under the lock
            self._index.add_with_ids(
                np.reshape(vector, (1, -1)), np.asarray([faiss_id], dtype=np.int64)
            )
            await asyncio.to_thread(faiss.write_index, self._index, str(self._index_path))

    # ── invalidation ──────────────────────────────────────────────────────

    async def remove_by_prompt_hash(self, target: str) -> int:
        ids = await asyncio.to_thread(
            self._ids_matching,
            "SELECT faiss_id FROM cache_entries WHERE prompt_hash = ?",
            (target,),
        )
        return await self._remove_ids_list(ids)

    async def remove_by_family(self, family: str) -> int:
        ids = await asyncio.to_thread(
            self._ids_matching,
            "SELECT faiss_id FROM cache_entries WHERE model_family = ?",
            (family,),
        )
        return await self._remove_ids_list(ids)

    async def flush_all(self) -> int:
        ids = await asyncio.to_thread(self._ids_matching, "SELECT faiss_id FROM cache_entries", ())
        return await self._remove_ids_list(ids)

    # ── internals ──────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        async with self._lock:
            faiss.write_index(self._index, str(self._index_path))
        await asyncio.to_thread(self._db.close)

    async def _remove(self, faiss_id: int) -> None:
        await self._remove_ids_list([faiss_id])

    async def _remove_ids_list(self, ids: list[int]) -> int:
        if not ids:
            return 0
        async with self._lock:
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            await asyncio.to_thread(self._delete_rows, ids)
            await asyncio.to_thread(faiss.write_index, self._index, str(self._index_path))
        return len(ids)

    async def _evict_oldest(self) -> None:
        row = await asyncio.to_thread(
            lambda: self._db.execute(
                "SELECT faiss_id FROM cache_entries ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        )
        if row:
            await self._remove(row[0])

    def _search(self, vector: np.ndarray):
        k = min(TOP_K, self._index.ntotal)
        return self._index.search(np.reshape(vector, (1, -1)).astype(np.float32), k)

    def _max_id(self) -> int:
        row = self._db.execute("SELECT COALESCE(MAX(faiss_id), -1) FROM cache_entries").fetchone()
        return int(row[0])

    def _fetch_row(self, faiss_id: int):
        return self._db.execute(
            "SELECT prompt_hash, model_family, response_json, created_at, expires_at "
            "FROM cache_entries WHERE faiss_id = ?",
            (faiss_id,),
        ).fetchone()

    def _insert_row(self, faiss_id, prompt_hash, family, response_json, created, expires):
        self._db.execute(
            "INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?)",
            (faiss_id, prompt_hash, family, response_json, created, expires),
        )
        self._db.commit()

    def _delete_rows(self, ids: list[int]):
        self._db.executemany("DELETE FROM cache_entries WHERE faiss_id = ?", [(i,) for i in ids])
        self._db.commit()

    def _ids_matching(self, query: str, params: tuple) -> list[int]:
        return [int(r[0]) for r in self._db.execute(query, params).fetchall()]
