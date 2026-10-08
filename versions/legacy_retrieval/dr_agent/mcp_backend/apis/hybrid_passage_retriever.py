# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""CPU-friendly dense retrieval and cross-encoder reranking for V27."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from diskcache import Cache


HYBRID_RETRIEVER_VERSION = "medgap_hybrid_passage_v27"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class TransformerHybridBackend:
    """Lazy, cached Transformers backend; CPU is the safe production default."""

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        embedding_revision: str | None = None,
        reranker_revision: str | None = None,
        query_prefix: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model or os.getenv(
            "MEDGAP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.reranker_model_name = reranker_model or os.getenv(
            "MEDGAP_RERANKER_MODEL", DEFAULT_RERANKER_MODEL
        )
        self.embedding_revision = embedding_revision or os.getenv(
            "MEDGAP_EMBEDDING_REVISION", "main"
        )
        self.reranker_revision = reranker_revision or os.getenv(
            "MEDGAP_RERANKER_REVISION", "main"
        )
        self.device = device or os.getenv("MEDGAP_RETRIEVAL_DEVICE", "cpu")
        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else os.getenv("MEDGAP_EMBEDDING_QUERY_PREFIX", DEFAULT_QUERY_PREFIX)
        )
        configured_cache = cache_dir or os.getenv("MEDGAP_HYBRID_RETRIEVAL_CACHE_DIR")
        if configured_cache is None:
            configured_cache = Path(os.getenv("MCP_CACHE_DIR", ".cache")) / "hybrid_v27"
        self.cache = Cache(str(Path(configured_cache).expanduser()))
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        # Keep cache lookup + fill atomic inside one process. Without this,
        # concurrent rollout threads can all miss the same document and repeat
        # the expensive CPU encoding/reranking work while waiting on inference.
        self._cache_fill_lock = threading.RLock()
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._reranker_tokenizer = None
        self._reranker_model = None
        self._embedding_commit = None
        self._reranker_commit = None

    def _resolve_commit(self, *, embedding: bool) -> str | None:
        attribute = "_embedding_commit" if embedding else "_reranker_commit"
        resolved = getattr(self, attribute)
        if resolved:
            return resolved
        model = self._embedding_model if embedding else self._reranker_model
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
        if not resolved:
            # A fully cached inference may never load the weights in this
            # process. Resolve the tiny config so observability still records
            # the immutable Hub commit rather than only a mutable `main` tag.
            from transformers import AutoConfig

            local_only = os.getenv("MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY", "false").lower() == "true"
            name = self.embedding_model_name if embedding else self.reranker_model_name
            revision = self.embedding_revision if embedding else self.reranker_revision
            config = AutoConfig.from_pretrained(
                name,
                revision=revision,
                local_files_only=local_only,
            )
            resolved = getattr(config, "_commit_hash", None)
        setattr(self, attribute, resolved)
        return resolved

    @property
    def identity(self) -> dict[str, Any]:
        value = {
            "pipeline_version": HYBRID_RETRIEVER_VERSION,
            "embedding_model": self.embedding_model_name,
            "embedding_revision": self.embedding_revision,
            "embedding_query_prefix": self.query_prefix,
            "embedding_commit": self._resolve_commit(embedding=True),
            "reranker_model": self.reranker_model_name,
            "reranker_revision": self.reranker_revision,
            "reranker_commit": self._resolve_commit(embedding=False),
            "device": self.device,
        }
        value["identity_sha256"] = _sha256_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        )
        return value

    def _load_embedding(self):
        with self._load_lock:
            if self._embedding_model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer

                local_only = os.getenv("MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY", "false").lower() == "true"
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    self.embedding_model_name,
                    revision=self.embedding_revision,
                    local_files_only=local_only,
                )
                self._embedding_model = AutoModel.from_pretrained(
                    self.embedding_model_name,
                    revision=self.embedding_revision,
                    local_files_only=local_only,
                ).to(self.device)
                self._embedding_model.eval()
                self._embedding_commit = getattr(self._embedding_model.config, "_commit_hash", None)
                if self.device == "cpu":
                    torch.set_num_threads(max(1, int(os.getenv("MEDGAP_RETRIEVAL_TORCH_THREADS", "4"))))
        return self._embedding_tokenizer, self._embedding_model

    def _load_reranker(self):
        with self._load_lock:
            if self._reranker_model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                local_only = os.getenv("MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY", "false").lower() == "true"
                self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                    self.reranker_model_name,
                    revision=self.reranker_revision,
                    local_files_only=local_only,
                )
                self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    self.reranker_model_name,
                    revision=self.reranker_revision,
                    local_files_only=local_only,
                ).to(self.device)
                self._reranker_model.eval()
                self._reranker_commit = getattr(self._reranker_model.config, "_commit_hash", None)
        return self._reranker_tokenizer, self._reranker_model

    def _encode_uncached(self, texts: list[str], *, query: bool) -> np.ndarray:
        import torch

        tokenizer, model = self._load_embedding()
        prefix = self.query_prefix if query else ""
        prepared = [prefix + text if query else text for text in texts]
        batch_size = max(1, int(os.getenv("MEDGAP_RETRIEVAL_EMBED_BATCH_SIZE", "16")))
        outputs = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                encoded = tokenizer(
                    prepared[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**encoded).last_hidden_state[:, 0]
                outputs.append(hidden.detach().float().cpu().numpy())
        return _normalized_rows(np.concatenate(outputs, axis=0))

    def encode_texts(self, texts: list[str], *, query: bool) -> tuple[np.ndarray, dict[str, int]]:
        with self._cache_fill_lock:
            namespace = "query" if query else "passage"
            query_prefix = self.query_prefix if query else ""
            rows: list[np.ndarray | None] = [None] * len(texts)
            missing_indices = []
            hits = 0
            for index, text in enumerate(texts):
                key = (
                    f"dense:{self.embedding_model_name}:{self.embedding_revision}:"
                    f"{namespace}:{_sha256_text(query_prefix + text)}"
                )
                cached = self.cache.get(key)
                if cached is None:
                    missing_indices.append(index)
                else:
                    rows[index] = np.asarray(cached, dtype=np.float32)
                    hits += 1
            if missing_indices:
                encoded = self._encode_uncached(
                    [texts[index] for index in missing_indices], query=query
                )
                for position, index in enumerate(missing_indices):
                    vector = encoded[position]
                    rows[index] = vector
                    key = (
                        f"dense:{self.embedding_model_name}:{self.embedding_revision}:"
                        f"{namespace}:{_sha256_text(query_prefix + texts[index])}"
                    )
                    self.cache.set(key, vector, expire=86400 * 30)
            if any(row is None for row in rows):
                raise RuntimeError("dense embedding cache fill lost one or more input rows")
            return np.stack(rows), {
                "hits": hits,
                "misses": len(missing_indices),
            }

    def dense_scores(self, query: str, passages: list[str]) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        query_vectors, query_cache = self.encode_texts([query], query=True)
        passage_vectors, passage_cache = self.encode_texts(passages, query=False)
        scores = passage_vectors @ query_vectors[0]
        return [float(value) for value in scores], {
            "query_cache": query_cache,
            "passage_cache": passage_cache,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def rerank_scores(self, query: str, passages: list[str]) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        payload_hash = _sha256_text(json.dumps([query, passages], ensure_ascii=False, separators=(",", ":")))
        key = f"rerank:{self.reranker_model_name}:{self.reranker_revision}:{payload_hash}"
        with self._cache_fill_lock:
            cached = self.cache.get(key)
            if cached is not None:
                return list(map(float, cached)), {"cache_hit": True, "elapsed_ms": 0.0}

            import torch

            tokenizer, model = self._load_reranker()
            batch_size = max(1, int(os.getenv("MEDGAP_RETRIEVAL_RERANK_BATCH_SIZE", "16")))
            scores = []
            with self._inference_lock, torch.inference_mode():
                for start in range(0, len(passages), batch_size):
                    batch = passages[start : start + batch_size]
                    encoded = tokenizer(
                        [query] * len(batch),
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = model(**encoded).logits.detach().float().cpu().numpy()
                    scores.extend(logits.reshape(-1).tolist())
            self.cache.set(key, scores, expire=86400 * 30)
            return [float(value) for value in scores], {
                "cache_hit": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }


_BACKEND: TransformerHybridBackend | None = None
_BACKEND_LOCK = threading.RLock()


def get_hybrid_backend() -> TransformerHybridBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = TransformerHybridBackend()
        return _BACKEND


def _rank_map(chunks: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    ranked = sorted(chunks, key=lambda item: (-float(item.get(key) or 0.0), item["section_index"], item["chunk_index"]))
    return {item["chunk_id"]: rank for rank, item in enumerate(ranked, 1)}


def hybrid_rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    backend: Any | None = None,
    bm25_top_n: int = 8,
    dense_top_n: int = 8,
    fusion_max: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return cross-encoder-ranked fusion candidates plus complete diagnostics."""
    if not chunks:
        return [], {"identity": {}, "fusion_candidates": [], "timing": {}}
    backend = backend or get_hybrid_backend()
    passage_texts = [f"{chunk['heading']}\n{chunk['text']}" for chunk in chunks]
    dense_scores, dense_diagnostics = backend.dense_scores(query, passage_texts)
    for chunk, dense_score in zip(chunks, dense_scores):
        chunk["dense_score"] = round(float(dense_score), 8)

    bm25_ranks = _rank_map(chunks, "bm25_score")
    dense_ranks = _rank_map(chunks, "dense_score")
    for chunk in chunks:
        chunk["bm25_rank"] = bm25_ranks[chunk["chunk_id"]]
        chunk["dense_rank"] = dense_ranks[chunk["chunk_id"]]
        chunk["fusion_candidate"] = False
        chunk["fusion_reasons"] = []

    selected_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for reason, rank_key, limit in (
        ("bm25", "bm25_rank", bm25_top_n),
        ("dense", "dense_rank", dense_top_n),
    ):
        for chunk in sorted(chunks, key=lambda item: item[rank_key])[:limit]:
            chunk["fusion_reasons"].append(reason)
            if chunk["chunk_id"] not in selected_ids:
                selected_ids.add(chunk["chunk_id"])
                candidates.append(chunk)

    candidates.sort(
        key=lambda item: (
            1.0 / (60 + item["bm25_rank"]) + 1.0 / (60 + item["dense_rank"]),
            -item["section_index"],
            -item["chunk_index"],
        ),
        reverse=True,
    )
    candidates = candidates[:fusion_max]
    for chunk in candidates:
        chunk["fusion_candidate"] = True

    rerank_scores, rerank_diagnostics = backend.rerank_scores(
        query,
        [f"{chunk['heading']}\n{chunk['text']}" for chunk in candidates],
    )
    for chunk, score in zip(candidates, rerank_scores):
        chunk["cross_encoder_score"] = round(float(score), 8)
        chunk["score"] = round(float(score), 6)
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["cross_encoder_score"]),
            item["bm25_rank"],
            item["dense_rank"],
            item["section_index"],
            item["chunk_index"],
        ),
    )
    for rank, chunk in enumerate(ranked, 1):
        chunk["rank"] = rank

    identity = getattr(backend, "identity", {})
    return ranked, {
        "identity": dict(identity) if isinstance(identity, dict) else {},
        "bm25_top_n": bm25_top_n,
        "dense_top_n": dense_top_n,
        "fusion_max": fusion_max,
        "fusion_candidates": [chunk["chunk_id"] for chunk in candidates],
        "dense": dense_diagnostics,
        "cross_encoder": rerank_diagnostics,
    }
