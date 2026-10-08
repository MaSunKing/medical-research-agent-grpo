# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Biomedical V28 passage retrieval with optional local evidence reading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from diskcache import Cache

from dr_agent.anchored_query import anchored_relevance_query
from .evidence_reader_v28 import get_evidence_reader


V28_RETRIEVER_VERSION = "medgap_medcpt_reader_v28"
DEFAULT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
DEFAULT_CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"

# Chunk headings are parser hints, not trustworthy evidence labels.  In
# particular, guideline pages sometimes place real recommendations under a
# misleading heading such as ``Keywords``.  Heading matches therefore affect
# ranking only; they never exclude a chunk by themselves.
_NON_EVIDENCE_HEADING_RE = re.compile(
    r"^(?:references?|bibliography|keywords?|author(?:s| information)?|"
    r"acknowledg(?:e)?ments?|funding|conflicts? of interest|disclosures?|"
    r"contact us|sign in|log in|related articles?|topic nomination form)\b",
    flags=re.IGNORECASE,
)
_LOW_SIGNAL_HEADING_RE = re.compile(
    r"(?:document scope|implementation resources?|supplemental resources?|"
    r"resource library|about this (?:page|guideline)|table of contents)",
    flags=re.IGNORECASE,
)
_BOILERPLATE_TEXT_RE = re.compile(
    r"(?:skip to (?:main )?content|accept all cookies|cookie preferences|"
    r"sign in to access|subscribe to continue|javascript is required|"
    r"privacy policy\s+terms of use|home\s+about us\s+contact us)",
    flags=re.IGNORECASE,
)
_EVIDENCE_SIGNAL_RE = re.compile(
    r"(?:\b(?:recommend(?:s|ed|ation)?|should|offer|avoid|associated|reduced|"
    r"increased|participants?|patients?|trial|cohort|outcome|adverse|safety|"
    r"mortality|risk ratio|hazard ratio|confidence interval)\b|"
    r"\b\d+(?:\.\d+)?\s*%|\b(?:RR|HR|OR)\s*[=:]?)",
    flags=re.IGNORECASE,
)


def _content_quality(row: dict[str, Any]) -> dict[str, Any]:
    heading = re.sub(r"\s+", " ", str(row.get("heading") or "")).strip()
    text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
    evidence_signal = bool(_EVIDENCE_SIGNAL_RE.search(text))
    heading_looks_non_evidence = bool(_NON_EVIDENCE_HEADING_RE.search(heading))
    heading_looks_low_signal = bool(_LOW_SIGNAL_HEADING_RE.search(heading))
    clear_boilerplate = bool(_BOILERPLATE_TEXT_RE.search(text)) and not evidence_signal

    # Hard exclusion is reserved for content that is intrinsically unusable.
    # A suspicious heading alone is never sufficient because headings produced
    # by HTML-to-markdown parsers are not reliable section boundaries.
    if clear_boilerplate:
        return {
            "quality_class": "excluded_non_evidence_section",
            "quality_eligible": False,
            "quality_penalty": 100.0,
            "quality_evidence_signal": evidence_signal,
            "quality_heading_mismatch": False,
            "quality_reason": "clear_boilerplate_without_evidence",
        }

    # Evidence-bearing prose overrides a misleading heading completely.  This
    # is the important recovery path for real guideline text parsed beneath a
    # ``Keywords`` or similar heading.
    if heading_looks_non_evidence and evidence_signal:
        return {
            "quality_class": "evidence_candidate",
            "quality_eligible": True,
            "quality_penalty": 0.0,
            "quality_evidence_signal": True,
            "quality_heading_mismatch": True,
            "quality_reason": "heading_mismatch_overridden_by_evidence",
        }

    # Ambiguous headings and short contextual chunks remain retrievable, but
    # rank below ordinary evidence candidates.  They can still be selected by
    # the Reader or used as a Top-3 fallback when stronger chunks are scarce.
    if heading_looks_non_evidence or heading_looks_low_signal or (
        len(text) < 320 and not evidence_signal
    ):
        if heading_looks_non_evidence:
            reason = "non_evidence_heading_soft_penalty"
        elif heading_looks_low_signal:
            reason = "low_signal_heading_soft_penalty"
        else:
            reason = "short_context_soft_penalty"
        return {
            "quality_class": "low_signal_context",
            "quality_eligible": True,
            "quality_penalty": 2.0,
            "quality_evidence_signal": evidence_signal,
            "quality_heading_mismatch": heading_looks_non_evidence,
            "quality_reason": reason,
        }
    return {
        "quality_class": "evidence_candidate",
        "quality_eligible": True,
        "quality_penalty": 0.0,
        "quality_evidence_signal": evidence_signal,
        "quality_heading_mismatch": False,
        "quality_reason": (
            "evidence_signal" if evidence_signal else "general_evidence_context"
        ),
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _normalized(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def _rank_map(chunks: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    ranked = sorted(
        chunks,
        key=lambda item: (
            -float(item.get(key) or 0.0), item["section_index"], item["chunk_index"]
        ),
    )
    return {item["chunk_id"]: rank for rank, item in enumerate(ranked, 1)}


class MedCPTRetrievalBackend:
    def __init__(self, *, device: str | None = None, cache_dir: str | Path | None = None) -> None:
        self.device = str(device or os.getenv("MEDGAP_V28_DEVICE", "cpu"))
        self.query_model_name = os.getenv("MEDGAP_MEDCPT_QUERY_MODEL", DEFAULT_QUERY_MODEL)
        self.article_model_name = os.getenv("MEDGAP_MEDCPT_ARTICLE_MODEL", DEFAULT_ARTICLE_MODEL)
        self.cross_encoder_name = os.getenv("MEDGAP_MEDCPT_CROSS_ENCODER", DEFAULT_CROSS_ENCODER)
        configured = cache_dir or os.getenv("MEDGAP_V28_RETRIEVAL_CACHE_DIR")
        if configured is None:
            configured = Path(os.getenv("MCP_CACHE_DIR", ".cache")) / "retrieval_v28"
        self.cache = Cache(str(Path(configured).expanduser()))
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._query_tokenizer = self._query_model = None
        self._article_tokenizer = self._article_model = None
        self._cross_tokenizer = self._cross_model = None

    @property
    def identity(self) -> dict[str, Any]:
        value = {
            "pipeline_version": V28_RETRIEVER_VERSION,
            "query_model": self.query_model_name,
            "article_model": self.article_model_name,
            "cross_encoder": self.cross_encoder_name,
            "device": self.device,
        }
        value["identity_sha256"] = _sha256(json.dumps(value, sort_keys=True))
        return value

    def _load_encoder(self, query: bool):
        with self._load_lock:
            tokenizer_attr = "_query_tokenizer" if query else "_article_tokenizer"
            model_attr = "_query_model" if query else "_article_model"
            tokenizer = getattr(self, tokenizer_attr)
            model = getattr(self, model_attr)
            if model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer

                name = self.query_model_name if query else self.article_model_name
                local_only = os.getenv("MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY", "false").lower() == "true"
                tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_only)
                model = AutoModel.from_pretrained(name, local_files_only=local_only).to(self.device)
                model.eval()
                if self.device == "cpu":
                    torch.set_num_threads(max(1, int(os.getenv("MEDGAP_RETRIEVAL_TORCH_THREADS", "4"))))
                setattr(self, tokenizer_attr, tokenizer)
                setattr(self, model_attr, model)
        return tokenizer, model

    def _load_cross(self):
        with self._load_lock:
            if self._cross_model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                local_only = os.getenv("MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY", "false").lower() == "true"
                self._cross_tokenizer = AutoTokenizer.from_pretrained(
                    self.cross_encoder_name, local_files_only=local_only
                )
                self._cross_model = AutoModelForSequenceClassification.from_pretrained(
                    self.cross_encoder_name, local_files_only=local_only
                ).to(self.device)
                self._cross_model.eval()
        return self._cross_tokenizer, self._cross_model

    def _encode_uncached(self, texts: list[str], *, query: bool) -> np.ndarray:
        import torch

        tokenizer, model = self._load_encoder(query)
        batch_size = max(1, int(os.getenv("MEDGAP_RETRIEVAL_EMBED_BATCH_SIZE", "16")))
        rows = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**encoded).last_hidden_state[:, 0]
                rows.append(hidden.detach().float().cpu().numpy())
        return _normalized(np.concatenate(rows, axis=0))

    def encode(self, texts: list[str], *, query: bool) -> tuple[np.ndarray, dict[str, int]]:
        model_name = self.query_model_name if query else self.article_model_name
        namespace = "query" if query else "article"
        rows: list[np.ndarray | None] = [None] * len(texts)
        missing = []
        hits = 0
        for index, text in enumerate(texts):
            key = f"medcpt:{model_name}:{namespace}:{_sha256(text)}"
            cached = self.cache.get(key)
            if cached is None:
                missing.append(index)
            else:
                rows[index] = np.asarray(cached, dtype=np.float32)
                hits += 1
        if missing:
            encoded = self._encode_uncached([texts[index] for index in missing], query=query)
            for position, index in enumerate(missing):
                rows[index] = encoded[position]
                self.cache.set(
                    f"medcpt:{model_name}:{namespace}:{_sha256(texts[index])}",
                    encoded[position],
                    expire=86400 * 30,
                )
        if any(row is None for row in rows):
            raise RuntimeError("MedCPT embedding cache fill lost rows")
        return np.stack(rows), {"hits": hits, "misses": len(missing)}

    def dense_scores(self, query: str, passages: list[str]) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        q, qcache = self.encode([query], query=True)
        p, pcache = self.encode(passages, query=False)
        return (p @ q[0]).tolist(), {
            "query_cache": qcache,
            "passage_cache": pcache,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def cross_scores(self, query: str, passages: list[str]) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        key = f"medcpt-cross:{self.cross_encoder_name}:{_sha256(json.dumps([query, passages]))}"
        cached = self.cache.get(key)
        if cached is not None:
            return list(map(float, cached)), {"cache_hit": True, "elapsed_ms": 0.0}
        import torch

        tokenizer, model = self._load_cross()
        scores: list[float] = []
        batch_size = max(1, int(os.getenv("MEDGAP_RETRIEVAL_RERANK_BATCH_SIZE", "16")))
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(passages), batch_size):
                batch = passages[start : start + batch_size]
                encoded = tokenizer(
                    [query] * len(batch), batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                ).to(self.device)
                logits = model(**encoded).logits.detach().float().cpu().numpy()
                scores.extend(logits.reshape(-1).tolist())
        self.cache.set(key, scores, expire=86400 * 30)
        return scores, {
            "cache_hit": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }


_BACKEND: MedCPTRetrievalBackend | None = None
_BACKEND_LOCK = threading.RLock()


def get_v28_backend() -> MedCPTRetrievalBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = MedCPTRetrievalBackend()
        return _BACKEND


def v28_rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    question: str = "",
    top_k: int = 3,
    backend: Any | None = None,
    reader: Any | None = None,
    bm25_top_n: int = 10,
    dense_top_n: int = 10,
    fusion_max: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        return [], {"identity": {}, "fusion_candidates": [], "reader": {}}
    backend = backend or get_v28_backend()
    retrieval_query = anchored_relevance_query(question, query)
    for row in chunks:
        row.update(_content_quality(row))
    passages = [f"{row['heading']}\n{row['text']}" for row in chunks]
    dense_scores, dense_diag = backend.dense_scores(retrieval_query, passages)
    for row, score in zip(chunks, dense_scores):
        row["medcpt_score"] = round(float(score), 8)
    bm25_ranks = _rank_map(chunks, "bm25_score")
    dense_ranks = _rank_map(chunks, "medcpt_score")
    for row in chunks:
        row["bm25_rank"] = bm25_ranks[row["chunk_id"]]
        row["medcpt_rank"] = dense_ranks[row["chunk_id"]]
        row["fusion_candidate"] = False
        row["fusion_reasons"] = []
    eligible_chunks = [row for row in chunks if row["quality_eligible"]]
    if not eligible_chunks:
        identity = getattr(backend, "identity", {})
        return [], {
            "identity": dict(identity) if isinstance(identity, dict) else {},
            "bm25_top_n": bm25_top_n,
            "dense_top_n": dense_top_n,
            "fusion_max": fusion_max,
            "fusion_candidates": [],
            "dense": dense_diag,
            "cross_encoder": {"skipped": "no_quality_eligible_chunks"},
            "reader": {"skipped": "no_quality_eligible_chunks"},
            "quality_gate": {
                "excluded_chunks": len(chunks),
                "soft_penalized_chunks": 0,
                "eligible_chunks": 0,
            },
        }
    selected: dict[str, dict[str, Any]] = {}
    for reason, rank_key, limit in (
        ("bm25", "bm25_rank", bm25_top_n), ("medcpt", "medcpt_rank", dense_top_n)
    ):
        for row in sorted(eligible_chunks, key=lambda item: item[rank_key])[:limit]:
            row["fusion_reasons"].append(reason)
            selected[row["chunk_id"]] = row
    candidates = sorted(
        selected.values(),
        key=lambda row: (
            -(1 / (60 + row["bm25_rank"]) + 1 / (60 + row["medcpt_rank"])),
            row["section_index"], row["chunk_index"],
        ),
    )[:fusion_max]
    for row in candidates:
        row["fusion_candidate"] = True
    cross_scores, cross_diag = backend.cross_scores(
        retrieval_query, [f"{row['heading']}\n{row['text']}" for row in candidates]
    )
    for row, score in zip(candidates, cross_scores):
        row["medcpt_cross_score"] = round(float(score), 8)
        row["score"] = round(
            float(score) - float(row.get("quality_penalty") or 0.0), 6
        )
    candidates.sort(
        key=lambda row: (
            -row["score"], row["bm25_rank"], row["medcpt_rank"]
        )
    )
    requested_top_k = min(max(1, int(top_k)), len(candidates))
    # Selection policy exposed to the Agent remains a bounded Top-K, but its
    # construction is deliberately asymmetric:
    #   1. preserve the best quality-eligible MedCPT cross-encoder passage;
    #   2. ask the frozen Reader for one complementary passage;
    #   3. fill the final slot only from the cross-encoder relevance window,
    #      preferring a new section inside that small window.
    # This prevents the Reader from discarding the strongest retrieval result
    # while avoiding three near-duplicate adjacent passages.
    anchor = candidates[0]
    chosen = [anchor]
    reader_pool_limit = max(
        requested_top_k,
        int(os.getenv("MEDGAP_EVIDENCE_READER_CANDIDATES", "6")),
    )
    reader_pool = candidates[1 : 1 + reader_pool_limit]
    reader = reader or get_evidence_reader()
    reader_selected: list[dict[str, Any]] = []
    if requested_top_k > 1 and reader_pool:
        anchor_context = (
            f"{question or query}\n\n"
            "An upstream biomedical cross-encoder already selected this anchor passage. "
            "Choose one different passage that adds complementary evidence rather than "
            "repeating it:\n"
            f"[{anchor.get('heading')}] {str(anchor.get('text') or '')[:1200]}"
        )
        try:
            indices, reader_diag = reader.select(
                question=anchor_context,
                focused_query=query,
                chunks=reader_pool,
                top_k=1,
            )
        except Exception as exc:
            indices = [0]
            reader_diag = {
                "version": "medgap_evidence_reader_v28",
                "enabled": bool(getattr(reader, "enabled", False)),
                "fallback": "upstream_rank",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:320],
            }
        reader_selected = [
            reader_pool[index]
            for index in indices[:1]
            if isinstance(index, int) and 0 <= index < len(reader_pool)
        ]
        if not reader_selected:
            reader_selected = [reader_pool[0]]
        chosen.extend(reader_selected)
    else:
        reader_diag = {
            "version": "medgap_evidence_reader_v28",
            "enabled": bool(getattr(reader, "enabled", False)),
            "skipped": "no_complement_slot_or_candidate",
            "selected_indices": [],
        }

    chosen_ids = {row["chunk_id"] for row in chosen}
    diversity_selected: list[dict[str, Any]] = []
    third_slot_qualified: list[dict[str, Any]] = []
    third_slot_soft_fallback: list[dict[str, Any]] = []
    if len(chosen) < requested_top_k:
        used_sections = {row["section_index"] for row in chosen}
        remaining_candidates = [
            row for row in candidates if row["chunk_id"] not in chosen_ids
        ]
        relevance_window = max(
            1, int(os.getenv("MEDGAP_TOP3_RELEVANCE_WINDOW", "3"))
        )
        # Prefer a normal evidence candidate for the optional third slot.
        # When the source contains fewer than three normal candidates, allow
        # the best soft-quality chunk inside the same narrow relevance window
        # rather than treating the soft label as a hidden hard gate.
        third_slot_qualified = [
            row
            for row in remaining_candidates[:relevance_window]
            if row.get("quality_class") == "evidence_candidate"
        ]
        third_slot_pool = third_slot_qualified
        if not third_slot_pool:
            third_slot_soft_fallback = remaining_candidates[:relevance_window]
            third_slot_pool = third_slot_soft_fallback
        if third_slot_pool:
            third = next(
                (
                    row
                    for row in third_slot_pool
                    if row["section_index"] not in used_sections
                ),
                third_slot_pool[0],
            )
            chosen.append(third)
            chosen_ids.add(third["chunk_id"])
            diversity_selected.append(third)
    ranked = chosen + [row for row in candidates if row["chunk_id"] not in chosen_ids]
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    identity = getattr(backend, "identity", {})
    return ranked, {
        "identity": dict(identity) if isinstance(identity, dict) else {},
        "query_bundle": {
            "enabled": bool(question and question.casefold() != query.casefold()),
            "focused_weight": 0.6,
            "original_weight": 0.4,
        },
        "bm25_top_n": bm25_top_n,
        "dense_top_n": dense_top_n,
        "fusion_max": fusion_max,
        "fusion_candidates": [row["chunk_id"] for row in candidates],
        "dense": dense_diag,
        "cross_encoder": cross_diag,
        "reader": {
            **reader_diag,
            "selection_policy": "medcpt_anchor_reader_complement_relevance_window_top3",
            "anchor_chunk_id": anchor["chunk_id"],
            "reader_pool_chunk_ids": [row["chunk_id"] for row in reader_pool],
            "reader_selected_chunk_ids": [
                row["chunk_id"] for row in reader_selected
            ],
            "diversity_selected_chunk_ids": [
                row["chunk_id"] for row in diversity_selected
            ],
            "third_slot_qualified_chunk_ids": [
                row["chunk_id"] for row in third_slot_qualified
            ],
            "third_slot_soft_fallback_chunk_ids": [
                row["chunk_id"] for row in third_slot_soft_fallback
            ],
            "returned_chunk_ids": [row["chunk_id"] for row in chosen],
        },
        "quality_gate": {
            "policy": "hard_boilerplate_soft_heading_v1",
            "excluded_chunks": sum(
                not bool(row.get("quality_eligible")) for row in chunks
            ),
            "soft_penalized_chunks": sum(
                bool(row.get("quality_eligible"))
                and float(row.get("quality_penalty") or 0.0) > 0
                for row in chunks
            ),
            "eligible_chunks": len(eligible_chunks),
        },
    }
