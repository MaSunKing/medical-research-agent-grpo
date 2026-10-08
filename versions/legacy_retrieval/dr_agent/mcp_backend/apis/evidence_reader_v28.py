# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Question-conditioned, provenance-preserving passage selector for V28.

The reader may only select indices of existing chunks.  It is deliberately
unable to rewrite text or mint citation identifiers.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import requests


READER_VERSION = "medgap_evidence_reader_v28"


def _json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    # Qwen may still emit a thinking wrapper even when the chat template asks
    # for non-thinking output.  Only parse the policy-owned answer after the
    # final closing wrapper; never interpret numbers from the reasoning text.
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1].strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
        if parsed is None:
            # Local readers commonly return either a bare index list or a
            # JSON object truncated immediately after that list.  Recover only
            # the explicitly named selected_indices field (or an output that
            # consists solely of a list); _validated_indices still enforces
            # integer type, range, uniqueness, and Top-K.
            named = re.search(
                r"[\"']?selected_indices[\"']?\s*:\s*\[([^\]]*)",
                value,
                flags=re.IGNORECASE,
            )
            bare = re.fullmatch(r"\s*\[([^\]]*)\]\s*", value)
            candidate = named or bare
            if candidate:
                parsed = {
                    "selected_indices": [
                        int(item)
                        for item in re.findall(r"(?<![\d.])-?\d+(?![\d.])", candidate.group(1))
                    ]
                }
            else:
                # Some local Qwen generations omit JSON brackets but otherwise
                # contain only the requested selector, for example
                # ``selected_indices: 0, 2`` or simply ``0, 2``. Accept only a
                # tightly bounded whole-output form so numbers in prose or
                # hidden reasoning can never become indices.
                compact = re.sub(r"\s+", " ", value).strip()
                labelled = re.fullmatch(
                    r"(?i)(?:selected[_ ]indices?|indices?)\s*(?::|=|are)\s*"
                    r"\[?\s*((?:-?\d+\s*(?:,|and)?\s*)+)\]?\s*[.;]?",
                    compact,
                )
                bare_numbers = re.fullmatch(
                    r"\[?\s*((?:-?\d+\s*(?:,|and)?\s*)+)\]?\s*[.;]?",
                    compact,
                )
                selector = labelled or bare_numbers
                if selector:
                    parsed = {
                        "selected_indices": [
                            int(item)
                            for item in re.findall(r"-?\d+", selector.group(1))
                        ]
                    }
                else:
                    raise ValueError("evidence reader returned no parseable selected_indices")
    if isinstance(parsed, list):
        parsed = {"selected_indices": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("evidence reader response must be a JSON object")
    return parsed


def _validated_indices(payload: dict[str, Any], count: int, top_k: int) -> list[int]:
    raw = payload.get("selected_indices")
    if not isinstance(raw, list):
        raise ValueError("evidence reader omitted selected_indices")
    result: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < count and index not in result:
            result.append(index)
        if len(result) >= top_k:
            break
    if not result:
        raise ValueError("evidence reader selected no valid chunks")
    return result


class EvidenceReaderClient:
    """Small OpenAI-compatible client used only as a selector."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = str(
            base_url or os.getenv("MEDGAP_EVIDENCE_READER_BASE_URL") or ""
        ).rstrip("/")
        self.model = str(model or os.getenv("MEDGAP_EVIDENCE_READER_MODEL") or "")
        self.api_key = str(api_key or os.getenv("MEDGAP_EVIDENCE_READER_API_KEY") or "")
        self.timeout = float(timeout or os.getenv("MEDGAP_EVIDENCE_READER_TIMEOUT", "90"))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def select(
        self,
        *,
        question: str,
        focused_query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[int], dict[str, Any]]:
        if not self.enabled:
            return list(range(min(top_k, len(chunks)))), {
                "version": READER_VERSION,
                "enabled": False,
                "fallback": "upstream_rank",
            }
        candidate_lines = []
        for index, chunk in enumerate(chunks):
            candidate_lines.append(
                f"[{index}] heading={chunk.get('heading', '')}\n"
                f"{str(chunk.get('text') or '')}"
            )
        prompt = (
            "Select the original evidence chunks that most directly support an accurate answer. "
            "Do not rewrite evidence and do not infer hidden requirements. Cover distinct subquestions "
            "with complementary chunks. Prefer human outcome data, direct recommendations, quantitative "
            "results, and relevant safety evidence. Do not select a methods, GRADE, or certainty-only "
            "chunk when a direct outcome chunk is available. Return JSON "
            f"only as {{\"selected_indices\":[0],\"reason\":\"short\"}} with at most {top_k} indices.\n\n"
            f"Question/context:\n{question}\n\nFocused browse query:\n{focused_query}\n\n"
            "Candidate chunks:\n" + "\n\n".join(candidate_lines)
        )
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a conservative medical evidence passage selector."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 160,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        try:
            indices = _validated_indices(_json_object(content), len(chunks), top_k)
        except ValueError as exc:
            excerpt = re.sub(r"\s+", " ", content).strip()[:240]
            raise ValueError(f"{exc}; reader_output={excerpt!r}") from exc
        return indices, {
            "version": READER_VERSION,
            "enabled": True,
            "model": self.model,
            "selected_indices": indices,
        }


class LocalEvidenceReader:
    """Direct local Qwen reader for workstation experiments and deployment."""

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self._tokenizer = None
        self._model = None
        self._selection_cache: dict[str, list[int]] = {}
        # FastMCP can dispatch many Browse requests concurrently, but a single
        # local Transformers generation model is not a thread-safe inference
        # service. Serialize load/cache/generate to prevent duplicate 8B model
        # loads, overlapping KV allocations, and non-deterministic CUDA OOMs.
        self._inference_lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return True

    def _load(self):
        with self._inference_lock:
            if self._model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True, trust_remote_code=True
                )
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map={"": 0},
                    quantization_config=quantization,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
        return self._tokenizer, self._model

    def select(
        self,
        *,
        question: str,
        focused_query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[int], dict[str, Any]]:
        # One local Reader instance backs all concurrent MCP Browse requests.
        # Keep the cache lookup and complete CUDA generation in the same lock.
        with self._inference_lock:
            return self._select_serialized(
                question=question,
                focused_query=focused_query,
                chunks=chunks,
                top_k=top_k,
            )

    def _select_serialized(
        self,
        *,
        question: str,
        focused_query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[int], dict[str, Any]]:
        import torch

        tokenizer, model = self._load()
        chunk_char_limit = max(
            600, min(3000, int(os.getenv("MEDGAP_EVIDENCE_READER_CHUNK_CHARS", "1800")))
        )
        max_input_tokens = max(
            1024, min(8192, int(os.getenv("MEDGAP_EVIDENCE_READER_MAX_INPUT_TOKENS", "3072")))
        )
        reader_chunks = list(chunks)

        def render(limit: int) -> tuple[str, str]:
            candidate_text = "\n\n".join(
                f"[{index}] heading={row.get('heading', '')}\n"
                f"{str(row.get('text') or '')[:limit]}"
                for index, row in enumerate(reader_chunks)
            )
            prompt = (
                "Choose complementary original chunks that directly answer all distinct parts of the medical "
                "question. Prioritize direct recommendations, human outcome comparisons, quantitative results, "
                "and adverse events. Do not select a methods, GRADE, or certainty-only chunk when a direct "
                "outcome chunk is available. Do not rewrite or combine chunks. Return one compact JSON line "
                "only, then stop immediately. Use the form {\"selected_indices\":[0,1]} with at most "
                f"{top_k} distinct indices.\n\nQuestion/context:\n{question}\n\nFocused query:\n"
                f"{focused_query}\n\nCandidates:\n{candidate_text}"
            )
            messages = [
                {"role": "system", "content": "You are a conservative medical evidence passage selector."},
                {"role": "user", "content": prompt},
            ]
            return candidate_text, tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

        candidates, rendered = render(chunk_char_limit)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=False)
        input_tokens = int(inputs.input_ids.shape[1])
        # Keep all high-ranked candidates when possible, but shrink each raw
        # passage before failing. If the fixed question/prompt overhead is still
        # too large, drop only the lowest-ranked tail candidates while retaining
        # at least Top-K. This preserves index provenance and avoids blind token
        # truncation in the middle of the candidate list.
        while input_tokens > max_input_tokens and chunk_char_limit > 600:
            proportional = int(chunk_char_limit * max_input_tokens / input_tokens * 0.9)
            chunk_char_limit = max(600, min(chunk_char_limit - 1, proportional))
            candidates, rendered = render(chunk_char_limit)
            inputs = tokenizer(rendered, return_tensors="pt", truncation=False)
            input_tokens = int(inputs.input_ids.shape[1])
        while input_tokens > max_input_tokens and len(reader_chunks) > max(1, top_k):
            reader_chunks.pop()
            candidates, rendered = render(chunk_char_limit)
            inputs = tokenizer(rendered, return_tensors="pt", truncation=False)
            input_tokens = int(inputs.input_ids.shape[1])
        if input_tokens > max_input_tokens:
            raise ValueError(
                f"evidence reader input has {input_tokens} tokens after bounded compaction; "
                f"limit is {max_input_tokens}"
            )

        cache_key = json.dumps(
            [question, focused_query, top_k, candidates], ensure_ascii=False, separators=(",", ":")
        )
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return list(cached), {
                "version": READER_VERSION,
                "enabled": True,
                "backend": "local_transformers_4bit",
                "model": self.model_path,
                "selected_indices": list(cached),
                "cache_hit": True,
                "input_tokens": input_tokens,
                "max_input_tokens": max_input_tokens,
                "chunk_char_limit": chunk_char_limit,
                "reader_candidate_count": len(reader_chunks),
            }
        inputs = inputs.to("cuda")
        max_new_tokens = max(
            24, min(96, int(os.getenv("MEDGAP_EVIDENCE_READER_MAX_NEW_TOKENS", "48")))
        )
        max_time_seconds = max(
            2.0, min(30.0, float(os.getenv("MEDGAP_EVIDENCE_READER_MAX_TIME", "12")))
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                max_time=max_time_seconds,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        output_tokens = int(output.shape[1] - inputs.input_ids.shape[1])
        content = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        indices = _validated_indices(_json_object(content), len(chunks), top_k)
        self._selection_cache[cache_key] = list(indices)
        return indices, {
            "version": READER_VERSION,
            "enabled": True,
            "backend": "local_transformers_4bit",
            "model": self.model_path,
            "selected_indices": indices,
            "cache_hit": False,
            "input_tokens": input_tokens,
            "max_input_tokens": max_input_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": max_new_tokens,
            "max_time_seconds": max_time_seconds,
            "chunk_char_limit": chunk_char_limit,
            "reader_candidate_count": len(reader_chunks),
            "elapsed_ms": elapsed_ms,
        }


_CLIENT: EvidenceReaderClient | LocalEvidenceReader | None = None


def get_evidence_reader() -> EvidenceReaderClient | LocalEvidenceReader:
    global _CLIENT
    if _CLIENT is None:
        local_model = str(os.getenv("MEDGAP_EVIDENCE_READER_LOCAL_MODEL") or "").strip()
        _CLIENT = LocalEvidenceReader(local_model) if local_model else EvidenceReaderClient()
    return _CLIENT
