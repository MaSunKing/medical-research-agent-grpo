# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Local parsers for medical HTML, PubMed XML, and PMC/JATS XML."""

from __future__ import annotations

import re
import math
import hashlib
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from .browse_preprocess import preprocess_document, strip_html_templates, classify_section
from .table_integrity import table_signature
from bs4 import BeautifulSoup
from lxml import etree


CHUNKER_VERSION = "medical_chunker_atomic_word_boundaries_v3"
RERANKER_VERSION = "local_bm25_focus80_question20_v71_2f"
HYBRID_RERANKER_VERSION = "bm25_bge_minilm_v27"
V28_RERANKER_VERSION = "medcpt_reader_v28"


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _join_text(node) -> str:
    return _clean(" ".join(node.itertext())) if node is not None else ""


def _make_result(
    *, title: str, sections: List[Dict[str, str]], source_format: str
) -> Dict[str, Any]:
    sections = [
        {**section, "heading": _clean(section.get("heading")), "text": _clean(section.get("text"))}
        for section in sections
        if _clean(section.get("text"))
    ]
    return {
        "title": _clean(title),
        "text": "\n\n".join(section["text"] for section in sections),
        "sections": sections,
        "metadata": {
            "source_format": source_format,
            "section_count": len(sections),
        },
    }


def parse_medical_html(content: bytes | str) -> Dict[str, Any]:
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "form"]):
        tag.decompose()
    template_audit = strip_html_templates(soup)

    title_node = (
        soup.select_one("h1.content-title")
        or soup.select_one("h1")
        or soup.select_one("meta[name='citation_title']")
        or soup.title
    )
    if title_node and title_node.name == "meta":
        title = _clean(title_node.get("content"))
    else:
        title = _clean(title_node.get_text(" ", strip=True) if title_node else "")

    root = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one(".article")
        or soup.body
        or soup
    )
    sections: List[Dict[str, str]] = []
    heading = "Document"
    paragraphs: List[str] = []

    def flush() -> None:
        nonlocal paragraphs
        text = _clean(" ".join(paragraphs))
        if text:
            sections.append({"heading": heading, "text": text})
        paragraphs = []

    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"], recursive=True):
        if node.find_parent('table'):
            continue
        text = _clean(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name.startswith("h"):
            flush()
            heading = text
        elif node.name == "table":
            flush()
            rows = [' | '.join(_clean(c.get_text(' ',strip=True)) for c in row.find_all(['th','td'],recursive=False))
                    for row in node.find_all('tr')]
            sections.append({'heading':heading,'text':'\n'.join(rows),
                             'structure_kind':'table','structure_origin':'html_table',
                             'table_structure_complete':bool(node.find('th') and node.find('td'))})
        elif not node.find_parent(["p", "li", "tr"]):
            paragraphs.append(text)
    flush()
    result = _make_result(title=title, sections=sections, source_format="html")
    result['metadata']['html_template_exclusions'] = template_audit
    return result


def parse_medical_markdown(content: bytes | str) -> Dict[str, Any]:
    """Normalize MinerU/Crawl markdown while preserving heading boundaries."""
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    title = ""
    heading = "Document"
    sections: List[Dict[str, str]] = []
    lines: List[str] = []

    def flush() -> None:
        nonlocal lines
        body = _clean(" ".join(lines))
        if body:
            sections.append({"heading": heading, "text": body})
        lines = []

    for raw_line in text.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", raw_line)
        if match:
            flush()
            heading = _clean(match.group(2)) or "Section"
            if not title and len(match.group(1)) == 1:
                title = heading
        elif raw_line.strip():
            lines.append(raw_line.strip())
    flush()
    if not sections and _clean(text):
        sections = [{"heading": "Document", "text": _clean(text)}]
    return _make_result(title=title, sections=sections, source_format="markdown")


def _parse_pubmed_xml(root) -> Dict[str, Any]:
    article = root.find(".//Article")
    if article is None:
        raise ValueError("PubMed XML does not contain an Article element")
    title = _join_text(article.find(".//ArticleTitle"))
    sections = []
    for node in article.findall(".//Abstract/AbstractText"):
        heading = node.get("Label") or node.get("NlmCategory") or "Abstract"
        text = _join_text(node)
        if text:
            sections.append({"heading": heading, "text": text})
    return _make_result(title=title, sections=sections, source_format="pubmed_xml")


def _parse_jats_xml(root) -> Dict[str, Any]:
    title = _join_text(root.find(".//article-title"))
    sections: List[Dict[str, str]] = []

    abstracts = root.findall(".//abstract")
    # Some PMC records expose a short teaser before key points and the real
    # structured abstract. Taking only the first node silently discards the
    # results. Prefer the untyped/full abstract, retain key points, and use a
    # teaser only when it is the sole abstract available.
    ranked_abstracts = sorted(
        abstracts,
        key=lambda node: (
            0 if not node.get("abstract-type") else
            1 if node.get("abstract-type") == "key-points" else
            3 if node.get("abstract-type") == "teaser" else 2
        ),
    )
    seen_abstracts = set()
    for abstract in ranked_abstracts:
        abstract_type = abstract.get("abstract-type") or "full"
        if abstract_type == "teaser" and len(abstracts) > 1:
            continue
        abstract_text = _clean(" ".join(_join_text(p) for p in abstract.findall(".//p")))
        if not abstract_text:
            abstract_text = _join_text(abstract)
        if abstract_text and abstract_text not in seen_abstracts:
            seen_abstracts.add(abstract_text)
            heading = "Abstract" if abstract_type == "full" else f"Abstract ({abstract_type})"
            sections.append({"heading": heading, "text": abstract_text})

    body = root.find(".//body")
    if body is not None:
        # Preserve source order and table ownership rather than flattening
        # table-wrap descendants into their containing paragraph.
        for node in body.iter():
            ancestors=list(node.iterancestors())
            if node.tag not in {'p','table-wrap'} or any(a.tag=='table-wrap' for a in ancestors):
                continue
            sec=next((a for a in ancestors if a.tag=='sec'),None)
            heading=_join_text(sec.find('./title')) if sec is not None else 'Body'
            if node.tag=='table-wrap':
                label=_join_text(node.find('./label'))
                caption=_join_text(node.find('./caption'))
                rows=[' | '.join(_join_text(c) for c in row if c.tag in {'th','td'})
                      for row in node.findall('.//tr')]
                footer=_join_text(node.find('./table-wrap-foot'))
                text=_clean(' '.join(filter(None,[label,caption]+rows+[footer])))
                if text:
                    sections.append({'heading':_clean(' '.join(filter(None,[heading,label,caption]))),
                                     'text':text,'structure_kind':'table',
                                     'structure_origin':'jats_table_wrap',
                                     'table_structure_complete':bool(node.find('.//th') is not None and node.find('.//td') is not None)})
            else:
                import copy
                paragraph=copy.deepcopy(node)
                for table in paragraph.findall('.//table-wrap'):
                    # Keep the original prose tail following an embedded table.
                    tail=table.tail or ''
                    previous=table.getprevious()
                    parent=table.getparent()
                    if previous is not None: previous.tail=(previous.tail or '')+tail
                    else: parent.text=(parent.text or '')+tail
                    parent.remove(table)
                text=_join_text(paragraph)
                if text:
                    sections.append({'heading':heading or 'Section','text':text})
        if not any(section["heading"] != "Abstract" for section in sections):
            body_text = _join_text(body)
            if body_text:
                sections.append({"heading": "Body", "text": body_text})
    return _make_result(title=title, sections=sections, source_format="pmc_jats_xml")


def parse_medical_xml(content: bytes | str) -> Dict[str, Any]:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
    root = etree.fromstring(raw, parser=parser)
    # Remove XML namespaces so standard PubMed/JATS paths remain stable.
    for node in root.iter():
        if isinstance(node.tag, str) and "}" in node.tag:
            node.tag = node.tag.split("}", 1)[1]
    etree.cleanup_namespaces(root)

    if root.find(".//PubmedArticle") is not None or root.tag == "PubmedArticle":
        return _parse_pubmed_xml(root)
    if root.tag == "article" or root.find(".//article-meta") is not None:
        return _parse_jats_xml(root)
    raise ValueError(f"Unsupported medical XML root: {root.tag}")


def parse_medical_document(
    content: bytes | str, source_format: Optional[str] = None
) -> Dict[str, Any]:
    if source_format:
        normalized = source_format.lower().lstrip(".")
    else:
        prefix = content[:200] if isinstance(content, bytes) else content[:200]
        if isinstance(prefix, bytes):
            prefix = prefix.decode("utf-8", errors="ignore")
        normalized = "html" if re.search(r"<!doctype\s+html|<html\b", prefix, re.I) else "xml"

    if normalized in {"html", "htm"}:
        return parse_medical_html(content)
    if normalized == "xml":
        return parse_medical_xml(content)
    raise ValueError(f"Unsupported source format: {source_format}")


def _tokenize_for_retrieval(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def chunk_medical_document(
    document: Dict[str, Any],
    *,
    source_id: str = "document",
    chunk_chars: int = 1800,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Split normalized sections into stable, bounded retrieval chunks."""
    if chunk_chars < 200:
        raise ValueError("chunk_chars must be at least 200")
    overlap_chars = max(0, min(overlap_chars, chunk_chars // 3))
    chunks = []
    for section_index, section in enumerate(document.get("sections", [])):
        heading = _clean(section.get("heading")) or "Section"
        text = _clean(section.get("text"))
        chunk_index = 0
        spans = section.get('retrieval_spans', [{'start_char':0,'end_char':len(text),'kind':'body'}])
        for span in spans:
            start = span['start_char']; limit = span['end_char']
            assert 0 <= start <= limit <= len(text)
            while start < limit:
                # Explicit source tables are atomic; budget handoff either
                # admits the complete structure or retains an exclusion receipt.
                end = limit if span.get('kind') == 'table' else min(limit, start + chunk_chars)
                if end < limit:
                    boundary = text.rfind(" ", start + chunk_chars // 2, end)
                    if boundary > start: end = boundary
                raw_chunk = text[start:end]
                left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
                right_trim = len(raw_chunk) - len(raw_chunk.rstrip())
                chunk_text = raw_chunk.strip()
                if chunk_text:
                    start_char = start + left_trim
                    end_char = end - right_trim
                    span_kind = str(span.get("kind") or "body")
                    structure = table_signature(chunk_text, span_kind)
                    structure_kind = structure["structure_kind"]
                    chunks.append({
                        "chunk_id": f"{source_id}#s{section_index}-c{chunk_index}",
                        "heading": heading,
                        "text": chunk_text,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "start_char": start_char,
                        "end_char": end_char,
                        "content_span_start_char": span['start_char'],
                        "content_span_end_char": limit,
                        "structure_kind": structure_kind,
                        "structure_audit": structure,
                        "structure_origin": section.get('structure_origin'),
                        "text_sha256": hashlib.sha256(
                            chunk_text.encode("utf-8", errors="replace")
                        ).hexdigest(),
                    })
                    chunk_index += 1
                if end >= limit: break
                next_start = max(start + 1, end - overlap_chars)
                # Character overlap must never create a Latin word fragment.
                # CJK text has no equivalent whitespace boundary and is left intact.
                if (next_start > span['start_char'] and next_start < limit
                        and text[next_start-1].isalnum() and text[next_start].isalnum()
                        and (text[next_start-1].isascii() or text[next_start].isascii())):
                    boundary = text.find(" ", next_start, min(limit, next_start + 128))
                    next_start = boundary + 1 if boundary >= 0 else end
                start = next_start
    return chunks


def select_relevant_chunks(
    document: Dict[str, Any],
    query: str,
    *,
    source_id: str = "document",
    top_k: int = 3,
    max_chars: int = 3000,
    chunk_chars: int = 1800,
    overlap_chars: int = 200,
    max_output_tokens: Optional[int] = None,
    tokenizer: Any = None,
    include_observability: bool = False,
    retrieval_mode: Optional[str] = None,
    hybrid_backend: Any = None,
    original_question: Optional[str] = None,
    evidence_reader: Any = None,
) -> Dict[str, Any]:
    """Return a hard-budgeted Agent view rather than the complete article."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1 or max_chars < 200:
        raise ValueError("top_k must be positive and max_chars must be at least 200")

    document = preprocess_document(document)
    chunks = chunk_medical_document(
        document,
        source_id=source_id,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    for chunk in chunks:
        kind, reason = classify_section(chunk['heading'], chunk['text'])
        chunk['quality_eligible'] = kind == 'content'
        chunk['quality_reason'] = reason
    rankable = [chunk for chunk in chunks if chunk['quality_eligible']]
    query_terms = _tokenize_for_retrieval(query)
    tokenized = [
        _tokenize_for_retrieval((chunk["heading"] + " ") * 4 + chunk["text"])
        for chunk in chunks
    ]
    document_frequency = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    total_chunks = max(1, len(chunks))

    for chunk, terms in zip(chunks, tokenized):
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (total_chunks - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(terms) / max(1, average_length)
            )
            score += inverse_document_frequency * frequency * 2.5 / denominator
        chunk["bm25_score"] = round(score, 6)
        chunk["score"] = chunk["bm25_score"]

    mode = str(
        retrieval_mode or os.getenv("MEDGAP_PASSAGE_RETRIEVAL_MODE", "bm25")
    ).strip().lower()
    hybrid_diagnostics = None
    if mode == "hybrid":
        from .hybrid_passage_retriever import hybrid_rank_chunks

        ranked, hybrid_diagnostics = hybrid_rank_chunks(
            query,
            rankable,
            backend=hybrid_backend,
            bm25_top_n=max(1, int(os.getenv("MEDGAP_BM25_TOP_N", "8"))),
            dense_top_n=max(1, int(os.getenv("MEDGAP_DENSE_TOP_N", "8"))),
            fusion_max=max(1, int(os.getenv("MEDGAP_FUSION_MAX", "12"))),
        )
    elif mode == "bm25":
        anchor_terms = set(_tokenize_for_retrieval(str(original_question or "")))
        anchor_scores = []
        for terms in tokenized:
            frequencies = Counter(terms)
            score = 0.0
            for term in anchor_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(terms) / max(1, average_length))
                score += idf * frequency * 2.5 / denominator
            anchor_scores.append(score)
        focus_max = max((c["bm25_score"] for c in chunks), default=0.0)
        anchor_max = max(anchor_scores, default=0.0)
        for chunk, anchor in zip(chunks, anchor_scores):
            chunk["original_question_bm25_score"] = anchor
            chunk["score"] = (0.8 * chunk["bm25_score"] / focus_max if focus_max > 0 else 0.0) + (0.2 * anchor / anchor_max if anchor_max > 0 else 0.0)
            chunk["ranking_protocol"] = "v71_2f_focus80_question20"
        ranked = sorted(
            rankable,
            key=lambda item: (
                -item["score"], item["section_index"], item["chunk_index"]
            ),
        )
        for rank, chunk in enumerate(ranked, 1):
            chunk["rank"] = rank
            chunk["bm25_rank"] = rank
    elif mode == "v28":
        from .medical_passage_retriever_v28 import v28_rank_chunks

        ranked, hybrid_diagnostics = v28_rank_chunks(
            query,
            chunks,
            question=str(original_question or query),
            top_k=top_k,
            backend=hybrid_backend,
            reader=evidence_reader,
            bm25_top_n=max(1, int(os.getenv("MEDGAP_BM25_TOP_N", "10"))),
            dense_top_n=max(1, int(os.getenv("MEDGAP_DENSE_TOP_N", "10"))),
            fusion_max=max(1, int(os.getenv("MEDGAP_FUSION_MAX", "16"))),
        )
    else:
        raise ValueError("MEDGAP_PASSAGE_RETRIEVAL_MODE must be bm25, hybrid, or v28")
    # Prefer section diversity so two adjacent chunks from the introduction do
    # not crowd out a slightly lower-scoring Methods/Results passage.
    if mode == "v28":
        # The evidence reader has already made an explicit question-conditioned
        # selection.  Reapplying section diversity here would silently replace
        # one of its chosen original chunks.
        returned_ids = set(
            hybrid_diagnostics.get("reader", {}).get("returned_chunk_ids") or []
        )
        candidates = (
            [row for row in ranked if row.get("chunk_id") in returned_ids]
            if returned_ids
            else list(ranked[:top_k])
        )
    else:
        candidates = []
        used_sections = set()
        for chunk in ranked:
            if chunk["section_index"] not in used_sections:
                candidates.append(chunk)
                used_sections.add(chunk["section_index"])
            if len(candidates) >= top_k:
                break
        if len(candidates) < top_k:
            selected_ids = {chunk["chunk_id"] for chunk in candidates}
            candidates.extend(
                chunk for chunk in ranked if chunk["chunk_id"] not in selected_ids
            )

    selected = []
    remaining = max_chars
    for selection_rank, chunk in enumerate(candidates[:top_k], 1):
        if remaining <= 0:
            break
        selected_chunk = dict(chunk)
        selected_chunk["selection_rank"] = selection_rank
        selected_chunk["pre_budget_text_sha256"] = selected_chunk["text_sha256"]
        selected_chunk["text"] = selected_chunk["text"][:remaining].rstrip()
        if not selected_chunk["text"]:
            break
        selected.append(selected_chunk)
        remaining -= len(selected_chunk["text"])

    token_count = None
    token_budget_mode = "disabled"
    if max_output_tokens is not None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if tokenizer is not None:
            token_budget_mode = "tokenizer"
            remaining_tokens = max_output_tokens
            token_count = 0
            token_limited = []
            for index, chunk in enumerate(selected):
                token_ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
                if remaining_tokens <= 0:
                    break
                # Reserve a fair share for every selected Top-K chunk. The old
                # sequential allocation allowed the first chunk to consume the
                # entire budget and left a second Results/Safety chunk as one
                # token, despite advertising a bounded Top-K to the Agent.
                chunks_left = len(selected) - index
                fair_cap = max(1, remaining_tokens // max(1, chunks_left))
                kept_ids = token_ids[:fair_cap]
                limited_chunk = dict(chunk)
                limited_chunk["text"] = tokenizer.decode(
                    kept_ids, skip_special_tokens=True
                ).strip()
                if limited_chunk["text"]:
                    token_limited.append(limited_chunk)
                    token_count += len(kept_ids)
                    remaining_tokens -= len(kept_ids)
            selected = token_limited
        else:
            # Conservative dependency-free fallback. A character can map to more
            # than one token for unusual Unicode, so production should configure
            # MEDICAL_TOKENIZER_PATH and use the exact tokenizer branch above.
            token_budget_mode = "conservative_character_fallback"
            remaining_chars = max_output_tokens
            fallback_limited = []
            for index, chunk in enumerate(selected):
                if remaining_chars <= 0:
                    break
                limited_chunk = dict(chunk)
                chunks_left = len(selected) - index
                fair_cap = max(1, remaining_chars // max(1, chunks_left))
                limited_chunk["text"] = limited_chunk["text"][:fair_cap].rstrip()
                if limited_chunk["text"]:
                    fallback_limited.append(limited_chunk)
                    remaining_chars -= len(limited_chunk["text"])
            selected = fallback_limited

    for chunk in selected:
        chunk["returned_text_sha256"] = hashlib.sha256(
            chunk["text"].encode("utf-8", errors="replace")
        ).hexdigest()

    result = {
        "title": document.get("title", ""),
        "query": query,
        "chunks": selected,
        "metadata": {
            "source_id": source_id,
            "retrieval": (
                "medcpt_reader_v28" if mode == "v28" else
                "hybrid_bm25_bge_minilm" if mode == "hybrid" else "local_bm25"
            ),
            "ranking_protocol": RERANKER_VERSION if mode == "bm25" else mode,
            "browse_preprocessing": document["browse_preprocessing"],
            "total_chunks": len(chunks),
            "returned_chunks": len(selected),
            "max_chars": max_chars,
            "max_output_tokens": max_output_tokens,
            "output_tokens": token_count,
            "token_budget_mode": token_budget_mode,
            "truncated": len(selected) < len(chunks),
        },
    }
    if include_observability:
        chunk_fields = (
            "chunk_id", "heading", "section_index", "chunk_index", "start_char",
            "end_char", "text_sha256", "score", "rank", "bm25_score", "bm25_rank",
            "content_span_start_char", "content_span_end_char",
            "dense_score", "dense_rank", "fusion_candidate", "fusion_reasons",
            "cross_encoder_score", "medcpt_score", "medcpt_rank", "medcpt_cross_score",
            "quality_class", "quality_eligible", "quality_penalty",
            "quality_evidence_signal", "quality_heading_mismatch", "quality_reason",
        )
        returned_fields = chunk_fields + (
            "selection_rank", "pre_budget_text_sha256", "returned_text_sha256",
        )
        result["_retrieval_observability"] = {
            "source_id": source_id,
            "chunker": {
                "name": "medical_document_chunker",
                "version": CHUNKER_VERSION,
                "chunk_chars": chunk_chars,
                "overlap_chars": overlap_chars,
            },
            "reranker": {
                "name": (
                    "medcpt_reader_v28" if mode == "v28" else
                    "hybrid_bm25_bge_minilm" if mode == "hybrid" else "local_bm25"
                ),
                "version": (
                    V28_RERANKER_VERSION if mode == "v28" else
                    HYBRID_RERANKER_VERSION if mode == "hybrid" else RERANKER_VERSION
                ),
                "heading_repeat": 4,
                "k1": 1.5,
                "b": 0.75,
                "section_diversity": True,
                "hybrid": hybrid_diagnostics,
            },
            "budgets": {
                "top_k": top_k,
                "max_chars": max_chars,
                "max_output_tokens": max_output_tokens,
                "token_budget_mode": token_budget_mode,
            },
            "all_chunks": [
                {key: chunk.get(key) for key in chunk_fields}
                for chunk in sorted(
                    chunks,
                    key=lambda item: (
                        int(item.get("bm25_rank") or 10**9),
                        item["section_index"],
                        item["chunk_index"],
                    ),
                )
            ],
            "returned_chunks": [
                {key: chunk.get(key) for key in returned_fields} for chunk in selected
            ],
            "browse_preprocessing": document['browse_preprocessing'],
        }
    return result


# Local inference-only adapter, original ranker unchanged.
from .evidence_handoff_v3 import install_parser as _install_handoff
import sys as _handoff_sys
_install_handoff(_handoff_sys.modules[__name__])
