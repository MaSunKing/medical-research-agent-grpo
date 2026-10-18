"""Trafilatura extraction as a selection signal over original HTML nodes.

No generated text is used as evidence. Scientific tables, explicit article/main
prose, quantities, negation and recommendations override extractor omissions.
"""
import hashlib
import re
from .browser import clean_text, FACTUAL_PROSE

VERSION = 'trafilatura_original_dom_selection_v1'

def select_html_body(soup, original):
    from trafilatura import bare_extraction, __version__
    payload = original if isinstance(original, bytes) else str(original).encode('utf8')
    audit = {'version':VERSION, 'library':'trafilatura', 'library_version':__version__,
             'raw_source_sha256':hashlib.sha256(payload).hexdigest(),
             'excluded_blocks':[], 'protected_blocks':0, 'coverage_verified':False}
    # Recall mode retains tables/short recommendations; duplicate removal belongs
    # to our audited span layer, not Trafilatura's process-global LRU.
    doc = bare_extraction(str(soup), favor_recall=True, include_comments=False,
                          include_tables=True, include_links=False, deduplicate=False)
    body = getattr(doc, 'body', None) if doc is not None else None
    extracted = clean_text(' '.join(body.itertext())) if body is not None else ''
    if not extracted:
        audit['status'] = 'empty_extraction_original_parser_fallback'
        return audit
    audit['status'] = 'original_dom_selection'
    audit['extracted_text_sha256'] = hashlib.sha256(extracted.encode()).hexdigest()
    for index, node in enumerate(list(soup.find_all(['p','li']))):
        if node.name is None or node.attrs is None or node.find_parent(['p','li','table']):
            continue
        text = clean_text(node.get_text(' ',strip=True))
        if not text or text in extracted:
            continue
        # Preserve body semantics even when an external extractor under-selects.
        protected = bool(node.find_parent(['main','article']) or FACTUAL_PROSE.search(text)
                         or re.search(r'\d|\b(?:not|no|without|versus|compared|risk|benefit|adverse|treatment|participants|patients)\b', text, re.I))
        if protected:
            audit['protected_blocks'] += 1
            continue
        audit['excluded_blocks'].append({'block_index':index, 'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                        'chars':len(text), 'reason':'outside_extracted_body'})
        node.decompose()
    return audit
