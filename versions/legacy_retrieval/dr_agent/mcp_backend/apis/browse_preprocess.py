# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Conservative, query-independent preprocessing of parsed Browse sections."""
import copy
import hashlib
import re

VERSION = 'browse_sections_generic_spans_v3'

# Combine template purpose with text shape, never site/domain or topic rules.
TEMPLATE_HEADING = re.compile(r'导航|热门产品|热门推荐|更多推荐|扫码|扫一扫|分享|技术支持|版权|友情链接|社区|活动|圈层|开发者|'
    r'\b(?:navigation|related|recommended|share|cookie|support|products|community|newsletter|copyright)\b', re.I)
LEGAL = re.compile(r'copyright|版权所有|all rights reserved|ICP备|公网安备', re.I)
SUPPORT = re.compile(r'技术支持|technical support|本系统由', re.I)
UI = re.compile(r'浏览器|分辨率|IE\s*\d|浏览本页面|browser|resolution', re.I)
FORMATS = re.compile(r'APA|MLA|GB/T\s*7714|格式引文', re.I)
PLACEHOLDER = re.compile(r'\{\{\s*(?:javascript\s*:[^{}]*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:[^{}]*))\s*\}\}', re.I)
CODE = re.compile(r'```.*?```|`[^`]*`', re.S)
PAGE_UI = re.compile(r'欢迎访问|官方网站|分享到|Author information|作者信息|文章历史|原文顺序|文献年度倒序|文中引用次数倒序', re.I)
SORT_UI = re.compile(r'原文顺序|文献年度倒序|文中引用次数倒序|sort by|citation count', re.I)
IMAGE = re.compile(r'!\[[^\]]*\]\(\s*[^\s()]+(?:\s+"[^"]*")?\s*\)')
ORPHAN_IMAGE = re.compile(r'^[A-Za-z0-9][\w./:%?&=+#~-]*\.(?:png|jpe?g|gif|webp|svg)(?:[?#][^\s)]*)?\)', re.I)
REFERENCE_MARKER = re.compile(r'参考文献|\bReferences\b|\bBibliography\b', re.I)
REFERENCE_RECORD = re.compile(r'\b(?:19|20)\d{2}\b|\bdoi\b|\bPMID\b|\bet al\.|\d+\s*:\s*\d+\s*[-–]\s*\d+', re.I)
BIBLIOGRAPHIC_SIGNATURE = re.compile(r'\bdoi\b|\bPMID\b|\d+\s*:\s*\d+\s*[-–]\s*\d+', re.I)


def unresolved_placeholders(text):
    protected = [m.span() for m in CODE.finditer(text)]
    return [m for m in PLACEHOLDER.finditer(text)
            if not any(a <= m.start() < b for a,b in protected)]


def classify_section(heading, text):
    text = clean_text(text)
    placeholders = unresolved_placeholders(text)
    plain = text
    for m in reversed(placeholders): plain=plain[:m.start()]+plain[m.end():]
    prose_sentences = len(re.findall(r'[。！？!?]|(?<!\d)\.(?=\s|$)', plain))
    if placeholders and not plain.strip(' |:-+*'):
        return 'web_boilerplate', 'unrendered_placeholder_only'
    if len(placeholders) >= 2 and prose_sentences < 2 and len(PAGE_UI.findall(plain)) >= 2:
        return 'web_boilerplate', 'unrendered_metadata_shell'
    if len(SORT_UI.findall(text)) >= 2 and len(text) <= 800 and prose_sentences < 2:
        return 'web_boilerplate', 'reference_sort_controls'
    if heading_kind(heading) == 'bibliography' and REFERENCE_RECORD.search(text):
        return 'bibliography', 'bibliographic_records_not_article_body'
    # Prose can discuss copyright/products/sharing; do not censor the topic.
    sentences = len(re.findall(r'[。！？!?]|(?<!\d)\.(?=\s|$)', text))
    substantive = len(text) >= 180 and sentences >= 2
    template_heading = bool(TEMPLATE_HEADING.search(str(heading)))
    # Copyright sentences can make a footer look like prose. Require both
    # multiple legal cues and a template heading before overriding that guard.
    if len(LEGAL.findall(text)) >= 2 and len(text) <= 1200 and (template_heading or not substantive):
        return 'web_boilerplate', 'multiple_legal_template_cues'
    if template_heading and len(text) <= 400 and re.search(r'扫码关注|扫一扫|subscribe|sign up', text, re.I) and re.search(r'领取|代金券|newsletter|subscribe', text, re.I):
        return 'web_boilerplate', 'subscription_or_promotion_controls'
    if substantive:
        return 'content', 'substantive_prose_preserved'
    if SUPPORT.search(text) and UI.search(text) and len(text) <= 600:
        return 'web_boilerplate', 'support_and_browser_ui'
    if len(FORMATS.findall(text)) >= 2 and len(text) <= 400:
        return 'web_boilerplate', 'citation_format_controls'
    short_units = len(text.split()) >= 3
    if TEMPLATE_HEADING.search(str(heading)) and len(text) <= 400 and sentences == 0 and short_units:
        return 'web_boilerplate', 'template_heading_and_short_list'
    return 'content', 'insufficient_template_evidence'


def retrieval_spans(heading, text):
    """Select contiguous original-text spans; never reconstruct a cleaned claim."""
    text = clean_text(text)
    kind, reason = classify_section(heading, text)
    if kind != 'content':
        return [], [{'start_char':0, 'end_char':len(text), 'kind':kind, 'reason':reason}]
    protected = [m.span() for m in CODE.finditer(text)]
    exclusions = []
    for expression, why in ((IMAGE,'markdown_image_asset'), (ORPHAN_IMAGE,'orphan_image_link_fragment')):
        for m in expression.finditer(text):
            if not any(a <= m.start() < b for a,b in protected):
                exclusions.append((m.start(),m.end(),why))
    for m in unresolved_placeholders(text):
        exclusions.append((m.start(),m.end(),'unrendered_template_expression'))
    for m in REFERENCE_MARKER.finditer(text):
        if any(a <= m.start() < b for a,b in protected): continue
        tail = text[m.end():]
        starts_record = re.match(r'\s*[:：]?\s*(?:\d+\s*[.、)]|[A-Z][a-z]+\b)', tail)
        record_shape = BIBLIOGRAPHIC_SIGNATURE.search(tail) or re.search(r'\b2\s*[.、)]', tail)
        if starts_record and record_shape and len(REFERENCE_RECORD.findall(tail)) >= 2:
            exclusions.append((m.start(),len(text),'embedded_bibliography'))
            break
    merged=[]
    for a,b,why in sorted(exclusions):
        if merged and a <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]),merged[-1][2]+';'+why)
        else: merged.append((a,b,why))
    spans=[]; cursor=0
    for a,b,why in merged+[(len(text),len(text),'end')]:
        left=cursor; right=a
        while left < right and text[left].isspace(): left+=1
        while right > left and text[right-1].isspace(): right-=1
        if left < right and re.search(r'[\w\u3400-\u9fff]',text[left:right]):
            spans.append({'start_char':left,'end_char':right,'kind':'body'})
        cursor=max(cursor,b)
    return spans, [{'start_char':a,'end_char':b,'kind':'non_body','reason':why} for a,b,why in merged]


def strip_html_templates(soup):
    """Query-independent structural cleanup; retain an exclusion audit."""
    audit = []
    hints = re.compile(r'(?:^|[-_\s])(?:nav|navbar|footer|sidebar|breadcrumb|related|recommendations?|share|social|cookie|copyright|product-list)(?:$|[-_\s])', re.I)
    for node in list(soup.find_all(['aside', 'div', 'section', 'ul'])):
        if node.name is None:
            continue
        attrs = ' '.join([str(node.get('id', '')), ' '.join(node.get('class', [])), str(node.get('role', ''))])
        text = clean_text(node.get_text(' ', strip=True))
        links = sum(len(clean_text(a.get_text(' ', strip=True))) for a in node.find_all('a'))
        density = min(1.0, links / max(1, len(text)))
        kind, reason = classify_section(attrs, text)
        structural = bool(hints.search(attrs) or node.get('role') == 'navigation')
        prose = len(text) >= 180 and len(re.findall(r'[。！？!?]|\.(?=\s|$)', text)) >= 2
        if kind == 'web_boilerplate' or (structural and density >= .65 and len(text) <= 1400 and not prose):
            audit.append({'attributes': attrs, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'chars': len(text), 'link_density': density,
                          'reason': reason if kind == 'web_boilerplate' else 'structural_link_dense_template'})
            node.decompose()
    return audit


def clean_text(text):
    # Keep numbers, negation, equations and table separators. No model rewrite.
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(text or ''))
    return re.sub(r'\s+', ' ', text).strip()


def heading_kind(heading):
    value = re.sub(r'^PDF version:\s*[^|]+\|\s*', '', str(heading), flags=re.I)
    value = re.sub(r'^\s*(?:\d+[.\s]+|[IVX]+[.\s]+)', '', value)
    value = value.strip(' #*.:').casefold()
    if value in {'references', 'bibliography', 'works cited', 'literature cited', '参考文献'}:
        return 'bibliography'
    if value in {'cookie settings', 'cookie preferences', 'manage cookies',
                 'site navigation', 'share this article', '网站导航'}:
        return 'web_boilerplate'
    return 'content'


def preprocess_document(document):
    result = copy.deepcopy(document)
    audit = []
    for index, section in enumerate(result.get('sections', [])):
        original = str(section.get('text') or '')
        cleaned = clean_text(original)
        kind, reason = classify_section(section.get('heading', ''), cleaned)
        spans, excluded = retrieval_spans(section.get('heading', ''), cleaned)
        # Never renumber sections: existing source/chunk coordinates stay stable.
        # Retain normalized source and select spans instead of stitching prose.
        section['text'] = cleaned
        section['retrieval_spans'] = spans
        audit.append({'section_index': index, 'heading': section.get('heading', ''),
                      'kind': kind, 'eligible': bool(spans),
                      'reason': reason,
                      'retrieval_spans':spans, 'excluded_spans':excluded,
                      'original_text_sha256': hashlib.sha256(original.encode()).hexdigest(),
                      'cleaned_chars': sum(s['end_char']-s['start_char'] for s in spans), 'original_chars': len(original)})
    result['browse_preprocessing'] = {'version': VERSION, 'sections': audit,
        'excluded_sections': sum(not a['eligible'] for a in audit),
        'document_readability_status':'readable_body' if any(a['eligible'] for a in audit) else 'no_readable_body',
        'raw_document_preserved': True}
    return result


def patch_parser(text):
    def once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('preprocessing anchor mismatch: ' + old[:70])
        text = text.replace(old, new)
    once('from bs4 import BeautifulSoup', 'from .browse_preprocess import preprocess_document\nfrom bs4 import BeautifulSoup')
    once('CHUNKER_VERSION = "medical_chunker_v1"', 'CHUNKER_VERSION = "medical_chunker_browse_sections_v1"')
    once('    chunks = chunk_medical_document(\n', '    document = preprocess_document(document)\n    chunks = chunk_medical_document(\n')
    once('"total_chunks": len(chunks),', '"browse_preprocessing": document["browse_preprocessing"],\n            "total_chunks": len(chunks),')
    # Structured HTML tables were previously omitted. Preserve rows without
    # duplicating nested paragraph/list contents, and leave medical lists intact.
    once('"h6", "p", "li"], recursive=True)', '"h6", "p", "li", "tr"], recursive=True)')
    once('        elif not node.find_parent(["p", "li"]):\n            paragraphs.append(text)',
         '        elif node.name == "tr":\n            cells = node.find_all(["th", "td"], recursive=False)\n            if cells:\n                paragraphs.append(" | ".join(_clean(c.get_text(" ", strip=True)) for c in cells))\n        elif not node.find_parent(["p", "li", "tr"]):\n            paragraphs.append(text)')
    return text
