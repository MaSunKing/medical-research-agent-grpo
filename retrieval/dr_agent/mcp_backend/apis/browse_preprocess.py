# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Conservative, query-independent preprocessing of parsed Browse sections."""
import copy
import hashlib
import re

VERSION = 'browse_sections_metadata_structures_v8'

# Combine template purpose with text shape, never site/domain or topic rules.
TEMPLATE_HEADING = re.compile(r'导航|热门产品|热门推荐|更多推荐|扫码|扫一扫|分享|技术支持|版权|友情链接|社区|活动|圈层|开发者|'
    r'\b(?:navigation|related|recommended|share|cookie|support|products|community|newsletter|copyright)\b', re.I)
LEGAL = re.compile(r'copyright|版权所有|all rights reserved|ICP备|公网安备|registered trademark|licensed\s+(?:under\s+)?CC[- ]BY', re.I)
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

# These recognize publication fields, not clinical dates mentioned in prose.
PUBLICATION_LABEL = r'(?:publication date|page last (?:updated|reviewed(?:/updated)?)|published online|article history|date of publication|last reviewed)'
PUBLICATION_FIELD = re.compile(r'\b'+PUBLICATION_LABEL+r'\s*(?::|(?=[A-Z0-9]))', re.I)
MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
FIELD_DATE = re.compile(r'\b'+PUBLICATION_LABEL+r'\s*:?\s*(?:'+MONTH+r'\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+'+MONTH+r'\s+\d{4})', re.I)
FACTUAL_PROSE = re.compile(r'\b(?:patients?\s+(?:should|must|require|received|underwent)|'
    r'should|must|advised|contraindicated|avoid|offer|consider|'
    r'(?:is|are)\s+(?:not\s+)?(?:recommended|required|indicated|preferred|effective|ineffective|first[- ]line)|'
    r'(?:we|clinicians?)\s+recommend|(?:guidelines?|panel)\s+recommend(?:s|ed)?|'
    r'(?:study|trial)\s+(?:enrolled|included|found|showed)|'
    r'(?:was|were)\s+(?:measured|randomized|observed)|'
    r'(?:recommend|recommended)\s+(?:testing|monitoring|treatment)|'
    r'\d+(?:\.\d+)?\s*%\s*(?:CI|confidence))\b', re.I)
INDEX_HEADING = re.compile(r'^(?:clinical guidelines|guidelines(?: directory| index)?|'
    r'guidelines in development|guideline development process|site directory|'
    r'clinical guidance index|指南目录)\s*$', re.I)


def publication_shape(heading, text):
    """Require format cues, not disease words or a site-specific allow/blocklist."""
    # Tables and clinical assertions outrank template cues. A resource heading
    # alone is not enough: require a catalog of repeated document-format labels.
    protected = bool(FACTUAL_PROSE.search(text) or re.search(r'\||<table\b|\b(?:reduced|increased|decreased|improved|associated with|statistically significant)\b', text, re.I))
    # Controlled-vocabulary descriptor lists can lose their heading during
    # HTML parsing. Identify repeated descriptor syntax, not descriptor values.
    if (len(text) <= 1200 and not protected
            and re.match(r'^D\d{6,9}\s*[-–]', text)
            and len(re.findall(r'\bD\d{6,9}\s*[-–]', text)) >= 3
            and not re.search(r'[.!?](?:\s|$)', text)):
        return 'bibliography', 'controlled_vocabulary_descriptor_list'
    resource_heading = re.fullmatch(r'(?:(?:supplemental|additional|related)\s+)?(?:implementation\s+)?(?:resources|downloads|resource library)', str(heading).strip(' #*.'), re.I)
    resource_labels = re.findall(r'\b(?:quick[- ]reference guide|slide set|checklist|podcast|download|patient information|fact sheet|video|toolkit)\b', text, re.I)
    if resource_heading and len(resource_labels) >= 3 and len(text) <= 4000 and not protected:
        return 'navigation_index', 'resource_format_catalog_without_clinical_assertions'
    if len(text) <= 400 and not protected and re.fullmatch(r'(?:design(?:ed)?\s+(?:and\s+)?(?:created\s+)?|site\s+(?:designed|developed)\s+)by\s+[^.!?]+[.]?', text, re.I):
        return 'web_boilerplate', 'standalone_site_production_credit'
    if len(text) <= 500 and not protected and re.match(r'^(?:media|press)\s+contact\s*:', text, re.I) and re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\+?\d[\d ()-]{7,}\d', text):
        return 'publication_metadata', 'standalone_media_contact'
    if len(PUBLICATION_FIELD.findall(text)) >= 2 and len(text) <= 1200 and not FACTUAL_PROSE.search(text):
        return 'publication_metadata', 'publication_field_shell'
    if re.fullmatch(r'PUBLISHED\s+\d{1,2}/\d{1,2}/\d{2,4}\s+BY\s+[^.!?]+', text, re.I):
        return 'publication_metadata', 'publication_byline'
    if re.search(r'\bResearch output\s*:\s*Contribution to journal\b', text, re.I) and not FACTUAL_PROSE.search(text):
        return 'publication_metadata', 'repository_publication_record'
    if str(heading).strip().casefold() in {'document overview', 'citation', 'suggested citation', 'how to cite'}:
        signatures = len(re.findall(r'\bdoi\b|\bPMID\s*:|https?://(?:dx\.)?doi\.org/', text, re.I))
        if signatures >= 2 and not FACTUAL_PROSE.search(text):
            return 'bibliography', 'overview_citation_records'
    has_guidelines_heading = bool(re.fullmatch(r'(?:[\w-]+\s+){0,4}guidelines(?:\s*[-:]\s*(?:endorsed|directory|index))?|指南目录', str(heading).strip(' #*.'), re.I))
    sentences = len(re.findall(r'[。！？!?]|(?<!\d)\.(?=\s|$)', text))
    table_markup = bool(re.search(r'\||<table\b', text, re.I))
    if has_guidelines_heading and not table_markup and sentences == 0 and len(text.split()) >= 4 and not FACTUAL_PROSE.search(text):
        return 'navigation_index', 'guideline_heading_and_topic_list'
    navigation_cues = len(re.findall(r'\b(?:guidelines in development|sections will be released|'
                                    r'expected in|read the (?:guideline|joint statement)|view key messages)\b', text, re.I))
    index_context = INDEX_HEADING.fullmatch(str(heading).strip(' #*.')) or (
        re.search(r'\bguideline development process\b', text, re.I)
        and re.search(r'\blearn more\b', text, re.I)) or (has_guidelines_heading and navigation_cues >= 2)
    if index_context and not table_markup and not FACTUAL_PROSE.search(text):
        if re.search(r'\b(?:access .*resources|guidelines in development|guideline development process|'
                     r'learn more|read (?:the |in )|view .*guidelines|expected in|sections will be released|'
                     r'these include guidelines)\b', text, re.I):
            return 'navigation_index', 'guideline_directory_without_recommendations'
    return None


def unresolved_placeholders(text):
    protected = [m.span() for m in CODE.finditer(text)]
    return [m for m in PLACEHOLDER.finditer(text)
            if not any(a <= m.start() < b for a,b in protected)]


def classify_section(heading, text):
    text = clean_text(text)
    if heading_kind(heading) == 'publication_metadata':
        return 'publication_metadata', 'explicit_publication_backmatter_heading'
    if heading_kind(heading) == 'bibliography' and str(heading).strip().casefold() in {'mesh terms', 'mesh keywords', 'index terms', 'subject headings'}:
        return 'bibliography', 'publication_index_terms'
    shape = publication_shape(heading, text)
    if shape:
        return shape
    if len(text) <= 300 and re.fullmatch(r'(?:Correspondence to\s+[^!?]+|Reprints and permissions)\.?', text, re.I):
        return 'publication_metadata', 'standalone_publication_contact_or_permissions'
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
    if heading_kind(heading) == 'bibliography' and REFERENCE_RECORD.search(text) and not FACTUAL_PROSE.search(text):
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
    # Remove bounded fields from a mixed body section without joining the prose
    # on either side. Each returned span retains the normalized-source offsets.
    for match in FIELD_DATE.finditer(text):
        if not any(a <= match.start() < b for a,b in protected):
            exclusions.append((match.start(), match.end(), 'bounded_publication_date_field'))
    # Readers sometimes serialize HTML head metadata as Markdown front matter.
    # Exclude only the bounded metadata block, not the medical prose it repeats.
    front = re.match(r'^---\s+(.*?)\s+---(?=\s|$)', text, re.S)
    if front and re.search(r'\bmeta-(?:dcterms[.:]|og:|twitter:|description:|viewport:)', front.group(1), re.I):
        exclusions.append((front.start(), front.end(), 'serialized_head_metadata'))
    # A standalone tracking iframe converted to an (occasionally broken) link
    # is not article content. Avoid deleting prose discussing tracking tools.
    tracker = re.compile(r'\[https?://(?:www\.)?googletagmanager\.com/ns\.html\?[^\s\]]*(?:\])?', re.I)
    for m in tracker.finditer(text):
        if not any(a <= m.start() < b for a,b in protected):
            exclusions.append((m.start(),m.end(),'tracking_iframe_link'))
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
    # MinerU emits HTML tables inside Markdown. Preserve their original spans,
    # so ordinary sentence clipping cannot leak a partial table as "prose".
    structured=[]
    table_excluded=[]
    for span in spans:
        cursor=span['start_char']; limit=span['end_char']
        while cursor < limit:
            opening=re.search(r'<table\b[^>]*>',text[cursor:limit],re.I)
            if not opening:
                structured.append({'start_char':cursor,'end_char':limit,'kind':'body'})
                break
            start=cursor+opening.start()
            if start>cursor and text[cursor:start].strip():
                structured.append({'start_char':cursor,'end_char':start,'kind':'body'})
            closing=re.search(r'</table\s*>',text[cursor+opening.end():limit],re.I)
            if not closing:
                table_excluded.append({'start_char':start,'end_char':limit,'kind':'non_body',
                                       'reason':'unclosed_html_table'})
                break
            end=cursor+opening.end()+closing.end()
            structured.append({'start_char':start,'end_char':end,'kind':'table'})
            cursor=end
    return structured, ([{'start_char':a,'end_char':b,'kind':'non_body','reason':why} for a,b,why in merged]
                        + table_excluded)


def strip_html_templates(soup):
    """Query-independent structural cleanup; retain an exclusion audit."""
    audit = []
    hints = re.compile(r'(?:^|[-_\s])(?:nav|navbar|footer|sidebar|breadcrumb|related|recommendations?|share|social|cookie|copyright|product-list)(?:$|[-_\s])', re.I)
    for node in list(soup.find_all(['aside', 'div', 'section', 'ul'])):
        # A parent may have been decomposed earlier in this snapshot. BeautifulSoup
        # can retain a descendant's name while clearing its attributes.
        if node.name is None or node.attrs is None:
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
    if value in {'reference', 'references', 'reference list', 'list of references',
                 'bibliographic references', 'bibliography', 'works cited', 'literature cited', '参考文献',
                 'mesh terms', 'mesh keywords', 'index terms', 'subject headings'}:
        return 'bibliography'
    if value in {'author information', 'correspondence', 'corresponding author',
                 'reprints and permissions', 'rights and permissions', 'copyright and permissions',
                 'funding', 'funding information', 'funding sources', 'acknowledgements',
                 'acknowledgement', 'acknowledgments', 'acknowledgment', '作者信息', '通讯作者', '资助信息',
                 'publication date', 'article history', 'publication history'}:
        return 'publication_metadata'
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
        if kind == 'content' and section.get('structure_kind') in {'table', 'table_like'}:
            # Parser ownership is authoritative. Do not reclassify a table as
            # prose merely because one cell contains full sentences.
            spans = [{'start_char':0,'end_char':len(cleaned),'kind':'table'}] if cleaned else []
            excluded = []
            if section.get('table_structure_complete') is False:
                spans = []
                excluded = [{'start_char':0,'end_char':len(cleaned),'kind':'non_body',
                             'reason':'source_table_header_or_data_unverified'}]
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
        'coverage_verified': False,
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
