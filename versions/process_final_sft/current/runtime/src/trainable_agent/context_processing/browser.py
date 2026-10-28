"""Conservative, query-independent preprocessing of parsed Browse sections."""
import copy
import hashlib
import re

VERSION = 'browse_sections_bounded_nonbody_v18'

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


# Complete access-control sentences only, never topic-based sentence deletion.
ACCESS_NOTICE = re.compile(
    r'(?:This is a preview of subscription content[,;.]?\s*)?'
    r'(?:log in|sign in) via an institution to (?:check|get) access[.!]?(?=\s|$)', re.I)
BARE_DATE = re.compile(r'(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{4})')


def short_body_content_protected(text):
    """Protect claim/list/table shapes before the two weak heading filters.

    This is a conservative preservation test, not a relevance or medical-truth
    classifier. A topic, a bare number, or punctuation alone is not sufficient.
    Explicit access/UI/catalogue rules retain their earlier precedence.
    """
    table = re.search(r'<table\b|\|\s*:?-{2,}:?\s*\||\|\s*\d+(?:\.\d+)?%?\s*\|', text, re.I)
    measurement = re.search(
        r'(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|[µμ]g|g/kg|mL|ml|L/min|mmHg|mmol/L|'
        r'mmol|mEq|bpm|cmH2O|毫克|微克|毫升|毫米汞柱)(?![A-Za-z])|'
        r'\b(?:p|n)\s*[<=>≤≥]\s*\d|[<>≤≥]\s*\d', text, re.I)
    # Quantified targets/rates and care intervals can be complete clinical
    # instructions without a finite verb or terminal punctuation. Require a
    # measurement/plan cue as well as a value, so percentages and dates in a
    # menu, promotion or release schedule are not blanket exemptions.
    percentage = re.search(r'(?<![\d.])\d+(?:[.,]\d+)?\s*(?:[%％]|per\s+cent|percent(?:age)?)', text, re.I)
    percentage_context = re.search(
        r'\b(?:saturation|spo2|oxygenation|fraction|risk|mortality|incidence|'
        r'prevalence|efficacy|sensitivity|specificity|response\s+rate|survival)\b|'
        r'饱和度|血氧|发生率|死亡率|生存率|风险', text, re.I)
    interval = re.search(
        r'(?<!\w)\d+(?:[.,]\d+)?\s*(?:[-–—]\s*)?(?:hours?|days?|weeks?|months?|years?)\b|'
        r'\d+(?:[.,]\d+)?\s*(?:小时|天|周|个月|月|年)', text, re.I)
    care_plan = re.search(
        r'\b(?:follow[-\s]?up|reassess(?:ment)?|re[-\s]?evaluat(?:e|ion)|'
        r'monitor(?:ing)?|retest|recheck|check[-\s]?up|repeat\s+(?:test|assessment|measurement)|'
        r'review\s+(?:treatment|therapy|symptoms))\b|随访|复查|复评|复诊|监测', text, re.I)
    assertion = re.search(
        r'\b(?:no|not|without|versus|vs|compared|should|must|advised|recommended|'
        r'is|are|was|were|has|have|had|requires?|indicates?|shows?|showed|found|'
        r'reduces?|reduced|increases?|increased|decreases?|decreased|improves?|improved|'
        r'persists?|remains?|associated|contraindicated|avoid|administer)\b', text, re.I)
    chinese_statement = re.search(
        r'应当|应予|必须|不应|不宜|禁用|慎用|未见|未发现|无明显|'
        r'不(?:推荐|建议)|(?:建议|推荐)(?:使用|给予|进行|监测|随访)|'
        r'需(?:要)?(?:监测|随访|治疗)|(?:患者|受试者).{0,20}(?:应|需|接受|使用|出现)|'
        r'(?:监测|评估|记录)(?:症状|血压|血氧|血糖|疗效|不良反应|用药)|'
        r'(?:降低|增加|改善|减少|升高|下降).{0,16}(?:风险|死亡|症状|发生率|血压|血氧)', text)
    return bool(table or FACTUAL_PROSE.search(text) or measurement
                or (percentage and percentage_context) or (interval and care_plan)
                or assertion or chinese_statement)


def high_confidence_nonbody_shape(heading, text):
    """Shared pre-ranking / pre-registration rule for complete non-body blocks.

    Unknown text stays available. These rules do not score relevance, infer a
    medical claim, or remove a figure caption simply because it is short.
    """
    value = clean_text(text)
    # A publisher format chooser may contain one pipe without being a table.
    # Every other table separator and every HTML table remains protected.
    without_format_chooser = re.sub(r'\b(?:power\s*point|pptx?)\s*\|\s*pdf\b', '', value, flags=re.I)
    if re.search(r'\||<table\b', without_format_chooser, re.I):
        return None
    # Publisher identity catalogues are not article body. Require repeated
    # identity/affiliation structure, not a clinical topic or institution name.
    # Mixed claims, measurements and dates remain untouched as uncertain text.
    # A capitalized connector inside an institution's proper name is not by
    # itself a clinical comparison. Other claims/measurements remain protected.
    catalogue_probe = re.sub(r'(?i:(\b(?:centre|center|institute|department|school|division)\b))'
                             r'([^,.;]{0,90})\bVersus\b', r'\1\2NameConnector', value)
    substantive = (FACTUAL_PROSE.search(value) or re.search(
        r'\b(?:patients?|participants?|outcomes?|results?|conclusions?|effects?|efficacy|'
        r'safety|mortality|adverse|benefit|risk|increased|decreased|reduced|improved|associated|shows?|'
        r'randomi[sz]\w*|treated|treatment|therapy|'
        r'no|not|without|versus|vs|groups?|comparison|comparator|is|are|was|were|has|have|had|may|can|should|must)\b|'
        r'\b(?:19|20)\d{2}\b|\d\s*(?:%|mg\b|mmol\b)|\bn\s*=|\bp\s*[<=>]|[<>≤≥]', catalogue_probe, re.I))
    if not substantive:
        orcids = re.findall(r'\b(?:https?://)?orcid\.org/\d{4}-\d{4}-\d{4}-[\dX]{4}\b', value, re.I)
        if len(orcids) >= 3 and len(value) <= 15000:
            return 'bibliography', 'repeated_author_orcid_catalog_without_body'
        institutions = re.findall(r'\b(?:departments?|universit(?:y|ies)|institutes?|schools?|'
                                  r'hospitals?|facult(?:y|ies)|cent(?:er|re)s?|divisions?)\b', value, re.I)
        ordinals = re.findall(r'\b\d{1,3}\s+(?=[A-Za-zÀ-ž])', value)
        if len(institutions) >= 6 and len(ordinals) >= 4 and value.count(',') >= 8 and len(value) <= 15000:
            return 'bibliography', 'numbered_author_affiliation_catalog_without_body'
    # Checkout messages form a complete, explicitly administrative phrase run.
    # No arbitrary suffix/prefix (including a clinical claim) is swallowed.
    checkout_part = (r'(?:Tax calculation will be finali[sz]ed at checkout|'
                     r'Purchases are for personal use only|Institutional subscriptions)')
    if (len(value) <= 700 and len(re.findall(checkout_part, value, re.I)) >= 2
            and re.fullmatch(r'(?:' + checkout_part + r'[.!;,]?\s*)+', value, re.I)):
        return 'web_boilerplate', 'complete_checkout_subscription_controls'

    # A number, negative/comparative result, instruction or finite assertion
    # defeats the following catalogue/title rules even under a resource heading.
    protected = (FACTUAL_PROSE.search(value) or re.search(
        r'\d|\b(?:no|not|without|versus|vs|increased|decreased|reduced|improved|'
        r'associated|risk ratio|odds ratio|was|were|is|are|has|have|had|'
        r'causes?|caused|reduces?|increases?|prevents?|prevented|recommended|'
        r'recommend|requires?|indicates?|shows?|showed|found|lasts?|follow|'
        r'administer|avoid|contraindicated)\b|[<>≤≥]', value, re.I))
    if protected:
        return None
    labels = re.findall(r'\b(?:teaching slides|(?:educational )?slide sets?|'
                        r'infographics?|development process|resources|downloads?|'
                        r'power\s*point|pdf)\b', value, re.I)
    format_chooser = re.search(r'\b(?:power\s*point|pptx?)\s*\|\s*pdf\b', value, re.I)
    navigation = re.search(r'\b(?:back to|view all|download|resources)\b', value, re.I)
    # Multiple distinct resource forms plus a format chooser and navigation
    # disambiguate a link catalogue from substantive educational content.
    if len(value) <= 2500 and len(set(x.casefold() for x in labels)) >= 4 and format_chooser and navigation:
        return 'navigation_index', 'resource_format_chooser_and_navigation_catalog'

    figure_heading = re.fullmatch(r'(?:figure|fig\.?|image|illustration)\s+[A-Za-z]?\d+[A-Za-z]?', str(heading).strip(' #.: '), re.I)
    # Only an identified figure, a noun-phrase diagram title and attribution.
    # Actual captions with results, steps, numbers, dates or uncertain shape stay.
    bare_title = re.fullmatch(
        r'[\w ()/,-]{1,100}\b(?:life cycle|flow chart|flow diagram|schematic diagram|overview diagram)'
        r'\.\s*(?:Source|Credit):\s*[\w &.,()/-]{1,80}', value, re.I)
    if figure_heading and bare_title:
        return 'publication_metadata', 'figure_title_and_attribution_without_readable_figure'
    return None



def administrative_shape(heading, text):
    # Match bounded administrative shapes, never disease/site words.
    table = bool(re.search(r'\||<table\b', text, re.I))
    disclaimer = re.fullmatch(r'Recommendations made by .{1,180} are independent of .{1,180}\. They should not be construed as an official position of .{1,350}\.',text,re.I)
    if table or (FACTUAL_PROSE.search(text) and not disclaimer):
        return None
    if disclaimer:
        return 'web_boilerplate','institutional_position_disclaimer'
    labels=re.findall(r'\b(?:quick reference guide|infographic set|executive summary|central illustration|top \d+ takeaways|guideline suite|translations|ancillary publications|implementation resources|slide set|podcast|toolkit)\b',text,re.I)
    catalog_context=re.search(r'\b(?:resources|downloads|publications|guideline suite|translations)\b',str(heading)+' '+text,re.I)
    claims=re.search(r'\b(?:no|not|without|versus|vs|increased|decreased|reduced|improved|associated|risk ratio|odds ratio)\b|\d\s*%|[<>≤≥]',text,re.I)
    if len(text)<=7000 and catalog_context and len(labels)>=3 and not claims:
        return 'navigation_index','repeated_resource_format_catalog'
    if len(text)<=700 and not claims and re.fullmatch(r'For (?:updated|more|additional) (?:guidance|information), (?:review|visit|see|consult) .{1,350}(?:web page|website|online resource) (?:regularly|for updates)\. (?:Email|Contact) [\w.+-]+@[\w.-]+\.[A-Za-z]{2,} for .{1,220}\.',text,re.I):
        return 'navigation_index','navigation_and_contact_only'
    return None


def publication_shape(heading, text):
    """Require format cues, not disease words or a site-specific allow/blocklist."""
    # High confidence shape rules, never a disease/domain blocklist.
    if BARE_DATE.fullmatch(text) and not re.search(r'\b(?:table|figure|results|outcomes|timeline)\b', str(heading), re.I):
        return 'publication_metadata', 'standalone_date_not_clinical_evidence'
    if re.fullmatch(r'(?i:Technical bulletin|Research article|Original article)(?:\s+[A-Z][A-Z0-9-]{1,9})?', text) and len(text) <= 80:
        return 'publication_metadata', 'standalone_document_type_label'
    if ACCESS_NOTICE.fullmatch(text):
        return 'web_boilerplate', 'standalone_subscription_access_notice'

    if re.fullmatch(r'[-* ]*(?:Review|Open table in a new tab|View full text|Download PDF|Previous article|Next article)[-* ]*', text, re.I):
        return 'web_boilerplate', 'standalone_publisher_control'
    if str(heading).strip(' #*.:').casefold() in {'keywords', 'key words', 'keyword', 'tags', 'index words'} and len(text) < 1200 and not FACTUAL_PROSE.search(text) and not re.search(r'\b(?:no|not|without|versus|vs)\b', text, re.I) and not re.search(r'[.!?](?:\s|$)|<table\b|\d+\s*%', text, re.I):
        return 'bibliography', 'explicit_keyword_list'
    pages = re.findall(r'(?<!\w)S?\d{1,4}\s*\|', text)
    if len(pages) >= 3 and re.search(r'\b(?:contents|tables,? figures|chapter|foreword|summary of recommendation)\b', text, re.I) and not FACTUAL_PROSE.search(text):
        return 'navigation_index', 'repeated_page_number_catalog'
    # Tables and clinical assertions outrank template cues. A resource heading
    # alone is not enough: require a catalog of repeated document-format labels.
    protected = bool(FACTUAL_PROSE.search(text) or re.search(r'\||<table\b|\b(?:reduced|increased|decreased|improved|associated with|statistically significant)\b', text, re.I))
    # Pure resource pointers require explicit access-location syntax, not merely
    # a guideline/topic word. Recommendations, findings and tables are protected.
    pointer = re.search(r'\b(?:additional information .{0,100}can be found online at|'
                        r'(?:is|are) available under the (?:supporting and related materials|downloads?) tab)', text, re.I)
    location = re.search(r'https?://|\b(?:docket|online at|supporting and related materials)\b', text, re.I)
    if len(text) <= 1400 and pointer and location and not protected:
        return 'navigation_index', 'pure_resource_location_without_clinical_claim'

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
    body_protected = short_body_content_protected(text)
    if has_guidelines_heading and not table_markup and sentences == 0 and len(text.split()) >= 4 and not body_protected:
        return 'navigation_index', 'guideline_heading_and_topic_list'
    navigation_cues = len(re.findall(r'\b(?:guidelines in development|sections will be released|'
                                    r'expected in|read the (?:guideline|joint statement)|view key messages)\b', text, re.I))
    index_context = INDEX_HEADING.fullmatch(str(heading).strip(' #*.')) or (
        re.search(r'\bguideline development process\b', text, re.I)
        and re.search(r'\blearn more\b', text, re.I)) or (has_guidelines_heading and navigation_cues >= 2)
    if index_context and not table_markup and not body_protected:
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
    shape = high_confidence_nonbody_shape(heading, text) or administrative_shape(heading, text) or publication_shape(heading, text)
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
    if (TEMPLATE_HEADING.search(str(heading)) and len(text) <= 400 and sentences == 0
            and short_units and not short_body_content_protected(text)):
        return 'web_boilerplate', 'template_heading_and_short_list'
    return 'content', 'insufficient_template_evidence'


def retrieval_spans(heading, text, source_blocks=()):
    """Select contiguous original-text spans; never reconstruct a cleaned claim."""
    text = clean_text(text)
    kind, reason = classify_section(heading, text)
    if kind != 'content':
        return [], [{'start_char':0, 'end_char':len(text), 'kind':kind, 'reason':reason}]
    protected = [m.span() for m in CODE.finditer(text)]
    exclusions = []
    seen_blocks = {}
    for block in source_blocks:
        a,b = block['start_char'],block['end_char']
        if not 0 <= a < b <= len(text):
            raise ValueError('invalid_source_block_coordinates')
        block_text = text[a:b]
        block_kind,block_reason = classify_section(heading, block_text)
        if block_kind != 'content':
            exclusions.append((a,b,'source_block:'+block_reason))
        elif len(block_text) >= 80 and not re.search(r'\||<table\b',block_text,re.I):
            if block_text in seen_blocks:
                exclusions.append((a,b,'exact_repeated_paragraph'))
            else:
                seen_blocks[block_text] = (a,b)
    # Remove bounded fields from a mixed body section without joining the prose
    # on either side. Each returned span retains the normalized-source offsets.
    for match in ACCESS_NOTICE.finditer(text):
        # Require a paragraph/sentence boundary so prose quoting or discussing
        # an access message is preserved; offsets always refer to source text.
        if (match.start() == 0 or re.search(r'[.!?]\s*$', text[:match.start()])) and not any(a <= match.start() < b for a,b in protected):
            exclusions.append((match.start(), match.end(), 'bounded_subscription_access_notice'))
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
    # Bounded author-role runs; an appendix/body boundary ends the exclusion.
    roles = list(re.finditer(r'\b(?:Statistical expertise|Administrative, technical, or logistic support|Collection and assembly of data)\s*:', text, re.I))
    if len(roles) >= 2:
        end_marker = re.search(r'\b(?:Appendix (?:Figure|Table)|Abstract|Results|Discussion)\b', text[roles[0].start():], re.I)
        end = roles[0].start()+end_marker.start() if end_marker else len(text)
        run = text[roles[0].start():end]
        if not FACTUAL_PROSE.search(run):
            exclusions.append((roles[0].start(), end, 'bounded_contributor_role_catalog'))
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
                 'publication date', 'article history', 'publication history',
                 'author contributions', 'authors contributions', 'contributor information'}:
        return 'publication_metadata'
    if value in {'cookie settings', 'cookie preferences', 'manage cookies',
                 'site navigation', 'share this article', '网站导航'}:
        return 'web_boilerplate'
    return 'content'



def normalized_source_blocks(original, blocks):
    """Rebase validated source boundaries across deterministic normalization."""
    from bisect import bisect_left
    blocks = list(blocks)
    for block in blocks:
        a,b = block['start_char'],block['end_char']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(original):
            raise ValueError('invalid_source_block_coordinates')
    cleaned = clean_text(original)
    if original == cleaned:
        return cleaned, copy.deepcopy(blocks), None
    # Keep the original index of each emitted character. Remove controls first,
    # then collapse whitespace, matching clean_text exactly; never clamp offsets.
    emitted, positions = [], []
    for i,char in enumerate(original):
        if (ord(char) < 32 and char not in '\t\n\r') or ord(char)==127:
            continue
        # clean_text also deletes VT/FF even though Python calls them whitespace.
        if char in '\x0b\x0c':
            continue
        if char.isspace():
            if emitted and emitted[-1] != ' ':
                emitted.append(' ');positions.append(i)
        else:
            emitted.append(char);positions.append(i)
    if emitted and emitted[-1]==' ':
        emitted.pop();positions.pop()
    if ''.join(emitted) != cleaned:
        raise ValueError('source_normalization_mapping_mismatch')
    rebased, removed = [], []
    for block in blocks:
        a,b=bisect_left(positions,block['start_char']),bisect_left(positions,block['end_char'])
        if a < b:
            rebased.append(dict(block,start_char=a,end_char=b))
        else:
            removed.append(copy.deepcopy(block))
    audit=dict(version='source_boundary_mapping_v1',original_text_sha256=hashlib.sha256(original.encode()).hexdigest(),
        normalized_text_sha256=hashlib.sha256(cleaned.encode()).hexdigest(),original_blocks=copy.deepcopy(blocks),
        normalized_blocks=copy.deepcopy(rebased),empty_blocks=removed,original_chars=len(original),normalized_chars=len(cleaned))
    return cleaned, rebased, audit


def preprocess_document(document):
    result = copy.deepcopy(document)
    audit = []
    seen = {}
    duplicate_chars = 0
    for index, section in enumerate(result.get('sections', [])):
        original = str(section.get('text') or '')
        cleaned, blocks, mapping = normalized_source_blocks(original, section.get('source_blocks',()))
        section['source_blocks'] = blocks
        if mapping is not None:
            section['source_block_normalization'] = mapping
        kind, reason = classify_section(section.get('heading', ''), cleaned)
        spans, excluded = retrieval_spans(section.get('heading', ''), cleaned, section.get('source_blocks',()))
        if kind == 'content' and section.get('structure_kind') in {'table', 'table_like'}:
            # Parser ownership is authoritative. Do not reclassify a table as
            # prose merely because one cell contains full sentences.
            spans = [{'start_char':0,'end_char':len(cleaned),'kind':'table'}] if cleaned else []
            excluded = []
            if section.get('table_structure_complete') is False:
                spans = []
                excluded = [{'start_char':0,'end_char':len(cleaned),'kind':'non_body',
                             'reason':'source_table_header_or_data_unverified'}]
        # DataTrove Gopher repetition method adapted to section granularity:
        # exact normalized block counts, not fuzzy similarity or whole-doc drops.
        # Require the same heading/structure, keeping distinct clinical contexts.
        key = (str(section.get('heading','')).casefold(), section.get('structure_kind'), cleaned)
        duplicate_of = seen.get(key)
        if spans and len(cleaned) >= 80 and duplicate_of is not None:
            spans = []
            excluded = [{'start_char':0,'end_char':len(cleaned),'kind':'non_body',
                         'reason':'exact_repeated_section','duplicate_of_section':duplicate_of}]
            duplicate_chars += len(cleaned)
            reason = 'exact_repeated_section'
        elif spans:
            seen[key] = index
        # Never renumber sections: existing source/chunk coordinates stay stable.
        # Retain normalized source and select spans instead of stitching prose.
        section['text'] = cleaned
        section['retrieval_spans'] = spans
        audit.append({'section_index': index, 'heading': section.get('heading', ''),
                      'kind': kind, 'eligible': bool(spans),
                      'reason': reason,
                      'retrieval_spans':spans, 'excluded_spans':excluded,
                      'source_block_normalization':section.get('source_block_normalization'),
                      'original_text_sha256': hashlib.sha256(original.encode()).hexdigest(),
                      'cleaned_chars': sum(s['end_char']-s['start_char'] for s in spans), 'original_chars': len(original)})
    result['browse_preprocessing'] = {'version': VERSION, 'sections': audit,
        'excluded_sections': sum(not a['eligible'] for a in audit),
        'document_readability_status':'readable_body' if any(a['eligible'] for a in audit) else 'no_readable_body',
        'coverage_verified': False,
        'raw_document_preserved': True,
        'exact_duplicate_chars':duplicate_chars,
        'repetition_method':'datatrove_gopher_exact_blocks_adapted_no_document_drop'}
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


# Scoped to one Worker request. No global per-question mutable ledger.
from contextvars import ContextVar
from contextlib import contextmanager
_opened_windows = ContextVar('opened_browse_windows', default=())

@contextmanager
def opened_window_scope(rows):
    token = _opened_windows.set(tuple(dict(r) for r in rows))
    try:
        yield
    finally:
        _opened_windows.reset(token)

def current_opened_windows():
    return _opened_windows.get()

def repeated_window(text, source_id, previous, *, canonical_chunk_id=None):
    """Exact normalized containment in the same source, never fuzzy matching."""
    value = re.sub(r'\s+', ' ', text).strip()
    for row in previous:
        if row.get('source_id') != source_id:
            continue
        old = re.sub(r'\s+', ' ', str(row.get('text') or '')).strip()
        if value and (value == old or (len(value) >= 80 and value in old)):
            return {'reason': 'already_opened_window', 'evidence_id': row.get('evidence_id'),
                    'text_sha256': hashlib.sha256(value.encode()).hexdigest()}
    # A larger returned window may consist entirely of two previously read
    # windows. Single-row containment misses that case. Match verbatim text,
    # never semantic similarity, and retain every witness in the receipt.
    if len(value) < 80:
        return None
    ranges = []
    for row in previous:
        if row.get('source_id') != source_id:
            continue
        parent = re.sub(r'-v[a-f0-9]{16}$', '', str(row.get('chunk_id') or ''))
        if canonical_chunk_id and parent != canonical_chunk_id:
            continue
        old = re.sub(r'\s+', ' ', str(row.get('text') or '')).strip()
        if len(old) < 80:
            continue
        offset = 0
        while (offset := value.find(old, offset)) >= 0:
            ranges.append((offset, offset+len(old), row.get('evidence_id'), row.get('chunk_id')))
            offset += len(old)
    cursor, witnesses, covered = 0, [], []
    for start, end, eid, chunk_id in sorted(ranges, key=lambda r:(r[0],-r[1])):
        if value[cursor:start].strip() and start > cursor:
            return None
        if end <= cursor:
            continue
        covered.append(dict(start=start,end=end,evidence_id=eid,chunk_id=chunk_id))
        if eid and eid not in witnesses:
            witnesses.append(eid)
        cursor = end
        if cursor >= len(value):
            break
    if len(covered) >= 2 and cursor and not value[cursor:].strip():
        return dict(reason='already_opened_union_window', evidence_ids=witnesses,
                    normalized_text_ranges=covered, normalization='whitespace_only',
                    text_sha256=hashlib.sha256(value.encode()).hexdigest())
    return None
