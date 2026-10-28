"""Inference-only dual discovery. Raw abstracts, exact identity, explicit failures."""
import copy
import ipaddress
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit,unquote


def pdf_enabled():
    # Training/default callers must explicitly opt in; local harness opts in.
    from .pdf_policy_v54 import pdf_enabled as configured_pdf_enabled
    return configured_pdf_enabled()


def aliases(row):
    ext=row.get('externalIds') or {}
    pmid=str(row.get('pmid') or ext.get('PubMed') or '').removeprefix('PMID:')
    if not pmid and str(row.get('source_id','')).startswith('PMID:'):pmid=row['source_id'][5:]
    result=[]
    if pmid.isdigit():result.append('PMID:'+pmid)
    doi=str(row.get('doi') or ext.get('DOI') or '').strip().lower()
    doi=re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)','',doi)
    if doi.startswith('10.') and '/' in doi:result.append('DOI:'+doi)
    s2=str(row.get('paperId') or '')
    if re.fullmatch(r'[0-9a-fA-F]{40}',s2):result.append('S2:'+s2.lower())
    return result


def native_identity_group(row):
    """Project only recognized identifiers from one native provider row."""
    group = {}
    pmid = str(row.get('pmid') or '').removeprefix('PMID:')
    if pmid.isdigit():group['pmid'] = pmid
    paper = str(row.get('paperId') or '')
    if re.fullmatch(r'[0-9a-fA-F]{40}', paper):group['paperId'] = paper.lower()
    def doi(value):
        value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '',
                       str(value or '').strip().lower())
        return value if re.fullmatch(r'10\.[^\s]+/[^\s]+', value) else None
    if normalized := doi(row.get('doi')):group['doi'] = normalized
    ext = row.get('externalIds') or {}
    if not isinstance(ext, dict):raise ValueError('invalid_native_paper_identity')
    chosen = {}
    external_pmid = str(ext.get('PubMed') or '').removeprefix('PMID:')
    if external_pmid.isdigit():chosen['PubMed'] = external_pmid
    if normalized := doi(ext.get('DOI')):chosen['DOI'] = normalized
    if chosen:group['externalIds'] = chosen
    if (group.get('pmid') and chosen.get('PubMed') and group['pmid'] != chosen['PubMed']
            or group.get('doi') and chosen.get('DOI') and group['doi'] != chosen['DOI']):
        raise ValueError('conflicting_native_paper_identity')
    return group


def restored_native_groups(row, base_keys):
    """Validate the generated identifier graph, rooted in this canonical row."""
    groups = row.get('native_identity_groups', [])
    if not isinstance(groups, list):raise ValueError('invalid_restored_native_identity_groups')
    prepared = []
    for group in groups:
        if (not isinstance(group, dict) or not group
                or set(group)-{'pmid','paperId','doi','externalIds'}
                or group != native_identity_group(group) or not aliases(group)):
            raise ValueError('invalid_restored_native_identity_group')
        if group not in prepared:prepared.append(copy.deepcopy(group))
    connected = list(base_keys)
    pending = list(prepared)
    while pending:
        remaining = []
        for group in pending:
            keys = aliases(group)
            if set(keys) & set(connected):
                connected += [key for key in keys if key not in connected]
            else:remaining.append(group)
        if len(remaining) == len(pending):
            raise ValueError('disconnected_restored_native_identity_group')
        pending = remaining
    return prepared, connected


class Registry:
    def __init__(self):self.rows={};self.by_alias={}
    def _bind_aliases(self, sid, keys):
        """Join native-ID components without undoing an earlier merge."""
        canonical = self.by_alias.get(sid, sid)
        joined = {self.by_alias[key] for key in keys if key in self.by_alias}
        for key, target in list(self.by_alias.items()):
            if target in joined:self.by_alias[key] = canonical
        for key in keys:self.by_alias[key] = canonical
    def restore(self, records):
        """Restore verified Search rows without reselecting canonical IDs.

        A canonical S2 ID can legitimately predate a later discovered PMID.
        Keep returned IDs readable while rebuilding native-identifier merges.
        Older replayed rows may extend a known component, never split it or
        roll its canonical mapping back. Claimed alias lists are not trusted.
        """
        prepared = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('invalid_restored_paper_record')
            row = copy.deepcopy(record)
            sid = row.get('source_id')
            if not isinstance(sid, str) or not re.fullmatch(r'(?:PMID:[0-9]+|S2:[a-f0-9]{40})', sid):
                raise ValueError('invalid_restored_paper_source_id')
            if row.get('externalIds') is not None and not isinstance(row['externalIds'], dict):
                raise ValueError('invalid_restored_paper_identity')
            keys = aliases(row)
            if sid not in keys:
                raise ValueError('restored_paper_source_identity_mismatch')
            native_groups, keys = restored_native_groups(row, keys)
            claimed = row.get('identity_aliases', [])
            if not isinstance(claimed, list) or any(not isinstance(a, str) or not re.fullmatch(
                    r'(?:PMID:[0-9]+|S2:[a-f0-9]{40}|DOI:10\.[^\s]+/[^\s]+)', a) for a in claimed):
                raise ValueError('invalid_restored_paper_aliases')
            pdf = row.get('openAccessPdf')
            if pdf is not None and not isinstance(pdf, dict):
                raise ValueError('invalid_restored_paper_pdf')
            for url in (row.get('url'), (pdf or {}).get('url')):
                if not url:
                    continue
                if not isinstance(url, str):
                    raise ValueError('invalid_restored_paper_url')
                parsed = urlsplit(url)
                host = (parsed.hostname or '').lower().rstrip('.')
                if (parsed.scheme not in ('http','https') or not host or parsed.username or parsed.password
                        or host == 'localhost' or host.endswith(('.local','.localhost'))):
                    raise ValueError('unsafe_restored_paper_url')
                try:address = ipaddress.ip_address(host)
                except ValueError:address = None
                if address is not None and (address.is_private or address.is_loopback or address.is_link_local
                        or address.is_multicast or address.is_reserved or address.is_unspecified):
                    raise ValueError('unsafe_restored_paper_url')
            prepared.append((sid, row, keys, native_groups))
        # Validate the complete batch before changing either registry.
        for sid, row, keys, native_groups in prepared:
            prior = copy.deepcopy(self.rows.get(sid, {}))
            for key, value in row.items():
                if value not in (None,'',[]):prior.setdefault(key, value)
            # Match fuse: native fields keep their first useful value, while
            # per-discovery provenance follows the latest completed Search.
            for key in ('identity_aliases','discovery_channels','source_ranks','rrf_score'):
                if key in row:prior[key] = copy.deepcopy(row[key])
            if 'native_identity_groups' in row:
                accumulated = copy.deepcopy(prior.get('native_identity_groups', []))
                for group in native_groups:
                    if group not in accumulated:accumulated.append(group)
                prior['native_identity_groups'] = accumulated
            self.rows[sid] = prior
            # The row's returned ID identifies the canonical selected by the
            # original fuse. Resolve it first: a cached pre-merge S2 row must
            # not undo a newer S2 -> PMID relation. Native co-occurring IDs
            # then join complete alias components, including aliases absent
            # from this particular public row. Existing rows stay readable.
            self._bind_aliases(sid, keys)
        return len(prepared)
    def fuse(self,results,limit=10):
        entries=[]
        for channel,data in results.items():
            for rank,r in enumerate(data,1):
                keys=aliases(r)
                if keys:entries.append((channel,rank,copy.deepcopy(r),keys))
        # Connected components on verified external identifiers, never fuzzy titles.
        groups=[]
        for entry in entries:
            hits=[g for g in groups if set(entry[3]) & {a for e in g for a in e[3]}]
            group=[entry]
            for g in hits:group+=g;groups.remove(g)
            groups.append(group)
        ranked=[]
        for group in groups:
            keys=list(dict.fromkeys(a for e in group for a in e[3]))
            previous=[self.by_alias[a] for a in keys if a in self.by_alias]
            sid=previous[0] if previous else next((a for a in keys if a.startswith('PMID:')),next((a for a in keys if a.startswith('S2:')),None))
            if sid is None:continue
            row=copy.deepcopy(self.rows.get(sid,{}))
            # Retain each provider's ID co-occurrence rather than relying on
            # the first paperId/DOI retained in the merged display row. Only
            # our own generated groups are carried forward from stored rows;
            # a provider-supplied same-named metadata field is ignored.
            native_groups = []
            for previous_id in dict.fromkeys([sid] + previous):
                existing = self.rows.get(previous_id, {})
                prior_groups = existing.get('native_identity_groups')
                if prior_groups is None:
                    prior_groups = [native_identity_group(existing)]
                for item in prior_groups:
                    if item and item not in native_groups:native_groups.append(copy.deepcopy(item))
            for _, _, data, _ in group:
                item = native_identity_group(data)
                if item and item not in native_groups:native_groups.append(item)
            channel_ranks={}
            for channel,rank,data,_ in sorted(group,key=lambda e:e[0]!='pubmed'):
                for k,v in data.items():
                    if v not in (None,'',[]):row.setdefault(k,v)
                channel_ranks[channel]=min(rank,channel_ranks.get(channel,rank))
            self._bind_aliases(sid, keys)
            row.update(source_id=sid,identity_aliases=keys,discovery_channels=sorted(channel_ranks),
                       source_ranks=channel_ranks,rrf_score=sum(1/(60+r) for r in channel_ranks.values()),
                       native_identity_groups=native_groups)
            row['pmid']=next((a[5:] for a in keys if a.startswith('PMID:')),row.get('pmid'))
            self.rows[sid]=row
            ranked.append(row)
        return sorted(ranked,key=lambda r:(-r['rrf_score'],r['source_id']))[:limit]


def parallel_search(calls,registry,query,limit=10,*,original_question=None,rerank_backend=None):
    def run(fn):
        start=time.monotonic()
        try:
            response=fn()
            if response.get('failed') or response.get('error'):raise ValueError('backend_failure')
            if not isinstance(response.get('data'),list):raise ValueError('invalid_response_shape')
            return response['data'],dict(status='success',rows=len(response['data']),seconds=time.monotonic()-start)
        except Exception as exc:
            status=getattr(exc,'status_code',None) or getattr(getattr(exc,'response',None),'status_code',None)
            # Classify typed transport failures, never exception text or URLs.
            from requests.exceptions import Timeout, ConnectionError
            from .s2_transport import S2RequestError
            temporary = (isinstance(exc, (Timeout, ConnectionError))
                         or status in (408, 429, 500, 502, 503, 504)
                         or isinstance(exc, S2RequestError) and exc.kind == 'transport')
            return [],dict(status='failed',error_type=type(exc).__name__,http_status=status,
                           retryable=temporary,seconds=time.monotonic()-start)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={name:pool.submit(run,fn) for name,fn in calls.items()}
        resolved={name:f.result() for name,f in futures.items()}
    good=sum(meta['status']=='success' for _,meta in resolved.values())
    from .paper_semantic_ranking import rank_paper_search_candidates
    # Rank after identity fusion, before the public limit. Both channels enter
    # the same pool; a second RRF must not overwrite the semantic ordering.
    pool=registry.fuse({k:v[0] for k,v in resolved.items()},sum(len(v[0]) for v in resolved.values()))
    ranked=rank_paper_search_candidates(query,pool,original_question=original_question,backend=rerank_backend)
    result = dict(query=query,data=ranked[:limit],
                failed=good==0,error='all_paper_search_routes_failed' if good==0 else '',
                partial_success=good==1,backend_status={k:v[1] for k,v in resolved.items()},
                search_contract='pubmed_s2_parallel_v1',model_tool_calls=1)
    if good == 0 and resolved and all(meta.get('retryable') is True for _,meta in resolved.values()):
        result.update(failure_type='environment_failure', retryable=True,
                      error_code='retrieval_transport_failure')
    return result


def install(backend):
    from . import pubmed_apis as pubmed
    from .semantic_scholar_apis import search_semantic_scholar_keywords,SemanticScholarSearchQueryParams
    original_loader=backend.load_medical_document
    registry=Registry()
    cache={}
    def search(query,limit=10,offset=0):
        if offset:raise ValueError('dual_search_first_page_only')
        context=backend.decode_anchored_query(query)
        key=(query,limit)
        if key in cache:
            return {**copy.deepcopy(cache[key]),'cache_hit':True}
        result=parallel_search({
            'pubmed':lambda:pubmed.search_pubmed(context.focused_query,limit=limit,original_question=context.original_question or None),
            'semantic_scholar':lambda:search_semantic_scholar_keywords(
                SemanticScholarSearchQueryParams(query=context.focused_query),limit=limit,timeout=12,
                fields='paperId,title,abstract,year,url,externalIds,openAccessPdf,publicationTypes'),
        },registry,context.focused_query,limit,original_question=context.original_question or None)
        # Do not freeze partial outages into a success cache.
        if not result['failed'] and not result['partial_success']:cache[key]=copy.deepcopy(result)
        return result
    def load(source_id):
        sid=source_id.split('#',1)[0]
        row=registry.rows.get(sid)
        if row is None:
            if sid.startswith('PMID:'):row={'pmid':sid[5:]}
            else:raise ValueError('unregistered_paper_source')
        attempts=[]; document=None
        if row.get('pmid'):
            try:document=original_loader('PMID:'+str(row['pmid']))
            except Exception as exc:attempts.append(dict(stage='pmid_load',status='failed',error_type=type(exc).__name__))
        if document is None:
            text=str(row.get('abstract') or '')
            document=dict(title=row.get('title',''),text=text,sections=[dict(heading='Abstract',text=text)],
                metadata=dict(source_id=sid,url=row.get('url',''),content_level='semantic_scholar_abstract',
                              abstract_only=True,full_text_available=False,evidence_scope='abstract_only',
                              fetch_method='semantic_scholar_abstract',publication_types=row.get('publicationTypes',[])))
        from .medical_document_parser import reconcile_document_content
        document=reconcile_document_content(document)
        meta=document['metadata']
        meta['source_id']=sid
        attempts=list(meta.get('fetch_attempts',[]))+attempts
        if meta.get('abstract_only') and pdf_enabled():
            pdf=str((row.get('openAccessPdf') or {}).get('url') or '')
            mineru_ready=bool(os.getenv('MEDGAP_MINERU_BASE_URL') or os.getenv('MINERU_API_TOKEN'))
            if not mineru_ready:
                attempts.append(dict(stage='open_pdf',status='service_unavailable'))
            elif not pdf:
                if row.get('pmid'):
                    try:pdf=pubmed.resolve_pmid_to_open_pdf_url(str(row['pmid'])) or ''
                    except Exception as exc:attempts.append(dict(stage='open_pdf_resolve',status='failed',error_type=type(exc).__name__))
                if not pdf:attempts.append(dict(stage='open_pdf',status='no_open_link'))
            if pdf and mineru_ready:
                try:
                    # Only provider-supplied public links; no login/paywall circumvention.
                    parsed=urlsplit(pdf)
                    if parsed.scheme!='https' or not parsed.hostname or parsed.username:raise ValueError('invalid_public_pdf_url')
                    from .mineru_client import get_mineru_client
                    from .medical_document_parser import parse_medical_markdown
                    md,parse_audit=get_mineru_client().parse_url(pdf)
                    expected_title=row.get('title') or meta.get('title') or document.get('title','')
                    title_words=set(re.findall(r'[a-z]{3,}',expected_title.lower()))
                    front_words=set(re.findall(r'[a-z]{3,}',md[:4000].lower()))
                    if len(title_words)<5 or len(title_words & front_words)/len(title_words)<0.8:
                        raise ValueError('pdf_title_identity_unverified')
                    loaded=parse_medical_markdown(md)
                    if not any(s.get('text','').strip() for s in loaded['sections']):raise ValueError('empty_pdf')
                    document=loaded
                    document['title']=expected_title
                    document['metadata']={**meta,**document['metadata']}
                    version='preprint_repository_pdf' if parsed.hostname in ('researchsquare.com','www.researchsquare.com') else 'unverified_pdf_version'
                    for section in document['sections']:
                        section['heading']='PDF version: '+version+' | '+section.get('heading','')
                    document['metadata'].update(source_id=sid,url=pdf,abstract_only=False,full_text_available=True,
                        evidence_scope='full_text',content_level='open_pdf_full_text',fetch_method='open_pdf_mineru',
                        pdf_parse_audit=parse_audit,pdf_title_overlap_check=True,
                        document_version=version)
                    attempts.append(dict(stage='open_pdf',status='success'))
                except Exception as exc:attempts.append(dict(stage='open_pdf',status='failed',error_type=type(exc).__name__))
        document['metadata'].update(pdf_enabled=pdf_enabled(),fetch_attempts=attempts)
        return document
    backend._dual_search=search
    backend._restore_paper_search_sources=registry.restore
    backend.load_medical_document=load
