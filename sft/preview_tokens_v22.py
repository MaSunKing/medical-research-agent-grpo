"""Native-only Search previews, budgeted with the policy tokenizer."""
import copy, json, re
from tokenizers import Tokenizer

_tokenizer = None
RENDERED_FIELDS=('preview_fragment','preview_tokens','preview_budget_tokens','year','publication_year',
 'source_type','discovery_channels','source_id','title','search_preview','preview_kind','preview_truncated',
 'origin_search_tool','origin_query','origin_query_truncated','publication_types','already_opened',
 'previous_browse_queries','requires_new_query')
def candidate_tokens(rows):
    # Match priority_v5.compact_decision exactly; audit-only fields are not
    # part of the model-visible candidate region measured later.
    visible=[{k:v for k,v in row.items() if k in RENDERED_FIELDS} for row in rows]
    return count(json.dumps(visible,ensure_ascii=False))
def configure(path):
    global _tokenizer
    _tokenizer = Tokenizer.from_file(str(path))

def count(text):
    if _tokenizer is None:
        raise RuntimeError('v22 preview tokenizer must be configured before collection')
    return len(_tokenizer.encode(str(text), add_special_tokens=False).ids)

def exact_prefix(raw, ceiling):
    """Return an unchanged prefix fitting the token budget, never a summary."""
    raw=str(raw or '')
    if not raw or ceiling <= 0:return '', bool(raw)
    if count(raw)<=ceiling:return raw,False
    lo,hi=0,len(raw)
    while lo<hi:
        mid=(lo+hi+1)//2
        if count(raw[:mid])<=ceiling:lo=mid
        else:hi=mid-1
    end=lo
    boundary=raw.rfind(' ',0,end)
    if boundary>0:end=boundary
    return raw[:end],True

def select(raw, query, target, ceiling):
    from candidate_window import terms
    raw = re.sub(r'\s+', ' ', str(raw or '')).strip()
    if not raw: return '', [], False
    if count(raw) <= ceiling: return raw, [[0, len(raw)]], False
    spans=[]; start=0
    for m in re.finditer(r'[.!?。！？](?=\s|$)', raw):
        if re.search(r'\b(?:e\.g|i\.e|vs|al|Dr|Fig)\.$', raw[:m.end()], re.I): continue
        a=start
        while a<m.end() and raw[a].isspace(): a+=1
        spans.append((a,m.end()));start=m.end()
    if raw[start:].strip():
        while raw[start].isspace():start+=1
        spans.append((start,len(raw)))
    q=set(terms(query))
    ranked=sorted(spans,key=lambda s:(-len(q & set(terms(raw[s[0]:s[1]]))),s[0]))
    # Keep original introductory context when it fits, then relevant sentences.
    order=[spans[0]]+[s for s in ranked if s!=spans[0]]
    chosen=[]
    for span in order:
        trial=sorted(chosen+[span]);shown=' '.join(raw[a:b] for a,b in trial)
        if count(shown)<=ceiling:
            chosen=trial
            if count(shown)>=target:break
    if chosen:
        return ' '.join(raw[a:b] for a,b in chosen),[list(s) for s in chosen],False
    # No complete sentence fits; explicit fragment, never a rewritten summary.
    a,b=ranked[0];lo=a;hi=b
    while lo<hi:
        mid=(lo+hi+1)//2
        if count(raw[a:mid])<=ceiling:lo=mid
        else:hi=mid-1
    end=lo
    boundary=raw.rfind(' ',a,end)
    if boundary>a:end=boundary
    return raw[a:end],[[a,end]],True

def bound_previews(rows):
    result=copy.deepcopy(rows)
    share=min(128,1024//max(1,len(rows)))
    for row in result:
        paper=str(row.get('source_id','')).startswith(('PMID:','S2:'))
        ceiling=min(128 if paper else 96,share)
        raw=row.pop('_native_preview_text',None)
        if raw is None:
            raise RuntimeError('v22 native preview provenance missing')
        raw=re.sub(r'\s+',' ',str(raw)).strip()
        query=row.pop('_preview_query','')
        origin=re.sub(r'\s+',' ',str(row.get('origin_query') or '')).strip()
        shown_origin,origin_truncated=exact_prefix(origin,24)
        row.update(origin_query=shown_origin,
                   origin_query_truncated=origin_truncated)
        row['_v22_origin_query']=origin
        shown,spans,fragment=select(raw,query,min(96 if paper else 64,ceiling),ceiling)
        row.update(search_preview=shown,preview_truncated=shown!=raw,
                   preview_selected_spans=spans,preview_fragment=fragment,
                   preview_selection='native_token_budget_v22',preview_budget_tokens=ceiling,
                   preview_tokens=count(shown),preview_offsets_reference='whitespace_normalized_native_text')
        # Kept internally only until the candidate-section budget pass finishes.
        row['_v22_native']=raw;row['_v22_query']=query
    def public():return [{k:v for k,v in x.items() if not k.startswith('_v22')} for x in result]
    # The full query is already present once in search_history and in the
    # immutable receipt.  Compact its repeated per-row copy before sacrificing
    # native evidence previews.
    for cap in (16,8,0):
        if candidate_tokens(public())<=2000:break
        for row in result:
            shown,truncated=exact_prefix(row['_v22_origin_query'],cap)
            row.update(origin_query=shown,origin_query_truncated=truncated)
    # Includes titles, document IDs, origins and audit metadata, not just snippets.
    for cap in (96,80,64,48,32,16,0):
        if candidate_tokens(public())<=2000:break
        for row in result:
            ceiling=min(row['preview_budget_tokens'],cap)
            shown,spans,fragment=select(row['_v22_native'],row['_v22_query'],ceiling,ceiling) if ceiling else ('',[],False)
            row.update(search_preview=shown,preview_selected_spans=spans,preview_fragment=fragment,
                       preview_truncated=shown!=row['_v22_native'],preview_tokens=count(shown),preview_budget_tokens=ceiling)
    result=public()
    if candidate_tokens(result)>2000:
        raise RuntimeError('v22 candidate metadata alone exceeds 2000 tokens; do not silently drop IDs')
    assert sum(count(x['search_preview']) for x in result)<=1024
    return result

def install(collection):
    previous=collection.public_candidates
    def public_candidates(rows):
        out=previous(rows)
        assert len(out)==len(rows)
        for src,dst in zip(rows,out):
            dst['_native_preview_text']=src.get('_native_preview_text','')
            dst['_preview_query']=src.get('_preview_query',src.get('query',''))
        return out
    collection.public_candidates=public_candidates
    collection.bound_public_previews=bound_previews
