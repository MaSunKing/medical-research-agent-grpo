"""Question-agnostic sentence-prefix packing under actual token and char limits."""
import copy,functools,re
from pathlib import Path

@functools.lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    root=next(p for p in Path(__file__).resolve().parents if (p/'models/server_sft/Qwen3-8B-MedEvidence-SFT-Merged/tokenizer.json').is_file())
    return AutoTokenizer.from_pretrained(str(root/'models/server_sft/Qwen3-8B-MedEvidence-SFT-Merged'),local_files_only=True)

def pack(opened,max_docs=6,per_doc_chars=2400,per_doc_tokens=700,count=None):
    if count is None:count=lambda s:len(tokenizer().encode(s,add_special_tokens=False))
    docs={};seen=set()
    for e in reversed(opened or []):
        if (e.get('evidence') or {}).get('failed'):continue
        for c in (e.get('evidence') or {}).get('chunks',[]):
            sid=c.get('source_id');text=c.get('text','')
            if not sid or not text.strip():continue
            doc=sid.split('#')[0];key=(doc,re.sub(r'\s+',' ',text).strip())
            if key in seen:continue
            seen.add(key);docs.setdefault(doc,[]).append(c)
    out=[];audit=[]
    for doc,chunks in list(docs.items())[:max_docs]:
        ends=[[m.end() for m in re.finditer(r'[.!?](?:["\u201d\u2019)]*)(?=\s|$)',c['text'])] for c in chunks]
        lengths=[0]*len(chunks)
        # First reserve a fair share for each ranked chunk. Then redistribute
        # unused capacity round-robin. Never skip text inside a kept prefix.
        for i,c in enumerate(chunks):
            for end in ends[i]:
                if end<=per_doc_chars//len(chunks) and count(c['text'][:end])<=per_doc_tokens//len(chunks):lengths[i]=end
        while True:
            changed=False
            for i,c in enumerate(chunks):
                nxt=next((e for e in ends[i] if e>lengths[i]),None)
                if nxt is None:continue
                proposed=lengths.copy();proposed[i]=nxt
                if sum(proposed)<=per_doc_chars and sum(count(x['text'][:n]) for x,n in zip(chunks,proposed))<=per_doc_tokens:
                    lengths=proposed;changed=True
            if not changed:break
        kept=[]
        for c,n in zip(chunks,lengths):
            audit.append(dict(source_id=c['source_id'],selected=n>0,original_chars=len(c['text']),selected_chars=n,
                selected_tokens=count(c['text'][:n]),chunk_relative_span=[0,n],reason='fair_sentence_prefix' if n else 'no_complete_sentence_fits',ranking_changed=False))
            if n:
                item=copy.deepcopy(c);item['text']=c['text'][:n];item['handoff_relative_span']=[0,n];item['handoff_truncated']=n<len(c['text']);kept.append(item)
        if kept:out.append(dict(source_id=doc,chunks=kept))
    return out,audit
