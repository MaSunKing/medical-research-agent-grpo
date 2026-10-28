"""Deterministic new-question selection; never inspect generated answers."""
import csv
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from eval_common import read, save, digest, sha, require

SEED = 'newbench-20261026-v1'
HB_SHA = 'e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f'
MB_SHA = 'e2c5a3d26aed980b97d7ac92d874205fbc140ca041cb9746174d633e7680219f'

def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC',text).casefold()))

def rank(key):
    return digest([SEED,key])

def collect_exclusions(root):
    """Explicit question registries only; no model/tool dumps or credentials."""
    names={'questions.json','question.json','selected50.json','questions_index.json'}
    sources=[];records={}
    for top in ('incoming','outgoing'):
        import os
        for folder,dirs,files in os.walk(Path(root)/top):
            dirs[:]=[d for d in dirs if d not in {'worker_operations','shared_retrieval','captures','retrieval',
                'model-cache','models','.git','node_modules','multibench_sources_20261026',
                'server_process_benchmark_eval'} and not d.startswith('process_bench2x50_')]
            for name in sorted(files):
                if name.lower() not in names:continue
                p=Path(folder)/name
                if p.stat().st_size>20000000:continue
                value=read(p);found=[]
                def walk(x):
                    if isinstance(x,list):
                        for y in x:walk(y)
                    elif isinstance(x,dict):
                        text=x.get('question') or x.get('body')
                        pid=x.get('question_id') or x.get('prompt_id')
                        if isinstance(text,str):
                            found.append((str(pid or ''),text))
                        elif pid:found.append((str(pid),''))
                        if not text:
                            for k in ('questions','records','selected','items'): 
                                if k in x:walk(x[k])
                walk(value)
                if not found:continue
                rel=p.relative_to(root).as_posix()
                sources.append(dict(path=rel,sha256=sha(p),records=len(found)))
                for pid,text in found:
                    key=(pid,normalized(text))
                    records.setdefault(key,dict(question_id=pid,normalized_text=key[1],source=rel))
    require(sources and len(records)>1000,'known_exclusion_pools_incomplete')
    return dict(scope='all discovered explicit question registries in local incoming/outgoing; not pretraining decontamination',
        sources=sorted(sources,key=lambda s:s['path']),records=[records[k] for k in sorted(records)])

def overlap(text,pid,exclusions):
    n=normalized(text);words=set(n.split())
    for e in exclusions:
        if pid and pid==e['question_id']:return 'id',e['question_id']
        other=e['normalized_text']
        if not other:continue
        if n==other:return 'normalized_exact',e['question_id']
        if len(n)>=80 and min(len(n),len(other))/max(len(n),len(other))>=0.8:
            ow=set(other.split())
            if len(words&ow)/max(1,len(words|ow))>=0.65 and SequenceMatcher(None,n,other,autojunk=False).ratio()>=0.85:
                return 'near_duplicate',e['question_id']
    return None

def english_candidate(text):
    # Reproducible eligibility filter, with selected texts separately inspected.
    words=re.findall(r'[a-z]+',text.casefold())
    common={'the','a','is','are','what','how','and','to','of','in','for','my','i','with','can','you','this','patient'}
    return (len(words)>=12 and sum(c.isascii() for c in text)/max(1,len(text))>=0.97
            and sum(w in common for w in words)/len(words)>=0.12)

def select(sources,exclusions):
    sources=Path(sources)
    require(sha(sources/'healthbench.jsonl')==HB_SHA,'HealthBench_source_changed')
    require(sha(sources/'MedBrowseComp_605.csv')==MB_SHA,'MedBrowseComp_source_changed')
    hb=[json.loads(s) for s in (sources/'healthbench.jsonl').read_text(encoding='utf8').splitlines()]
    rows=[];rejected=[]
    for q in sorted(hb,key=lambda q:rank(q['prompt_id'])):
        if len(q['prompt'])!=1 or q['prompt'][0]['role']!='user':continue
        text=q['prompt'][0]['content']
        if not english_candidate(text):continue
        bad=overlap(text,q['prompt_id'],exclusions)
        if bad:rejected.append(dict(benchmark='healthbench',id=q['prompt_id'],reason=bad));continue
        if any(overlap(text,'',[dict(question_id=x['question_id'],normalized_text=normalized(x['question']))]) for x in rows):continue
        rows.append(dict(index=len(rows)+1,benchmark='healthbench',source_id=q['prompt_id'],
            question_id='hbnew_'+q['prompt_id'],question=text,native_prompt=q['prompt'],rubrics=q['rubrics'],
            cluster=q['prompt_id'],source_row=q,source_reference='official physician rubrics',
            selection='hash-ranked eligible single-turn English subset; no output-based selection'))
        if len(rows)==50:break
    require(len(rows)==50,'not_enough_new_HealthBench_questions')
    with (sources/'MedBrowseComp_605.csv').open(encoding='utf-8-sig',newline='') as f:mb=list(csv.DictReader(f))
    counts=Counter();trials=set()
    for original_index,q in sorted(enumerate(mb),key=lambda v:rank([v[1]['task_name'],v[0],v[1]['prompt']])):
        trial=re.findall(r'NCT\d{8}',q['prompt']);require(len(trial)==1,'ambiguous_trial_id')
        if counts[q['task_name']]>=10 or trial[0] in trials:continue
        # Preserve source cell verbatim, including any literal backslash-n.
        text=q['prompt'];pid='mb605-row-'+str(original_index+1)
        bad=overlap(text,pid,exclusions)
        if bad:rejected.append(dict(benchmark='medbrowsecomp',id=pid,reason=bad));continue
        trials.add(trial[0]);counts[q['task_name']]+=1
        rows.append(dict(index=len(rows)+1,benchmark='medbrowsecomp',source_id=pid,
            question_id='mbnew_'+digest(q)[:20],question=text,native_prompt=[dict(role='user',content=text)],
            reference_answer=q['gold'],task_name=q['task_name'],cluster=trial[0],source_row=q,
            rubrics=[dict(criterion='REFERENCE ANSWER (do not infer extra requirements): '+q['gold'],points=1,tags=[])],
            source_reference='https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp',
            selection='605 split; hash-ranked; 10 per task; one question per NCT trial'))
        if len(rows)==100:break
    require(len(rows)==100 and set(counts.values())=={10} and len(trials)==50,'MedBrowseComp_strata_incomplete')
    return rows,dict(seed=SEED,rejected_overlaps=rejected,medbrowse_task_counts=dict(counts),
        medbrowse_independent_trial_clusters=len(trials),question_ids=[q['question_id'] for q in rows],
        no_outputs_examined=True,single_turn_only=True,pretraining_contamination_unknown=True,
        known_local_overlap_check='id + NFKC/case/punctuation/whitespace exact + lexical near duplicate',
        semantic_paraphrase_disjointness_not_guaranteed=True)

def validate_rows(rows,exclusions):
    require(len(rows)==100 and [q['index'] for q in rows]==list(range(1,101)),'100_ordered_questions_required')
    require(Counter(q['benchmark'] for q in rows)=={'healthbench':50,'medbrowsecomp':50},'benchmark_counts_changed')
    require(len({q['question_id'] for q in rows})==100,'duplicate_question_ids')
    require(len({normalized(q['question']) for q in rows})==100,'duplicate_question_texts')
    require(len({q['cluster'] for q in rows if q['benchmark']=='medbrowsecomp'})==50,'duplicate_trial_clusters')
    for q in rows:
        require(q['native_prompt']==[dict(role='user',content=q['question'])],'question_projection_changed')
        require(not overlap(q['question'],q['source_id'],exclusions),'known_training_or_evaluation_overlap')
        if q['benchmark']=='healthbench':
            require(q['question']==q['source_row']['prompt'][0]['content'] and q['rubrics']==q['source_row']['rubrics'],
                'native_HealthBench_question_or_rubrics_changed')
        else:
            require(q['question']==q['source_row']['prompt'] and q['reference_answer']==q['source_row']['gold'],
                'native_MedBrowseComp_question_or_gold_changed')
        require(all(type(r['points']) in (int,float) for r in q['rubrics']) and
                any(r['points']>0 for r in q['rubrics']),'weighted_rubrics_required')
    return True
