"""Portable JSONL Search/Browse worker; no policy model, Judge or optimizer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from urllib.parse import quote
from paired_receipts import PairedReceiptStore

KEYS={'DASHSCOPE_API_KEY','SERPER_API_KEY','SERPER_API_KEY_FALLBACK','NCBI_API_KEY',
      'NCBI_EMAIL','S2_API_KEY','MINERU_API_TOKEN','MEDGAP_MINERU_BASE_URL'}
# Match the request layer's bounded, ordered fallback slots. These names must
# also pass credential loading and log redaction, not just API key selection.
KEYS.update(f'SERPER_API_KEY_FALLBACK_{index}' for index in range(2,21))
ALIASES={'PUBMED_API_KEY':'NCBI_API_KEY','SEMANTIC_SCHOLAR_API_KEY':'S2_API_KEY',
         'MINERU_API_KEY':'MINERU_API_TOKEN'}

def credentials(path):
    values={k:os.environ[k] for k in KEYS|set(ALIASES) if os.environ.get(k)}
    path=Path(path)
    if path.is_file():
        if os.name=='posix' and path.stat().st_mode & 0o077:
            raise ValueError('secrets file must have mode 600')
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line=line.strip()
            if line.startswith('export '):line=line[7:].lstrip()
            if not line or line.startswith('#') or '=' not in line:continue
            key,raw=line.split('=',1);key=key.strip()
            if key not in KEYS|set(ALIASES):continue
            parts=shlex.split(raw,comments=True)
            if len(parts)!=1:raise ValueError('invalid credential assignment')
            values[key]=parts[0]
    for alias,key in ALIASES.items():
        if values.get(alias):
            if values.get(key) and values[key]!=values[alias]:raise ValueError('conflicting credential aliases')
            values[key]=values[alias]
    return {k:v for k,v in values.items() if k in KEYS and v}

def configure(args):
    # Reuse a complete cached local MedCPT snapshot; never download a model
    # inside a live Search. An explicit path can be supplied on other hosts.
    paper_model=getattr(args,'paper_reranker_model',None)
    if paper_model is None:
        hub=args.reranker_model.parents[2]
        snapshots=hub/'models--ncbi--MedCPT-Cross-Encoder'/'snapshots'
        for path in sorted(snapshots.glob('*')):
            if (path/'config.json').is_file() and (path/'tokenizer_config.json').is_file() and any((path/n).is_file() for n in ('model.safetensors','pytorch_model.bin')):
                paper_model=path;break
    if paper_model is None or not Path(paper_model).is_dir():
        raise ValueError('missing local MedCPT cross encoder; refuse unvalidated paper reranking')
    os.environ['MEDGAP_MEDCPT_CROSS_ENCODER']=str(Path(paper_model).resolve())
    os.environ.update(MCP_CACHE_DIR=str(args.output/'tool_cache'),
        MEDGAP_RETRIEVAL_OUTPUT_DIR=str(args.output),
        MEDGAP_RETRIEVAL_OBSERVABILITY_DIR=str(args.output/'tool_observations'),
        MEDGAP_MINERU_CACHE_DIR=str(args.output/'mineru_cache'),
        MEDICAL_TOKENIZER_PATH=str(args.tokenizer),MEDGAP_RETRIEVAL_DEVICE='cpu',
        MEDGAP_V28_DEVICE='cpu',MEDGAP_PASSAGE_RETRIEVAL_MODE='hybrid',
        MEDGAP_EMBEDDING_MODEL=str(args.embedding_model),MEDGAP_RERANKER_MODEL=str(args.reranker_model),
        MEDGAP_BM25_TOP_N='8',MEDGAP_DENSE_TOP_N='8',MEDGAP_FUSION_MAX='12',
        MEDGAP_PAPER_PDF_ENABLED='1',MEDGAP_RETRIEVAL_LOCAL_FILES_ONLY='true',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYSERINI_DISABLE='1')


def safe_error_locations(exc, root):
    """Code locations only: no exception strings, source lines or frame locals."""
    import traceback
    root=Path(root).resolve()
    rows=[]
    for frame in traceback.extract_tb(exc.__traceback__):
        path=Path(frame.filename).resolve()
        if path.is_relative_to(root):
            rows.append(dict(file=path.relative_to(root).as_posix(),function=frame.name,line=frame.lineno))
    return rows[-12:]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--secrets',type=Path,default=Path.home()/'.config/medgap/secrets.env')
    for name in ('output','tokenizer','embedding-model','reranker-model'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--check-config',action='store_true')
    p.add_argument('--import-check',action='store_true')
    p.add_argument('--paper-reranker-model',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parent
    for name in ('tokenizer','embedding_model','reranker_model'):
        path=getattr(args,name).resolve()
        if not path.is_dir():raise ValueError('missing model directory: '+name)
        setattr(args,name,path)
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    keys=credentials(args.secrets)
    # Preserve stdout as the JSONL protocol. Redact logs as well as public results.
    secrets=[s for k,v in keys.items() if 'KEY' in k or 'TOKEN' in k for s in (v,quote(v,safe=''))]
    def redact(text):
        for value in sorted(secrets,key=len,reverse=True):text=text.replace(value,'[REDACTED]')
        return text
    original=sys.stdout
    class SafeStream:
        def write(self,text):return sys.__stderr__.write(redact(text))
        def flush(self):sys.__stderr__.flush()
    sys.stderr=SafeStream();sys.stdout=sys.stderr
    def emit(value):original.write(redact(json.dumps(value,ensure_ascii=False))+'\n');original.flush()
    os.environ.update(keys);configure(args)
    import trafilatura  # dependency preflight before any live action
    if args.check_config:
        emit(dict(status='config_checked',presence={k:bool(keys.get(k)) for k in sorted(KEYS)},
                  scope='paths_and_presence_only_no_network'));return
    if not all(keys.get(k) for k in ('NCBI_API_KEY','S2_API_KEY','SERPER_API_KEY','MINERU_API_TOKEN')):
        raise ValueError('missing PubMed/S2/web/MinerU credentials; refuse silent feature omission')
    sys.path.insert(0,str(root/'support'));sys.path.insert(0,str(root/'backend/agent'))
    import dotenv
    dotenv.load_dotenv=lambda *a,**kw:False
    from dr_agent import medical_source_metadata
    from engineering_v47 import install_metadata
    install_metadata(medical_source_metadata)
    from dr_agent.mcp_backend import main as backend
    from dr_agent.anchored_query import encode_anchored_query
    spec=importlib.util.spec_from_file_location('portable_common',root/'backend/common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    from web_region import call_search,call_browse
    from source_metadata_v58 import public_source_facts
    receipt_store=PairedReceiptStore(
        os.environ.get('MEDGAP_PAIRED_RECEIPT_DIR'),
        release_id=os.environ.get('MEDGAP_RETRIEVAL_RELEASE_ID'),
        mode=os.environ.get('MEDGAP_PAIRED_RECEIPT_MODE','off'))
    emit(dict(status='ready',paper='pubmed_s2_parallel',web='broad_native_snippets',pdf='mineru_enabled',
              passage='hybrid_bm25_bge_cross_encoder',reranker=str(args.reranker_model),history_reranker=str(args.reranker_model),paper_reranker=os.environ['MEDGAP_MEDCPT_CROSS_ENCODER'],scope='imports_only_not_live_validation'))
    if args.import_check:return
    rank_model = None
    from trainable_agent.retrieval_deadlines import (OperationJournal, RetrievalDeadlineExceeded, RetrievalIntegrityError,
        operation_deadline, bounded_timeout, reconcile_operation, receipt_digest, timeout_value)
    def rank_history(question, texts):
        nonlocal rank_model
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if rank_model is None:
            torch.set_num_threads(2)
            rank_model = (AutoTokenizer.from_pretrained(args.reranker_model, local_files_only=True),
                          AutoModelForSequenceClassification.from_pretrained(args.reranker_model, local_files_only=True).cpu().eval())
        tokenizer, model = rank_model
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start:start+16]
                encoded = tokenizer([question]*len(batch), batch, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt')
                scores.extend(model(**encoded).logits.float().reshape(-1).tolist())
        return scores
    for line in sys.stdin:
        request_id = None
        try:
            message=json.loads(line);request_id=message.get('request_id');operation=message['operation']
            arguments=message['arguments']
            operation_id=message.get('operation_id') or receipt_digest(message)
            journal=OperationJournal(args.output,operation_id,operation,arguments)
            recovered,mode=reconcile_operation(args.output,operation_id,operation,arguments)
            if recovered is not None:
                emit(dict(status='ok',value=recovered,request_id=request_id));continue
            if (journal.folder/'state.json').exists() and mode!='resume_same_task':
                raise ValueError('unknown_worker_operation_requires_reconciliation')
            journal.start()
            if operation=='search':execute=lambda:public_source_facts(call_search(common,backend,encode_anchored_query,**arguments))
            elif operation=='browse':
                from trainable_agent.context_processing.browser import opened_window_scope
                def execute():
                    options=dict(arguments)
                    opened=options.pop('opened_windows',[])
                    if any(row.get('source_id') != options['source_id'] for row in opened):
                        raise ValueError('opened_window_source_mismatch')
                    with opened_window_scope(opened):
                        return public_source_facts(call_browse(common,backend,encode_anchored_query,**options))
            elif operation=='rank_history':execute=lambda:rank_history(**arguments)
            else:raise ValueError('unsupported operation')
            try:
                with operation_deadline(message.get('deadline_seconds',360),journal.progress):
                    journal.progress('execute',{})
                    value,receipt=receipt_store.resolve(operation,arguments,execute)
                    bounded_timeout(.001)  # A backend catch cannot erase our expired deadline.
            except RetrievalDeadlineExceeded:
                if operation not in {'search','browse'}:
                    raise RuntimeError('retrieval_internal_operation_timeout') from None
                recovered,mode=reconcile_operation(args.output,operation_id,operation,arguments)
                value=recovered if recovered is not None else timeout_value(operation_id,'worker_operation',mode)
                receipt=None
            value=json.loads(redact(json.dumps(value,ensure_ascii=False)))
            journal.finish(value)
            response=dict(status='ok',value=value,request_id=request_id)
            if receipt is not None:response['paired_receipt']=receipt
            emit(response)
        except RetrievalIntegrityError as exc:
            emit(dict(status='worker_error',error_type='RetrievalIntegrityError',error_code=exc.error_code,
                      request_id=request_id,error_locations=safe_error_locations(exc,root.parent)))
        except Exception as exc:
            emit(dict(status='worker_error',error_type=type(exc).__name__,request_id=request_id,
                      error_locations=safe_error_locations(exc,root.parent)))

if __name__=='__main__':main()
