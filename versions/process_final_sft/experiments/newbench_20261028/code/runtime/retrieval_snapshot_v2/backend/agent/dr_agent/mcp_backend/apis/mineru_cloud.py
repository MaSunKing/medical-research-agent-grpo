"""Versioned cloud cache and conservative submission ledger. No credential logs."""
import hashlib, json, os, re, time
from pathlib import Path
from trainable_agent.retrieval_deadlines import bounded_timeout, operation_progress, receipt_digest, RetrievalDeadlineExceeded, RetrievalIntegrityError
try:
    from . import mineru_cloud_legacy_v54 as legacy
except ImportError:
    import mineru_cloud as legacy

MinerUCloudError=legacy.MinerUCloudError
PROXY_PUBLIC_HOSTS=legacy.PROXY_PUBLIC_HOSTS
public_url=legacy.public_url

class Rejected(MinerUCloudError):
    """Server explicitly refused a request; no task ID was accepted."""


def local_record(path):
    """Malformed local state is not an unreadable source or remote API failure."""
    try:
        value=json.loads(path.read_text(encoding='utf8'))
    except (OSError,ValueError,UnicodeError):
        raise RetrievalIntegrityError('local_task_record_unreadable') from None
    if not isinstance(value,dict):
        raise RetrievalIntegrityError('local_task_record_invalid_schema')
    return value

def configure_shared_cache(root):
    """Import ONLY old ambiguous guards; never relabel old unversioned results."""
    root=Path(root)
    cache=Path(os.environ.setdefault('MEDGAP_MINERU_CACHE_DIR',str(root/'shared_mineru_cache_v54')))
    guards=cache/'url_guards';guards.mkdir(parents=True,exist_ok=True)
    # Known project layout, not a recursive scan of home/model/secret folders.
    for pattern in ('*/tool_cache/mineru_cloud/*/task.json','mineru_paper_probe*/cache/*/task.json'):
        for path in root.glob(pattern):
            row=json.loads(path.read_text(encoding='utf8'))
            if row.get('state') not in {'submitting','submission_uncertain'}:continue
            url=row.get('request_identity',{}).get('url')
            if not url:continue
            key=hashlib.sha256(url.encode()).hexdigest()
            with legacy.locked(guards/(key+'.lock')):
                target=guards/(key+'.json')
                if not target.exists():
                    legacy.atomic(target,{'state':'submission_uncertain','legacy_ledger':str(path)})
    return cache

class MinerUCloudClient(legacy.MinerUCloudClient):
    def __init__(self,token=None,cache_dir=None,timeout=None,session=None):
        super().__init__(token=token,cache_dir=cache_dir,timeout=timeout,session=session)
        if not cache_dir and not os.getenv('MEDGAP_MINERU_CACHE_DIR'):
            self.cache=Path.home()/'.cache'/'medgap'/'mineru_v54'
            self.cache.mkdir(parents=True,exist_ok=True)

    def api(self,method,url,**kwargs):
        if not self.enabled:raise Rejected('token_missing')
        poll_deadline=kwargs.pop('_poll_deadline',None)
        request_budget=30 if poll_deadline is None else max(.001,poll_deadline-time.monotonic())
        poll_budget_limited=poll_deadline is not None and request_budget<30
        try:
            r=self.session.request(method,url,headers={'Authorization':'Bearer '+self.token},
                                   timeout=bounded_timeout(min(30,request_budget)),allow_redirects=False,**kwargs)
            # Explicit client/auth/rate rejection. 408/5xx and malformed success
            # are ambiguous: a remote task may already exist.
            if r.status_code in {400,401,403,404,405,413,415,422,429}:
                raise Rejected('api_http_'+str(r.status_code))
            if r.status_code!=200:raise MinerUCloudError('api_http_'+str(r.status_code))
            body=r.json()
            if not isinstance(body,dict):raise MinerUCloudError('api_invalid_schema')
            if body.get('code') not in (None,0):raise Rejected('api_rejected')
            if body.get('code')!=0 or not isinstance(body.get('data'),dict):
                raise MinerUCloudError('api_invalid_schema')
            return body['data']
        except legacy.requests.Timeout:
            # Only a real transport timeout of the shortened poll request may
            # become the worker's own deadline. HTTP/JSON failures retain type.
            if poll_budget_limited and time.monotonic()>=poll_deadline:
                raise RetrievalDeadlineExceeded() from None
            raise MinerUCloudError('api_transport_timeout') from None
        except (legacy.requests.RequestException,ValueError):
            raise MinerUCloudError('api_transport_or_json_error') from None

    def pdf_version(self,url):
        # Bounded, TLS-verified fetch, with safe redirects; API token is never
        # sent to the document host. Hash bytes instead of trusting latest.pdf.
        operation_progress('pdf_version_fetch')
        blob=self.download(url)
        if len(blob)>int(os.getenv('MEDGAP_MINERU_MAX_PDF_BYTES',str(50*1024*1024))):
            raise MinerUCloudError('pdf_size_limit')
        if not blob.lstrip().startswith(b'%PDF-'):raise MinerUCloudError('input_not_pdf')
        return hashlib.sha256(blob).hexdigest()

    def parse_url(self,url):
        legacy.public_url(url)
        guards=self.cache/'url_guards';guards.mkdir(exist_ok=True)
        key=hashlib.sha256(url.encode()).hexdigest()
        guard=guards/(key+'.json')
        with legacy.locked(guards/(key+'.lock')):
            prior=local_record(guard) if guard.exists() else {}
            states={'submitting','submission_uncertain','pending','done','rejected','failed','source_changed'}
            if guard.exists() and prior.get('state') not in states:
                raise RetrievalIntegrityError('local_guard_state_invalid')
            if prior.get('state') in {'submitting','submission_uncertain'}:
                raise MinerUCloudError('saved_task_submission_uncertain')
            # Existing pending task always resumes its own ID/version. Never
            # submit a new version while an earlier task is still unresolved.
            pending=prior.get('state')=='pending'
            if pending:
                identity=prior.get('request_identity')
                if (not isinstance(identity,dict) or identity.get('url')!=url or identity.get('model_version')!=self.model
                        or identity.get('adapter_version')!='mineru_cloud_v54'
                        or not re.fullmatch(r'[a-f0-9]{64}',str(identity.get('pdf_sha256','')))):
                    raise RetrievalIntegrityError('pending_identity_invalid')
                if self.pdf_version(url)!=identity['pdf_sha256']:
                    raise MinerUCloudError('pending_document_version_changed')
            else:
                identity={'url':url,'model_version':self.model,'adapter_version':'mineru_cloud_v54',
                          'pdf_sha256':self.pdf_version(url)}
            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            entry=self.cache/digest;entry.mkdir(exist_ok=True)
            ledger=entry/'task.json';result=entry/'result.json'
            if result.exists():
                row=local_record(result)
                try:
                    md=(entry/'full.md').read_text(encoding='utf8')
                except (OSError,UnicodeError):
                    raise RetrievalIntegrityError('cache_content_unreadable') from None
                if hashlib.sha256(md.encode()).hexdigest()!=row.get('markdown_sha256'):
                    raise RetrievalIntegrityError('cache_integrity_failed')
                if row.get('request_identity')!=identity:raise RetrievalIntegrityError('cache_identity_failed')
                legacy.atomic(guard,{'state':'done','request_identity':identity})
                return md,{**row,'cache_hit':True,'cache_entry':str(entry)}
            state=local_record(ledger) if ledger.exists() else {}
            if ledger.exists() and state.get('state') not in states:
                raise RetrievalIntegrityError('local_task_state_invalid')
            if pending and (not state.get('task_id') or state.get('request_identity')!=identity):
                raise RetrievalIntegrityError('pending_ledger_missing_or_mismatched')
            if state.get('task_id') and (not isinstance(state['task_id'],str)
                    or not re.fullmatch(r'[a-zA-Z0-9-]{1,128}',state['task_id'])):
                raise RetrievalIntegrityError('saved_task_id_invalid')
            if state.get('state') in {'submitting','submission_uncertain','failed','source_changed'}:
                raise MinerUCloudError('saved_task_'+state['state'])
            if state.get('state')=='rejected' and not (
                os.getenv('MEDGAP_MINERU_RETRY_REJECTED')=='1' and time.time()-state.get('at',0)>=60):
                raise MinerUCloudError('saved_task_rejected_requires_explicit_retry')
            def save(row):
                row={**row,'request_identity':identity}
                legacy.atomic(ledger,row);legacy.atomic(guard,row)
            def progress(stage):
                output=os.getenv('MEDGAP_RETRIEVAL_OUTPUT_DIR')
                if not output:return
                path=ledger.resolve();output=Path(output).resolve()
                if not path.is_relative_to(output/'mineru_cache'):
                    raise RetrievalIntegrityError('task_ledger_outside_operation_output')
                operation_progress(stage,task_id=state.get('task_id'),task_state=state.get('state'),
                    pdf_sha256=identity['pdf_sha256'],resume_identity_sha256=receipt_digest(identity),
                    task_ledger=path.relative_to(output).as_posix(),model_version=self.model,
                    adapter_version=identity['adapter_version'])
            if not state.get('task_id'):
                save({'state':'submitting'})
                try:
                    data=self.api('POST',legacy.API,json={'url':url,'model_version':self.model,
                                  'language':'en','data_id':digest})
                    task=data.get('task_id','')
                    if not isinstance(task,str) or not re.fullmatch(r'[a-zA-Z0-9-]{1,128}',task):
                        raise MinerUCloudError('invalid_task_id')
                except Rejected as exc:
                    save({'state':'rejected','error':str(exc),'at':time.time()});raise
                except Exception:
                    save({'state':'submission_uncertain','error':'ambiguous_submission'});raise
                state={'state':'pending','task_id':task};save(state)
            progress('cloud_task_resume' if pending else 'cloud_task_submitted')
            deadline=time.monotonic()+self.timeout
            while time.monotonic()<deadline:
                bounded_timeout(.001)
                data=self.api('GET',legacy.API+'/'+state['task_id'],_poll_deadline=deadline)
                status=data.get('state')
                operation_progress('cloud_task_poll',task_state=status)
                if status=='failed':
                    save({**state,'state':'failed'});raise MinerUCloudError('cloud_parse_failed')
                if status=='done':
                    progress('cloud_result_download')
                    blob=self.download(data.get('full_zip_url',''))
                    md=legacy.markdown_from_zip(blob)
                    # URL API parses remotely, so verify it didn't change during
                    # the task. This cannot prove the remote fetch's byte hash.
                    if self.pdf_version(url)!=identity['pdf_sha256']:
                        save({**state,'state':'source_changed'});raise MinerUCloudError('pdf_changed_during_parse')
                    (entry/'full.md').write_text(md,encoding='utf8',newline='\n')
                    row={'request_identity':identity,'task_id':state['task_id'],
                         'markdown_sha256':hashlib.sha256(md.encode()).hexdigest(),
                         'document_version':'local_pdf_sha256_before_after_remote_fetch_unverified',
                         'cache_hit':False,'cache_entry':str(entry)}
                    legacy.atomic(result,row);save({**state,'state':'done'})
                    return md,row
                if status not in {'pending','running','converting'}:raise MinerUCloudError('unexpected_task_state')
                time.sleep(bounded_timeout(min(3,max(.001,deadline-time.monotonic()))))
            operation_progress('cloud_poll_timeout',task_state='pending')
            raise RetrievalDeadlineExceeded()
