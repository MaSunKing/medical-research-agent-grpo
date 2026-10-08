# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Versioned cloud cache and conservative submission ledger. No credential logs."""
import hashlib, json, os, re, time
from pathlib import Path
try:
    from . import mineru_cloud_legacy_v54 as legacy
except ImportError:
    import mineru_cloud as legacy

MinerUCloudError=legacy.MinerUCloudError
PROXY_PUBLIC_HOSTS=legacy.PROXY_PUBLIC_HOSTS
public_url=legacy.public_url

class Rejected(MinerUCloudError):
    """Server explicitly refused a request; no task ID was accepted."""

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
        try:
            r=self.session.request(method,url,headers={'Authorization':'Bearer '+self.token},
                                   timeout=30,allow_redirects=False,**kwargs)
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
        except (legacy.requests.RequestException,ValueError):
            raise MinerUCloudError('api_transport_or_json_error') from None

    def pdf_version(self,url):
        # Bounded, TLS-verified fetch, with safe redirects; API token is never
        # sent to the document host. Hash bytes instead of trusting latest.pdf.
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
            prior=json.loads(guard.read_text()) if guard.exists() else {}
            if prior.get('state') in {'submitting','submission_uncertain'}:
                raise MinerUCloudError('saved_task_submission_uncertain')
            # Existing pending task always resumes its own ID/version. Never
            # submit a new version while an earlier task is still unresolved.
            pending=prior.get('state')=='pending'
            if pending:
                identity=prior['request_identity']
            else:
                identity={'url':url,'model_version':self.model,'adapter_version':'mineru_cloud_v54',
                          'pdf_sha256':self.pdf_version(url)}
            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            entry=self.cache/digest;entry.mkdir(exist_ok=True)
            ledger=entry/'task.json';result=entry/'result.json'
            if result.exists():
                row=json.loads(result.read_text())
                md=(entry/'full.md').read_text(encoding='utf8')
                if hashlib.sha256(md.encode()).hexdigest()!=row['markdown_sha256']:
                    raise MinerUCloudError('cache_integrity_failed')
                if row.get('request_identity')!=identity:raise MinerUCloudError('cache_identity_failed')
                legacy.atomic(guard,{'state':'done','request_identity':identity})
                return md,{**row,'cache_hit':True,'cache_entry':str(entry)}
            state=json.loads(ledger.read_text()) if ledger.exists() else {}
            if pending and (not state.get('task_id') or state.get('request_identity')!=identity):
                raise MinerUCloudError('pending_ledger_missing_or_mismatched')
            if state.get('state') in {'submitting','submission_uncertain','failed','source_changed'}:
                raise MinerUCloudError('saved_task_'+state['state'])
            if state.get('state')=='rejected' and not (
                os.getenv('MEDGAP_MINERU_RETRY_REJECTED')=='1' and time.time()-state.get('at',0)>=60):
                raise MinerUCloudError('saved_task_rejected_requires_explicit_retry')
            def save(row):
                row={**row,'request_identity':identity}
                legacy.atomic(ledger,row);legacy.atomic(guard,row)
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
            deadline=time.monotonic()+self.timeout
            while time.monotonic()<deadline:
                data=self.api('GET',legacy.API+'/'+state['task_id'])
                status=data.get('state')
                if status=='failed':
                    save({**state,'state':'failed'});raise MinerUCloudError('cloud_parse_failed')
                if status=='done':
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
                time.sleep(min(3,max(0,deadline-time.monotonic())))
            raise MinerUCloudError('task_pending_resume_same_id')
