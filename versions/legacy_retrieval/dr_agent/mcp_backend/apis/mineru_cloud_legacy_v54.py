# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""Official MinerU cloud adapter; bounded polling, pinned cache, no secret logging."""
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import time
from urllib.parse import urlsplit, urljoin
import zipfile
import requests
try:
    from .s2_transport import locked
except ImportError:
    from s2_transport import locked

API='https://mineru.net/api/v4/extract/task'
VERSION='mineru_cloud_v1'
# Explicit public origins used by this integration. Local TUN proxies may
# resolve them to RFC2544 fake IPs; TLS certificate validation remains enabled.
PROXY_PUBLIC_HOSTS={'www.researchsquare.com','researchsquare.com','cdn-mineru.openxlab.org.cn'}
PROXY_PUBLIC_HOSTS |= {'www.nice.org.uk', 'nice.org.uk'}  # V53 explicit public origin

class MinerUCloudError(RuntimeError):
    pass

def public_url(url):
    p=urlsplit(url)
    if p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in (None,443):
        raise MinerUCloudError('invalid_public_https_url')
    try:
        addresses=socket.getaddrinfo(p.hostname,443,type=socket.SOCK_STREAM)
    except OSError:
        raise MinerUCloudError('public_url_dns_failed') from None
    ips=[ipaddress.ip_address(a[4][0]) for a in addresses]
    proxy_public=p.hostname in PROXY_PUBLIC_HOSTS and all(
        ip.version==4 and ip in ipaddress.ip_network('198.18.0.0/15') for ip in ips)
    if not ips or (not proxy_public and any(not ip.is_global for ip in ips)):
        raise MinerUCloudError('non_public_url')
    return url

def markdown_from_zip(blob):
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            entries=z.infolist()
            if len(entries)>10000 or sum(x.file_size for x in entries)>512*1024*1024:
                raise MinerUCloudError('oversized_archive')
            for x in entries:
                p=PurePosixPath(x.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in x.orig_filename:
                    raise MinerUCloudError('unsafe_archive_path')
            full=[x for x in entries if PurePosixPath(x.filename).name=='full.md']
            if len(full)!=1 or full[0].file_size>20*1024*1024:
                raise MinerUCloudError('missing_or_ambiguous_markdown')
            text=z.read(full[0]).decode('utf-8-sig')
            if not text.strip():raise MinerUCloudError('empty_markdown')
            return text
    except (zipfile.BadZipFile,UnicodeError):
        raise MinerUCloudError('invalid_result_archive') from None

def atomic(path,value):
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

class MinerUCloudClient:
    def __init__(self, token=None, cache_dir=None, timeout=None, session=None):
        self.token=token or os.getenv('MINERU_API_TOKEN','')
        self.cache=Path(cache_dir or os.getenv('MEDGAP_MINERU_CACHE_DIR') or
                        Path(os.getenv('MCP_CACHE_DIR','.'))/'mineru_cloud')
        self.cache.mkdir(parents=True,exist_ok=True)
        self.timeout=float(timeout or os.getenv('MEDGAP_MINERU_TIMEOUT','300'))
        self.session=session or requests.Session()
        self.model=os.getenv('MEDGAP_MINERU_MODEL','vlm')
        if self.model not in ('vlm','pipeline'):raise MinerUCloudError('unsupported_model')

    @property
    def enabled(self):return bool(self.token)

    def api(self,method,url,**kwargs):
        if not self.enabled:raise MinerUCloudError('token_missing')
        try:
            r=self.session.request(method,url,headers={'Authorization':'Bearer '+self.token},
                                   timeout=30,allow_redirects=False,**kwargs)
            if r.status_code!=200:raise MinerUCloudError('api_http_'+str(r.status_code))
            body=r.json()
            if not isinstance(body,dict) or body.get('code')!=0:
                raise MinerUCloudError('api_rejected')
            if not isinstance(body.get('data'),dict):raise MinerUCloudError('api_invalid_schema')
            return body['data']
        except (requests.RequestException,ValueError):
            raise MinerUCloudError('api_transport_or_json_error') from None

    def download(self,url):
        # CDN never receives the API token, including on redirects.
        for _ in range(4):
            public_url(url)
            try:
                with requests.get(url,timeout=45,stream=True,allow_redirects=False) as r:
                    if r.status_code in (301,302,303,307,308):
                        url=urljoin(url,r.headers.get('Location',''));continue
                    if r.status_code!=200:raise MinerUCloudError('archive_http_'+str(r.status_code))
                    out=bytearray()
                    for chunk in r.iter_content(1024*1024):
                        out.extend(chunk)
                        if len(out)>100*1024*1024:raise MinerUCloudError('archive_size_limit')
                    return bytes(out)
            except requests.RequestException:
                raise MinerUCloudError('archive_transport_failed') from None
        raise MinerUCloudError('archive_redirect_limit')

    def parse_url(self,url):
        public_url(url)
        request_identity={'url':url,'model_version':self.model,'adapter_version':VERSION}
        digest=hashlib.sha256(json.dumps(request_identity,sort_keys=True).encode()).hexdigest()
        entry=self.cache/digest;entry.mkdir(exist_ok=True)
        ledger=entry/'task.json';result=entry/'result.json'
        with locked(entry/'task.lock'):
            if result.exists():
                row=json.loads(result.read_text(encoding='utf-8'))
                md=(entry/'full.md').read_text(encoding='utf-8')
                if hashlib.sha256(md.encode()).hexdigest()!=row['markdown_sha256']:
                    raise MinerUCloudError('cache_integrity_failed')
                return md,{**row,'cache_hit':True}
            state=json.loads(ledger.read_text()) if ledger.exists() else {}
            if state.get('state') in ('submission_uncertain','submitting','failed'):
                raise MinerUCloudError('saved_task_'+state['state'])
            if not state.get('task_id'):
                atomic(ledger,{'state':'submitting','request_identity':request_identity})
                try:
                    data=self.api('POST',API,json={'url':url,'model_version':self.model,
                                                  'language':'en','data_id':digest})
                    task=data.get('task_id','')
                    if not re.fullmatch(r'[a-zA-Z0-9-]{1,128}',task):
                        raise MinerUCloudError('invalid_task_id')
                except Exception:
                    # No automatic duplicate charge after an ambiguous submit.
                    atomic(ledger,{'state':'submission_uncertain','request_identity':request_identity})
                    raise
                state={'task_id':task,'state':'pending','request_identity':request_identity}
                atomic(ledger,state)
            deadline=time.monotonic()+self.timeout
            while time.monotonic()<deadline:
                data=self.api('GET',API+'/'+state['task_id'])
                status=data.get('state')
                if status=='failed':
                    atomic(ledger,{**state,'state':'failed'})
                    raise MinerUCloudError('cloud_parse_failed')
                if status=='done':
                    blob=self.download(data.get('full_zip_url',''))
                    md=markdown_from_zip(blob)
                    (entry/'result.zip').write_bytes(blob)
                    (entry/'full.md').write_text(md,encoding='utf-8',newline='\n')
                    row={**request_identity,'task_id':state['task_id'],
                         'markdown_sha256':hashlib.sha256(md.encode()).hexdigest(),
                         'zip_sha256':hashlib.sha256(blob).hexdigest(),'cache_hit':False,
                         'document_version':'unverified_pdf_version'}
                    atomic(result,row);atomic(ledger,{**state,'state':'done'})
                    return md,row
                if status not in ('pending','running','converting'):
                    raise MinerUCloudError('unexpected_task_state')
                time.sleep(min(3,max(0,deadline-time.monotonic())))
            raise MinerUCloudError('task_pending_resume_same_id')
