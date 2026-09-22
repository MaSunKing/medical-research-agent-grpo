# Adapted public research release; see THIRD_PARTY_NOTICES.md.
"""One explicit PDF switch for Search, paper/web Browse and parsing."""
import os
from urllib.parse import urlsplit, parse_qs

def pdf_enabled():
    return (os.getenv('MEDGAP_PAPER_PDF_ENABLED','0').strip().lower() in {'1','true','yes','on'}
            and bool(os.getenv('MINERU_API_TOKEN') or os.getenv('MEDGAP_MINERU_BASE_URL')))

def is_pdf_url(url):
    p=urlsplit(url)
    if p.path.lower().rstrip('/').endswith('.pdf'):return True
    return any(v.lower()=='pdf' or v.lower().endswith('/pdf')
               for k,vs in parse_qs(p.query).items() if k.lower() in {'format','type','output','download'}
               for v in vs)

def require_pdf_enabled():
    if not pdf_enabled():raise ValueError('pdf_disabled_or_service_unconfigured')
