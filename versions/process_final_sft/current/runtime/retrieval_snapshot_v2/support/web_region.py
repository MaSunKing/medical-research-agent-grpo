"""Question-explicit geography only; does not select the search tool."""
import re
import os
from contextlib import contextmanager
from urllib.parse import urlsplit
from types import SimpleNamespace

@contextmanager
def focused_web(backend, source_types='guideline'):
    """Synchronous worker only: no medical domain hints or query expansion."""
    search = getattr(backend, 'search_medical_web', None)
    namespace = getattr(search, '__globals__', {})
    previous = namespace.get('web_query_variants')
    mode = os.environ.get('MEDGAP_WEB_DISCOVERY_MODE')
    os.environ['MEDGAP_WEB_DISCOVERY_MODE'] = 'broad'
    if previous is not None:
        namespace['web_query_variants'] = lambda *args, **kwargs: []
    # Broad discovery keeps the backend public-URL safety checks unchanged.
    # Authority preferences belong to ranking, never to the safety predicate.
    try:
        yield
    finally:
        if previous is not None: namespace['web_query_variants'] = previous
        if mode is None: os.environ.pop('MEDGAP_WEB_DISCOVERY_MODE', None)
        else: os.environ['MEDGAP_WEB_DISCOVERY_MODE'] = mode

def region(question):
    patterns={'uk':r'\b(?:UK|United Kingdom|Britain|British)\b|英国',
              'us':r'\b(?:US|USA|United States)\b|美国',
              'sg':r'\bSingapore\b|新加坡',
              'au':r'\bAustralia\b|澳大利亚',
              'ca':r'\bCanada\b|加拿大'}
    found=[code for code,p in patterns.items() if re.search(p,question,re.I)]
    return found[0] if len(found)==1 else 'us'

def call_search(common,backend,anchored,**arguments):
    if arguments['tool']!='medical_web_search':
        return common.call_search(backend,anchored,**arguments)
    gl=region(arguments['question'])
    def web(**kwargs):
        kwargs['gl']=gl
        with focused_web(backend, kwargs.get('source_types','guideline')):
            return common.run_callable(backend.medical_web_search,**kwargs)
    proxy=SimpleNamespace(medical_web_search=web)
    return common.call_search(proxy,anchored,**arguments)

def call_browse(common, backend, anchored, **arguments):
    # Browse must use the same URL policy as the Search that registered its ID.
    # The backend still validates the registered ID, URL safety and content.
    if arguments['search_tool'] != 'medical_web_search':
        return common.call_browse(backend, anchored, **arguments)
    with focused_web(backend):
        return common.call_browse(backend, anchored, **arguments)
